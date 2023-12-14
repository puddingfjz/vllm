# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata,
                                cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)



        # <jingzhi> to support loading parameters layer by layer
        # Initialize the stream for loading parameters.
        self.param_stream = torch.cuda.Stream()
        assert self.param_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.load_param_events = [torch.cuda.Event() for _ in self.layers]
        self.layer_waiting_for_params = [False for _ in self.layers]
        self.layer_num = len(self.layers)


        # <jingzhi> For offloading weights layer by layer
        # compute the size of parameters for each layer
        self.weight_cache: torch.Tensor = torch.Tensor([])
        self.weight_cache_cpu: torch.Tensor = torch.Tensor([])
        self.weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list() # (int, int, size_tuple): (start position, offset, tensor shape)
        self.weight_num_per_layer: int = -1
        # because of the high memory transfer cost: almost 1/2 decoding time per iter, only allows 1 layer to be transferred
        self.weight_cache_block_num: int = self.layer_num # self.layer_num - 1 # (self.layer_num + 1)//2
        # example: if totally 5 layers, the cache blk num is 3, then l0 -> blk0, l1 -> blk1, l2 -> blk2, l3->blk0, l4->blk1
        # self.weight_cache_block_idx: List[int] = list(range(self.weight_cache_block_num)) \
        #     + list(range(self.layer_num - self.weight_cache_block_num))
        # example: 5 layers, 4 param cache blk, l0->blk0, l1->blk1, l2-blk2, l3->blk0, l4->blk3
        # self.weight_cache_block_idx: List[int] = list(range((self.layer_num + 1)//2)) + [0] + list(range((self.layer_num + 1)//2, self.layer_num-1))
        # no weight loading during inference
        self.weight_cache_block_idx: List[int] = list(range(self.layer_num))
        # stores the parameters we will store in self.weight_cache in order.
        self.layer_params = [list() for _ in self.layers]




    # the original code from vllm with all parameters on GPU
    def forward_ori(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            cache_event = None if cache_events is None else cache_events[i]
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states




    # <jingzhi> In this function, we will try offloading 
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):

            # waiting for the param loading if there is
            # loading weight -----------------------------------------------------------------------------------------
            # param_event = self.load_param_events[i] if self.layer_waiting_for_params[i] else None
            # if param_event is not None:
            #     param_event.wait()
            #     # update the status of this layer params
            #     self.layer_waiting_for_params[i] = False
            # loading weight END -----------------------------------------------------------------------------------------



            # print(f'in computation for layer {i}:')
            # for param in self.layer_params[i]:
            #     print(param)


            cache_event = None if cache_events is None else cache_events[i]
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
                residual,
            )


            # loading weight -----------------------------------------------------------------------------------------
            # # <jingzhi> after finishing this layer, start loading the params of a waiting layer
            # self.load_layer_params((i + self.weight_cache_block_num)%self.layer_num, self.weight_cache_block_idx[i], i)
            # # currently just a fixed mode, may consider dynamic weight_cache space later
            # self.layer_waiting_for_params[ (i + self.weight_cache_block_num)%self.layer_num ] = True

            # currently only allow one layer on CPU
            # if i in [0, ((self.layer_num + 1)//2)]:
            #     self.load_layer_params(((self.layer_num + 1)//2)-i, self.weight_cache_block_idx[i], i)
            #     self.layer_waiting_for_params[ ((self.layer_num + 1)//2)-i ] = True
            # loading weight END -----------------------------------------------------------------------------------------



        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states




    def load_layer_params(self, layer_i:int, dst_weight_blk_idx:int, replace_layer_i:int):
        param_event = self.load_param_events[layer_i]

        # print(f"to load later {layer_i} into layer {replace_layer_i}, dst_weight_blk_idx: {dst_weight_blk_idx}")
        # print(self.weight_cache_cpu[layer_i])

        with torch.cuda.stream(self.param_stream):
            self.weight_cache[dst_weight_blk_idx].copy_(self.weight_cache_cpu[layer_i], non_blocking=True)
            param_event.record(stream=self.param_stream)


        # if only allow one layer on CPU, no need to update the param tensor address
        return

        # update param tensor address if necessary
        if self.layer_num % self.weight_cache_block_num == 0:
            return

        # change the param tensor address
        self.weight_cache_block_idx[layer_i] = dst_weight_blk_idx
        for param, to_replace in zip(self.layer_params[layer_i], self.layer_params[replace_layer_i]):
            param.data = to_replace.data







class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = LlamaModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights_ori(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            # <jingzhi> For Profiling
            print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # <jingzhi> For Profiling
                print(f"specific weight loader: {name.replace(weight_name, param_name), loaded_weight.shape, loaded_weight.device}")
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                # <jingzhi> For Profiling
                print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)









    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):

        # <jingzhi> For offloading weights layer by layer
        # first do some preprocess
        self.preprocess_for_offloading_weights(model_name_or_path, cache_dir, load_format, revision)


        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            # <jingzhi> For Profiling
            print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # <jingzhi> For Profiling
                print(f"specific weight loader: {name.replace(weight_name, param_name), loaded_weight.shape, loaded_weight.device}")
                
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) < self.model.weight_cache_block_num):
                # currently only allow one layer on CPU
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) != ((self.model.layer_num + 1)//2)):
                if True:
                    print('loading-----')
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)

                    # print(f"layerBylayer_param_tensor of {name}: {param.data}")

                break
            else:

                # <jingzhi> For Profiling
                print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) < self.model.weight_cache_block_num):
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) != ((self.model.layer_num + 1)//2)):
                if True:
                    print('loading-----')
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

                    # print(f"layerBylayer_param_tensor of {name}: {param.data}")









    # <jingzhi> For offloading weights layer by layer
    # Set up: (1) weight_num_per_layer, (2) weight_range_in_blk, (3) weight cache blocks
    # (4) making the param tensors pointing to the new GPU physical addresses

    # NOTE: we cannot assume the parameter tensors for the same layer will appear together;
    #       we cannot assume the parameter tensors for different layers appear in the same order

    def preprocess_for_offloading_weights(self,
             model_name_or_path: str,
             cache_dir: Optional[str] = None,
             load_format: str = "auto",
             revision: Optional[str] = None):


        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())


        visited_param_names = [list() for _ in self.model.layers]

        print(f"init model size: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        # we only need to consider layer params.
        # we do weight loading.
        # we collect layer parameters in order (no repetition).
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            # <jingzhi> For Profiling
            # print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

            if 'model.layers.' not in name:
                # we only need to consider layer params
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # <jingzhi> For Profiling
                # print(f"specific weight loader: {name.replace(weight_name, param_name), loaded_weight.shape, loaded_weight.device}")

                full_para_name = name.replace(weight_name, param_name)
                param = params_dict[full_para_name]
                # need load weight so that we can get packed weight_cache_cpu
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                # print(f"ori_param_tensor of {name}: {param.data}")

                # store this parameter
                layer_i = self.get_layer_id_from_name(name)
                if full_para_name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(full_para_name)
                break
            else:

                # <jingzhi> For Profiling
                # print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

                # print(f"ori_param_tensor of {name}: {param.data}")

                # store this parameter
                layer_i = self.get_layer_id_from_name(name)
                if name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(name)
                


        # we need to ensure the params and param_names for different layers are in the same order
        # name: 'model.layers.30.input_layernorm.weight'
        for layer_i in range(self.model.layer_num):
            param_names = [name.split('.')[3:] for name in visited_param_names[layer_i]]
            order = sorted(range(len(param_names)), key=lambda i: param_names[i])
            params = self.model.layer_params[layer_i]
            self.model.layer_params[layer_i] = [params[i] for i in order]
            visited_param_names[layer_i] = [visited_param_names[layer_i][i] for i in order]



        weight_num_per_layer = 0
        weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list()
        params_dtype = None

        for param in self.model.layer_params[0]:
            # for these information we only need to consider layer 0, as other layers are the same as it
            # we also avoid repeat param as some weights may be packed into one param
            weight_range_in_blk.append((weight_num_per_layer, param.nelement(), param.shape))
            weight_num_per_layer += param.nelement()
            assert (params_dtype == None) or (params_dtype == param.dtype)
            params_dtype = param.dtype


        print(f"visited_param_names[30]: {visited_param_names[30]}")
        print(f"weight_range_in_blk: {weight_range_in_blk}")
        print(f"weight_num_per_layer: {weight_num_per_layer}")


        # obtain weights_cache_cpu and release the param tensors on GPU
        # make sure weight_cache_cpu is a pinned memory
        self.model.weight_cache_cpu = torch.empty(weight_num_per_layer*self.model.layer_num, dtype=self.model.layer_params[0][0].data.cpu().dtype, pin_memory=True)
        torch.cat([param.data.cpu().view(-1) for params in self.model.layer_params for param in params], out=self.model.weight_cache_cpu)
        self.model.weight_cache_cpu = self.model.weight_cache_cpu.view(self.model.layer_num, weight_num_per_layer)
        for params in self.model.layer_params:
            for param in params:
                param.data = torch.Tensor([])

        torch.cuda.empty_cache()
        print(f"release param tensors: {torch.cuda.memory_allocated()/1024/1024/1024} GB")

        self.model.weight_num_per_layer = weight_num_per_layer
        self.model.weight_range_in_blk = weight_range_in_blk
        self.model.weight_cache = torch.empty(self.model.weight_cache_block_num * self.model.weight_num_per_layer,
                                       device=torch.cuda.current_device(),
                                       dtype=params_dtype)

        
        print(f"after allocate param cache: {torch.cuda.memory_allocated()/1024/1024/1024} GB")

        # make the layer param tensors point to the new address in self.weight_cache
        for layer_i in range(len(self.model.layer_params)):
            params = self.model.layer_params[layer_i]
            for param, rng_info in zip(params, self.model.weight_range_in_blk):
                param.data = torch.narrow(self.model.weight_cache, 
                    0, 
                    self.model.weight_cache_block_idx[layer_i]*self.model.weight_num_per_layer + rng_info[0], 
                    rng_info[1]).view(rng_info[2])


        self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, self.model.weight_num_per_layer)

        torch.cuda.empty_cache()



    def get_layer_id_from_name(self, name:str) -> int:
        # 'model.layers.24.input_layernorm.weight'
        terms = name.split('.')
        return int(terms[2])