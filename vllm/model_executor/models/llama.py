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




# import ray
# import ray.util.collective as collective

# <jingzhi>
from vllm._C import cache_ops
import time
import os




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


        print(f"tp_size in LlamaAttention: {tp_size}")

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



        # ======================================================================

        # below are only used when we consider weights offloading


        # <jingzhi> to support loading parameters layer by layer
        self.cache_device_ids = [1] # example: [1] means we use GPU 1 as our cache
        # consider when there are multiple workers due to parallelism
        worker_num = torch.distributed.get_world_size()
        cache_gpu_num = (torch.cuda.device_count() - worker_num) // worker_num
        # we need do cuda order remapping because ray would mess it up
        # self.cache_device_ids = list(range(worker_num+torch.cuda.current_device()*cache_gpu_num, worker_num+(torch.cuda.current_device()+1)*cache_gpu_num))
        if worker_num == 1:
            self.cache_device_ids = list(range(worker_num+torch.cuda.current_device()*cache_gpu_num, worker_num+(torch.cuda.current_device()+1)*cache_gpu_num))
        else:
            cand_cache_device_names = os.environ['TOT_ORDERED_GPUS'].split(',')[worker_num:]
            cand_cache_device_ids = list()
            current_device_names_ordered = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            for cuda_name in cand_cache_device_names:
                for i, to_check in enumerate(current_device_names_ordered):
                    if cuda_name == to_check:
                        cand_cache_device_ids.append(i)
                        break
            self.cache_device_ids = cand_cache_device_ids[int(os.environ["LOCAL_RANK"])*cache_gpu_num:(int(os.environ["LOCAL_RANK"])+1)*cache_gpu_num]

        print(f'In model, os.environ["LOCAL_RANK"]:{os.getenv("LOCAL_RANK", "0")}, all card num: {torch.cuda.device_count()}, worker_num:{worker_num}, os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}, os.environ["TOT_ORDERED_GPUS"]:{os.environ["TOT_ORDERED_GPUS"]}  current_device: {torch.cuda.current_device()}, self.cache_device_ids: {self.cache_device_ids}')

        # Initialize the stream for loading parameters.
        # self.param_stream = torch.cuda.Stream()
        # assert self.param_stream != torch.cuda.current_stream()
        # prepare a stream for each cache GPU
        self.param_streams = [torch.cuda.Stream() for _ in self.cache_device_ids]
        for param_stream in self.param_streams:
            assert param_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.load_param_events = [[torch.cuda.Event() for _ in self.layers] for _ in self.cache_device_ids]
        self.layer_waiting_for_params = [False for _ in self.layers]
        self.layer_num = len(self.layers)


        # <jingzhi> For offloading weights layer by layer
        # compute the size of parameters for each layer
        self.weight_cache: torch.Tensor = torch.Tensor([])
        # self.weight_cache_cpu: torch.Tensor = torch.Tensor([])
        self.weight_cache_cpu: List[List[torch.Tensor]] = []
        self.weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list() # (int, int, size_tuple): (start position, offset, tensor shape)
        self.weight_num_per_layer: int = -1
        
        # because of the high memory transfer cost: almost 1/2 decoding time per iter, only allows 1 layer to be transferred
        # self.weight_cache_block_num: int = self.layer_num - 1 # (self.layer_num + 1)//2
        # we set a parameter: pipeline degree to control weight_cache_block_num
        self.pipeline_degree: int = 40 # 1 iter decoding time = pipeline_degree * 1 layer weight loading time
        self.pipeline_inteval = (self.layer_num + self.pipeline_degree - 1) // self.pipeline_degree # get ceiling value to ensure enough pipeline interval
        self.pipeline_degree = (self.layer_num + self.pipeline_inteval - 1) // self.pipeline_inteval # get the interval number
        self.weight_cache_block_num = self.layer_num - (self.pipeline_degree - 1) # assuming all layers involved in weight offloading will take the same weight cache block
        self.weight_cache_block_num += 1 # use another weight cache block so that we do not need to wait the current layer to finish

        # example: if totally 5 layers, the cache blk num is 3, then l0 -> blk0, l1 -> blk1, l2 -> blk2, l3->blk0, l4->blk1
        # self.weight_cache_block_idx: List[int] = list(range(self.weight_cache_block_num)) \
        #     + list(range(self.layer_num - self.weight_cache_block_num))
        # example: 5 layers, 4 param cache blk, l0->blk0, l1->blk1, l2-blk2, l3->blk0, l4->blk3
        # self.weight_cache_block_idx: List[int] = list(range((self.layer_num + 1)//2)) + [0] + list(range((self.layer_num + 1)//2, self.layer_num-1))
        # no weight loading during inference
        # self.weight_cache_block_idx: List[int] = list(range(self.layer_num))
        # compute the weight_cache_block_idx based on weight_cache_block_num automatically
        self.weight_cache_block_idx: List[int] = [0] * self.layer_num
        # example: layer_num=5 pipeline_degree=2 pipeline_interval=3 then weight_cache_block_idx=[0,1,2]+[0,3]
        for interval_i in range(self.pipeline_degree):
            self.weight_cache_block_idx[ interval_i*self.pipeline_inteval:(interval_i+1)*self.pipeline_inteval ] =\
                range(interval_i*self.pipeline_inteval - interval_i, (interval_i+1)*self.pipeline_inteval - interval_i)
            # self.weight_cache_block_idx[ interval_i*self.pipeline_inteval] = 0
            self.weight_cache_block_idx[ interval_i*self.pipeline_inteval] = (interval_i % 2) * (self.weight_cache_block_num - 1)


        self.buffer_params = dict() # stores the params in the weight cache blocks which we use to load weights during inference


        # stores the parameters we will store in self.weight_cache in order.
        self.layer_params = [list() for _ in self.layers]

        self.use_vllm = (os.environ['USE_VLLM'] == 'True')
        # ======================================================================



    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        if self.use_vllm:
            return self.forward_ori(input_ids, positions, kv_caches, input_metadata, cache_events)
        else:
            return self.forward_ours(input_ids, positions, kv_caches, input_metadata, cache_events)



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
    def forward_ours(
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

            # loading weight -----------------------------------------------------------------------------------------
            # param_event = self.load_param_events[i] if self.layer_waiting_for_params[i] else None
            # if param_event is not None:
            #     param_event.wait()
            #     # update the status of this layer params
            #     self.layer_waiting_for_params[i] = False
            # 

            # we can directly load the next layer to be loaded directly without waiting for this layer to finish its computation
            if (i%self.pipeline_inteval) == 0:
                # self.load_layer_params(((self.layer_num + 1)//2)-i, self.weight_cache_block_idx[i], i)
                to_load_layer_i = ((i//self.pipeline_inteval+1)%self.pipeline_degree)*self.pipeline_inteval
                self.load_layer_params(to_load_layer_i, i)
                self.layer_waiting_for_params[ to_load_layer_i ] = True

            # waiting for the param loading if there is
            if self.layer_waiting_for_params[i]:
                for device_events in self.load_param_events:
                    device_events[i].wait()
                # update the status of this layer params
                self.layer_waiting_for_params[i] = False                
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

            # if (i%self.pipeline_inteval) == 0:
            #     # self.load_layer_params(((self.layer_num + 1)//2)-i, self.weight_cache_block_idx[i], i)
            #     to_load_layer_i = ((i//self.pipeline_inteval+1)%self.pipeline_degree)*self.pipeline_inteval
            #     self.load_layer_params(to_load_layer_i, self.weight_cache_block_idx[i], i)
            #     self.layer_waiting_for_params[ to_load_layer_i ] = True
            # loading weight END -----------------------------------------------------------------------------------------



        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states




    def load_layer_params_deprecated(self, layer_i:int, dst_weight_blk_idx:int, replace_layer_i:int):
        # param_event = self.load_param_events[layer_i]

        # # print(f"to load later {layer_i} into layer {replace_layer_i}, dst_weight_blk_idx: {dst_weight_blk_idx}")
        # # print(self.weight_cache_cpu[layer_i])

        # print(f"loading layer weights: {self.weight_cache_cpu[layer_i][0].element_size(), self.weight_cache_cpu[layer_i][0].numel()}, {len(self.weight_cache_cpu), len(self.weight_cache_cpu[layer_i]), self.weight_cache_cpu[layer_i][0].shape, self.weight_cache_cpu[layer_i][0].dtype}")

        # with torch.cuda.stream(self.param_stream):
        #     # self.weight_cache[dst_weight_blk_idx].copy_(self.weight_cache_cpu[layer_i], non_blocking=True)
        #     # use nvlink to copy from another device
        #     for part_i, cache_device_i in enumerate(self.cache_device_ids):
        #         cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.weight_cache[dst_weight_blk_idx][part_i],
        #             layer_i, cache_device_i, 0, 0)
        #     param_event.record(stream=self.param_stream)

        # consider we have multiple GPUs as cache============================================================
        
        # print(f"loading layer weights: {self.weight_cache_cpu[layer_i][0].element_size(), self.weight_cache_cpu[layer_i][0].numel()}, {len(self.weight_cache_cpu), len(self.weight_cache_cpu[layer_i]), self.weight_cache_cpu[layer_i][0].shape, self.weight_cache_cpu[layer_i][0].dtype}")

        for part_i, cache_device_i in enumerate(self.cache_device_ids):
            param_event = self.load_param_events[part_i][layer_i]
            # use nvlink to copy from another device
            with torch.cuda.stream(self.param_streams[part_i]):
                cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.weight_cache[dst_weight_blk_idx][part_i],
                    layer_i, cache_device_i, 0, 0)
                param_event.record(stream=self.param_streams[part_i])


        # if only allow one layer on CPU, no need to update the param tensor address
        return

        # update param tensor address if necessary
        if self.layer_num % self.weight_cache_block_num == 0:
            return

        # change the param tensor address
        self.weight_cache_block_idx[layer_i] = dst_weight_blk_idx
        for param, to_replace in zip(self.layer_params[layer_i], self.layer_params[replace_layer_i]):
            param.data = to_replace.data






    def load_layer_params(self, layer_i:int, last_layer_i:int):

        # consider we have multiple GPUs as cache============================================================
        
        # print(f"loading layer weights: {self.weight_cache_cpu[layer_i][0].element_size(), self.weight_cache_cpu[layer_i][0].numel()}, {len(self.weight_cache_cpu), len(self.weight_cache_cpu[layer_i]), self.weight_cache_cpu[layer_i][0].shape, self.weight_cache_cpu[layer_i][0].dtype}")

        dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]
        last_layer_weight_blk_idx = self.weight_cache_block_idx[last_layer_i]
        if dst_weight_blk_idx == last_layer_weight_blk_idx:
            # we need to load layer weight in another buffer
            self.weight_cache_block_idx[layer_i] = self.weight_cache_block_num - 1 - dst_weight_blk_idx
            dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]

            for param, to_replace in zip(self.layer_params[layer_i], self.buffer_params[dst_weight_blk_idx]):
                param.data = to_replace.data


        # loading
        for part_i, cache_device_i in enumerate(self.cache_device_ids):
            param_event = self.load_param_events[part_i][layer_i]
            # use nvlink to copy from another device
            with torch.cuda.stream(self.param_streams[part_i]):
                # in distributed inference, the current device may not be cuda:0
                cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.weight_cache[dst_weight_blk_idx][part_i],
                    layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())
                param_event.record(stream=self.param_streams[part_i])


        return






    # def init_param_loaders(self):
    #     # imperative
    #     num_workers = 2
    #     workers = []
    #     init_rets = []

    #     # maybe the first solution to do initialization?
    #     # for i in range(num_workers):
    #     #    w = ParamLoader.remote()
    #     #    workers.append(w)
    #     #    init_rets.append(w.setup.remote(num_workers, i))
    #     # _ = ray.get(init_rets)


    #     # maybe the second solution to do initialization?
    #     workers = [0, 0]
    #     for i in range(num_workers):
    #         w = ParamLoader.remote([self.weight_cache, self.weight_cache_spare_gpu])
    #         workers[w.gpu_i] = w
    #     _options = {
    #         "group_name": "param_loading",
    #         "world_size": 2,
    #         "ranks": [0, 1],
    #         "backend": "nccl"
    #     }
    #     # Put A and B in a collective group
    #     collective.create_collective_group(workers, **_options)

    #     self.param_loaders = workers




    # def load_layer_weights(self, layer_i:int):
    #     # let A to send a message to B; a send/recv has to be specified once at each worker
    #     res_ref = [self.param_loaders[0].load_weights_from_gpu.remote(layer_i, 0), self.param_loaders[1].store_weights_to_gpu.remote(src_rank=1)]
    #     return res_ref





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

        # <jingzhi> For profiling
        torch.cuda.synchronize()
        time1 = time.perf_counter()

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)

        # <jingzhi> For profiling
        torch.cuda.synchronize()
        time2 = time.perf_counter()
        # in distributed inference, only the first worker will print information
        # if torch.cuda.current_device() == 0:
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"iter latency: {time2-time1}s")

        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens



    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        if os.environ['USE_VLLM'] == 'True':
            self.load_weights_ori(model_name_or_path, cache_dir, load_format, revision)
        else:
            self.load_weights_ours(model_name_or_path, cache_dir, load_format, revision)




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
            # print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

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
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                # <jingzhi> For Profiling
                # print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)









    def load_weights_ours(self,
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
            # print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

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
                
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) < self.model.weight_cache_block_num):
                # currently only allow one layer on CPU
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) != ((self.model.layer_num + 1)//2)):
                # determine the condition automatically based on pipeline_interval
                if ('model.layers.' not in name) or \
                    (self.get_layer_id_from_name(name)==0) or \
                    ((self.get_layer_id_from_name(name)%self.model.pipeline_inteval)!=0):
                    # if True:
                    # print('loading-----')
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)

                    # print(f"layerBylayer_param_tensor of {name}: {param.data}")
                    # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                break
            else:

                # <jingzhi> For Profiling
                # print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) < self.model.weight_cache_block_num):
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) != ((self.model.layer_num + 1)//2)):
                if ('model.layers.' not in name) or \
                    (self.get_layer_id_from_name(name)==0) or \
                    ((self.get_layer_id_from_name(name)%self.model.pipeline_inteval)!=0):
                    # if True:
                    # print('loading-----')
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

                    # print(f"layerBylayer_param_tensor of {name}: {param.data}")
                    # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

        print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")









    # <jingzhi> For offloading weights layer by layer
    # Set up: (1) weight_num_per_layer, (2) weight_range_in_blk, (3) weight cache blocks
    # (4) making the param tensors pointing to the new GPU physical addresses

    # NOTE: we cannot assume the parameter tensors for the same layer will appear together;
    #       we cannot assume the parameter tensors for different layers appear in the same order

    # NOTE: we need to ensure each parameter tensor is aligned in the our weigh cache, to avoid computation efficiency degradation.

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

                # to avoid OOM
                # param.data = param.data.cpu()

                # need load weight so that we can get packed weight_cache_cpu
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                # print(f"ori_param_tensor of {name}: {param.data}")

                # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

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

                # to avoid OOM
                # param.data = param.data.cpu()

                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

                # print(f"ori_param_tensor of {name}: {param.data}")

                # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

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

        alignment_unit = 128 // self.model.layer_params[0][0].element_size()

        for param in self.model.layer_params[0]:
            # for these information we only need to consider layer 0, as other layers are the same as it
            # we also avoid repeat param as some weights may be packed into one param
            weight_range_in_blk.append((weight_num_per_layer, param.nelement(), param.shape))
            # weight_num_per_layer += param.nelement()

            # NOTE that we need to make the parameter tensor aligned
            print(f"tensor size: {param.nelement()} -> {(((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)} alignment_unit: {alignment_unit}")
            weight_num_per_layer = weight_num_per_layer + (((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)

            assert (params_dtype == None) or (params_dtype == param.dtype)
            params_dtype = param.dtype


        print(f"visited_param_names[30]: {visited_param_names[30]}")
        print(f"weight_range_in_blk: {weight_range_in_blk}")

        # weight_num_per_layer should be the multiple of 1, 2, 3, i.e., multiple of 6
        # NOTE that weight_num_per_layer should also be multiple of alignment_unit
        need_be_multiple_of = len(self.model.cache_device_ids) * alignment_unit # 3 is from the number of cache gpus
        print(f"weight_num_per_layer: {weight_num_per_layer} -> multiple of {need_be_multiple_of} ==> {((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of}")

        # to_pad_each_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of - weight_num_per_layer
        weight_num_per_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of

        print(f"weight_num_per_layer: {weight_num_per_layer}")
        print(f"weight_cache_cpu size: {weight_num_per_layer*self.model.layer_num}")

        # obtain weights_cache_cpu and release the param tensors on GPU
        # make sure weight_cache_cpu is a pinned memory
        weight_cache_cpu = torch.empty(weight_num_per_layer*self.model.layer_num, dtype=self.model.layer_params[0][0].data.cpu().dtype, pin_memory=True)
        print(f"weight_cache_cpu size: {weight_cache_cpu.size()}")
        

        # torch.cat(\
        #     [param.data.cpu().view(-1) for params in self.model.layer_params for param in params] \
        #     + [torch.tensor(range(to_pad_each_layer*self.model.layer_num), dtype=weight_cache_cpu.dtype)], \
        #     out=weight_cache_cpu)
        # 
        # As we make each param tensor mem aligned, they may be empty places between tensors.
        weight_cache_cpu = weight_cache_cpu.view(self.model.layer_num, -1)
        for layer_i, params in enumerate(self.model.layer_params):
            for param, (offset, size, tensor_shape) in zip(params, weight_range_in_blk):
                print(f"offset: {offset}, size: {size}, tensor_shape: {tensor_shape}, param shape: {param.data.shape}, param size: {param.data.size()}")
                weight_cache_cpu[layer_i][offset:offset+size] = param.data.cpu().view(-1)

        print(f"weight_cache_cpu size: {weight_cache_cpu.size()}")
        weight_cache_cpu = weight_cache_cpu.view(self.model.layer_num, len(self.model.cache_device_ids), -1) #weight_num_per_layer)

        # we need to use gpus as a weight cache
        for layer_weights in weight_cache_cpu:
            self.model.weight_cache_cpu.append(list())
            for part_weights, cache_device_i in zip(layer_weights, self.model.cache_device_ids):
                self.model.weight_cache_cpu[-1].append( part_weights.to(cache_device_i) )

        # self.model.weight_cache_cpu = self.model.weight_cache_cpu.to(1)
        print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")
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


        # prepare buffer_params
        for weight_cache_block_idx in [0, self.model.weight_cache_block_num - 1]:
            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.weight_cache, 
                    0, 
                    weight_cache_block_idx*self.model.weight_num_per_layer + rng_info[0], 
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)



        # self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, self.model.weight_num_per_layer)
        self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, len(self.model.cache_device_ids), -1)

        torch.cuda.empty_cache()


        for cache_device_i in self.model.cache_device_ids:
            # cache_ops.init_P2P_access(cache_device_i, 0, 0)
            # in distributed inference, the current device may not be cuda:0
            cache_ops.init_P2P_access(cache_device_i, torch.cuda.current_device(), torch.cuda.current_device()) 

        print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")


    def get_layer_id_from_name(self, name:str) -> int:
        # 'model.layers.24.input_layernorm.weight'
        terms = name.split('.')
        return int(terms[2])








# # define worker to transfer data between GPU 
# # try to use Ray first to see performance
# @ray.remote(num_gpus=1)
# class ParamLoader:
#     def __init__(self, weights_gpus):
#         self.gpu_i = ray.get_gpu_ids()
#         self.weights_gpu = weights_gpus[self.gpu_i[0]]


#     # def setup(self, world_size, rank):
#     #     collective.init_collective_group(world_size, rank, "nccl", "param_loading")
#     #     return True

#     def load_weights_from_gpu(self, layer_i, tgt_rank):
#         collective.send(self.weights_gpu2[layer_i], tgt_rank, group_name="param_loading" )

#     def store_weights_to_gpu(self, layer_i, src_rank):
#         collective.recv(self.weights_gpu2[layer_i], tgt_rank, group_name="param_loading" )





