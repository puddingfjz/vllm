"""
This file contains the search method to find the best exec plans
for the given set of models and the given set of requests.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import itertools
import fake_scheduling
from my_per_iter_latency_estimator import CostTable, get_cost_table
import output_length_sampler

from vllm.transformers_utils.config import get_config
from vllm.worker.cache_engine import CacheEngine
from collections import defaultdict

from model_size_database import model_sizes
import time


_ENGINE_ARGS_LIST = dict()
_FAKE_SCHEDULING_RES = dict() # key: (model name, exec_plan, inp_lens, out_lens) value: the output of fake_scheduling
_MAX_SEQ_NUM = 0
_CHECKED_SEQ_NUM = 0
_MODEL_ID = 0

class InferenceArgs:
    """ The args of the inference setting. """
    def __init__(self, 
        scheduler_config,
        cache_config,
    ) -> None:
        self.prompt_limit = min(scheduler_config.max_model_len,
                                scheduler_config.max_num_batched_tokens)
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.max_seq_num = scheduler_config.max_num_seqs
        self.block_size=cache_config.block_size



class MyModelInfor:
    """ My model information class. Contains basic model information. """
    def __init__(
        self,
        model_id: int,
        cost_table: CostTable,
        model_path, # model_info
        outlen_generator,
        sample_config, trust_remote_code, revision,
        data_byte, # the number of bytes to store a value in model weights or intermediate tensors
        inp_lens: List[int], # the request input lengths.
        out_lens: List[int] = list(), # the request output lengths.
        inp_seq_ids: List[int] = list() # the inp seq ids, to support the chain summary case
    ) -> None:
        self.data_byte = data_byte
        self.model_name = None
        self.model_path = model_path
        self.set_model_name_from_path()
        self.trust_remote_code=trust_remote_code
        self.revision=revision
        # self.model_config=None
        self.hf_config=None
        self.layer_num = None
        self.set_hf_config()

        # self.flops_per_token = flops_per_token
        # self.param_byte_per_layer = None
        # self.extra_param_byte = None
        # self.set_param_byte_per_layer()
        # self.set_extra_param_byte()
        # self.left_flops_per_token = flops_per_token
        
        self.sample_config = sample_config

        # only this parameter will be updated during the search
        self.inp_lens = tuple(inp_lens)
        self.out_lens = tuple(outlen_generator(
            self.model_name, inp_lens)) if len(out_lens) == 0 else tuple(out_lens)
        

        print(f"{self.model_name} avg output len: {sum(self.out_lens)/len(self.out_lens)}")
        print(f"{self.model_name} max output len: {max(self.out_lens)}")

        self.remaining_decode_flops = None
        self.set_remaining_decode_flops(cost_table)

        self.model_id = model_id

        self.input_model_ids: List[int] = list()

        # this value is only set for base models at the beginning, and will not change during the search process
        self.inp_base_model_ids: List[int] = list()


        # model-level pipeline
        self.ori_tot_inp_num: int = len(inp_lens) # this value will not change during the search
        # self.inp_seq_ids = np.asarray(range(self.ori_tot_inp_num))
        self.inp_seq_ids = np.asarray(inp_seq_ids) if len(inp_seq_ids)>0 else np.asarray(range(self.ori_tot_inp_num))

        self.ori_tot_remaining_decode_flops = self.remaining_decode_flops

        self.ori_inp_seq_ids = sorted(self.inp_seq_ids)


    # def set_input_model_ids(self, input_model_ids: List[int]):
    #     self.input_model_ids = input_model_ids
        
        
    def get_base_model_ids(self):
        return [self.model_id]

    def get_base_models(self):
        return [self]
    
    # def get_inp_model_ids(self):
    #     return self.input_model_ids

    def not_started(self):
        return self.ori_tot_remaining_decode_flops == self.remaining_decode_flops

    def set_model_name_from_path(self):
        pos = self.model_path.find('/')
        model_name = self.model_path[pos+1:]
        self.model_name = model_name

    def set_hf_config(self):
        key = (self.model_path, self.trust_remote_code, self.revision)
        hf_config = get_config(*key)
        self.hf_config = hf_config
        L: int = hf_config.num_hidden_layers
        self.layer_num = L


    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_name(self):
        return self.model_name

    def update_inp_out_seqlens(
            self, inp_lens: List[int], out_lens: List[int], inp_seq_ids: List[int],
            cost_table: CostTable, 
            remaining_decode_flops = None):

        # we want to make sure there is not seq with 0 outlens stored.
        valid_inds = np.nonzero(out_lens)[0]
        inp_seq_ids = np.asarray(inp_seq_ids)[valid_inds]
        inp_lens = np.asarray(inp_lens)[valid_inds]
        out_lens = np.asarray(out_lens)[valid_inds]
        # -------------------------------------------------

        self.inp_seq_ids = inp_seq_ids
        self.inp_lens = tuple(inp_lens)
        self.out_lens = tuple(out_lens)
        self.set_remaining_decode_flops(cost_table, remaining_decode_flops)

    def get_inp_out_seqlens(self):
        return (self.inp_lens, self.out_lens)
    
    def get_inp_seq_ids(self):
        return self.inp_seq_ids


    def get_remaining_flops(self):
        ''' Return flops in TFLOPs '''
        return self.remaining_decode_flops
    
        return fake_scheduling.comp_flops_from_seqlens(
            self.inp_lens, self.out_lens, only_decode=True, cost_table=cost_table, 
            model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)


    def set_remaining_decode_flops(self, cost_table: CostTable, remaining_decode_flops = None):
        ''' Return flops in TFLOPs '''
        if remaining_decode_flops != None:
            self.remaining_decode_flops = remaining_decode_flops
        else:
            self.remaining_decode_flops = fake_scheduling.comp_flops_from_seqlens(
                self.inp_lens, self.out_lens, only_decode=True, cost_table=cost_table, 
                model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)



    # def get_remaining_decode_flops_after_infer_stage(
    #         self, cost_table: CostTable, inp_lens: List[int], out_lens: List[int]):
    #     '''
    #         Input:
    #             inp_lens, out_lens: the inp and out lens after an infer stage.
    #     '''
    #     return fake_scheduling.comp_flops_from_seqlens(
    #         inp_lens, out_lens, only_decode=True, cost_table=cost_table, 
    #         model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)



    def is_finished(self):
        return len(self.inp_lens) == 0


    def get_state(self):
        '''
            The state of the model, i.e., its inference progress, is determined by its remaining seqlens.
        '''
        # return (self.model_name, self.get_remaining_flops())
        return (self.model_name, self.model_id, self.inp_lens, self.out_lens)


    def __str__(self) -> str:
        return f'{self.model_name}'
















class MyFusedModelInfor(MyModelInfor):
    """ 
        My model information class. Contains basic model information. 
        Contains the informatio of a fused model.
    """
    def __init__(
        self,
        model_list: List[MyModelInfor],
    ) -> None:
        
        global _MODEL_ID

        self.model_list = model_list
        self.model_id = _MODEL_ID
        _MODEL_ID+=1
        
        model_0 = model_list[0]
        self.data_byte = model_0.data_byte
        self.model_name = model_0.model_name
        self.model_path = model_0.model_path

        self.trust_remote_code=model_0.trust_remote_code
        self.revision=model_0.revision
        # self.model_config=None
        self.hf_config=model_0.hf_config
        self.layer_num = model_0.layer_num

        
        self.sample_config = model_0.sample_config

        # only this parameter will be updated during the search
        self.inp_lens = None #[model.inp_lens for model in self.model_list]
        self.out_lens = None #[model.out_lens for model in self.model_list]
        

        # print(f"MyFusedModelInfor: {self.model_name} base_model_ids: {self.get_base_model_ids()} avg output len: {[sum(_)/len(_) for _ in [model.out_lens for model in self.model_list]]}")
        # print(f"MyFusedModelInfor: {self.model_name} base_model_ids: {self.get_base_model_ids()} max output len: {[max(_) for _ in [model.out_lens for model in self.model_list]]}")

        self.remaining_decode_flops = None #[model.remaining_decode_flops for model in self.model_list]

        # self.model_id = model_id

        # this variable will be set when we contruct new model sys by replacing models with fused models
        self.input_model_ids: List[int] = list()
        self.init_inp_model_ids()

        self.inp_base_model_ids = None
        self.init_inp_base_model_ids()

        # model-level pipeline
        self.ori_tot_inp_num: int = [model.ori_tot_inp_num for model in self.model_list] # this value will not change during the search
        # self.inp_seq_ids = np.asarray(range(self.ori_tot_inp_num))
        self.inp_seq_ids = None #[model.inp_seq_ids for model in self.model_list]

        self.ori_tot_remaining_decode_flops = [model.ori_tot_remaining_decode_flops for model in self.model_list]



    def get_base_model_ids(self):
        return [model.model_id for model in self.model_list]

    def get_base_models(self):
        return self.model_list
    
    def init_inp_model_ids(self):
        for model in self.model_list:
            self.input_model_ids = self.input_model_ids + model.input_model_ids
        # we do not include the in-fused-model in_edges here
        self.input_model_ids = sorted(set(self.input_model_ids).difference(self.get_base_model_ids()))


    def init_inp_base_model_ids(self):
        inp_base_model_ids = list()
        for model in self.model_list:
            inp_base_model_ids = inp_base_model_ids + model.inp_base_model_ids
        # we do not include the in-fused-model in_edges here
        self.inp_base_model_ids = sorted(set(inp_base_model_ids).difference(self.get_base_model_ids()))

    
    # def get_inp_model_ids(self):
    #     return np.concatenate([model.get_inp_model_ids() for model in self.model_list])
        
    def not_started(self):
        return self.ori_tot_remaining_decode_flops == [model.remaining_decode_flops for model in self.model_list]





    # def get_hidden_size(self):
    #     return self.hf_config.hidden_size

    # def get_name(self):
    #     return self.model_name


    def update_inp_out_seqlens(
            self, inp_lens_for_models: List[int], out_lens_for_models: List[int], inp_seq_ids_for_models: List[int],
            cost_table: CostTable, 
            remaining_decode_flops_for_models = None):

        if remaining_decode_flops_for_models == None:
            remaining_decode_flops_for_models = [None for _ in range(len(self.model_list))]

        print(f"in update_inp_out_seqlens\n")
        print(self.model_list, inp_lens_for_models, out_lens_for_models, inp_seq_ids_for_models, remaining_decode_flops_for_models)

        for model, inp_lens, out_lens, inp_seq_ids, remaining_decode_flops in zip(\
            self.model_list, inp_lens_for_models, out_lens_for_models, inp_seq_ids_for_models, remaining_decode_flops_for_models):
            model.update_inp_out_seqlens(inp_lens, out_lens, inp_seq_ids, cost_table, remaining_decode_flops=remaining_decode_flops)



    def get_inp_out_seqlens(self):
        return (tuple([model.inp_lens for model in self.model_list]), tuple([model.out_lens for model in self.model_list]))
    
    def get_inp_seq_ids(self):
        return [model.inp_seq_ids for model in self.model_list]


    def get_remaining_flops(self):
        ''' Return flops in TFLOPs '''
        return [model.remaining_decode_flops for model in self.model_list]
    
        return fake_scheduling.comp_flops_from_seqlens(
            self.inp_lens, self.out_lens, only_decode=True, cost_table=cost_table, 
            model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)


    def set_remaining_decode_flops(self, cost_table: CostTable, remaining_decode_flops = None):
        ''' 
            Return flops in TFLOPs.
            We do not need this function in a fused model.
        '''
        assert False



    def is_finished(self):
        return False not in [model.is_finished() for model in self.model_list]


    def get_state(self):
        '''
            The state of the model, i.e., its inference progress, is determined by its remaining seqlens.
        '''
        assert False
        # return (self.model_name, self.get_remaining_flops())
        return (self.model_name, tuple(self.get_base_model_ids()), \
                tuple([model.inp_lens for model in self.model_list]), \
                    tuple([model.out_lens for model in self.model_list]))


    def __str__(self) -> str:
        return f'{self.model_name}'

















class MyExecPlan:
    """ My execution plan definition. """
    def __init__(
        self,
        model: MyModelInfor, 
        num_worker: int, 
        wld_degree: int, 
        cache_gpu_num: int, 
        mem_per_comp_gpu: float, # gpu utilization ratio
        dp_size: int, # data parallelism degree: num of dp workers
        param_byte_per_comp_gpu: int, 
        param_byte_per_cache_gpu: int,
        gpu_cache_byte_per_block: int,
        infer_args: InferenceArgs,
        tot_gpu_mem_byte: int,
    ) -> None:
        self.model: MyModelInfor = model
        self.num_worker = num_worker
        self.wld_degree = wld_degree
        self.cache_gpu_num = cache_gpu_num
        self.mem_per_comp_gpu = mem_per_comp_gpu
        self.dp_size = dp_size
        self.param_byte_per_comp_gpu = param_byte_per_comp_gpu
        self.param_byte_per_cache_gpu = param_byte_per_cache_gpu
        self.gpu_cache_byte_per_block = gpu_cache_byte_per_block

        # infer_args: the settings of the inference process
        self.infer_args: InferenceArgs = infer_args
        # the total gpu memory available, e.g., 80GB for A100-80GB
        self.tot_gpu_mem_byte: int = tot_gpu_mem_byte
        # basic mem consumption includes the intermediate variables
        self.basic_mem_consumption: int = 0
        self.gpu_cache_block_num = None
        self.set_gpu_cache_block_num()
        

        # store some fake scheduling data
        # this parameters are supposed to be set only once
        # self.cumsum_latencys: List[float] = list()
        # self.cum_rng_nums: List[int] = list()
        # self.rng_starts: List[int] = list()
        # self.rng_ends: List[int] = list()
        # self.is_prefill_steps: List[bool] = list()   

        # self.total_latency: Optional[float] = None

        # self.cache_stop_time_info: Dict[int, Tuple[Tuple[List[int], List[int]], float]] = dict()

        # store some fake scheduling data
        # this parameters are supposed to be set only once
        # store the fake scheduling data for each dp worker
        self.cumsum_latencys_list: List[List[float]] = [list() for _ in range(self.dp_size)]
        self.cum_rng_nums_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.rng_starts_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.rng_ends_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.is_prefill_steps_list: List[List[bool]] = [list() for _ in range(self.dp_size)]

        self.total_latency_list: List[Optional[float]] = [None for _ in range(self.dp_size)]

        self.cache_stop_time_info_list: List[Dict[int, Tuple[Tuple[List[int], List[int]], float]]] \
            = [dict() for _ in range(self.dp_size)]
        
        self.dp_inp_lens_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.dp_out_lens_list: List[List[int]] = [list() for _ in range(self.dp_size)]

        # model-level pipeline 
        # ``finish_times_merged`` stores the finish time of each input in the order of input id (i.e., sequence id)
        # len(self.finish_times_merged) == the model's ori total inp number
        self.finish_times_merged: List[float] = list()
        self.finish_times_list: List[List[float]] = [list() for _ in range(self.dp_size)]
        # # NOTE: ``ref_dp_inp_seq_ids_list`` is the inp seq ids sorted by arrive times in the reused fake scheduling result
        # self.ref_dp_inp_seq_ids_list: List[int] = list() 
        # self.inp_arrive_times does not consider extra_cost
        self.inp_arrive_times: List[Tuple[float, int]] = list() # [(arrive_time, seq_id)]
        self.extra_cost: float = 0.0
        # store the inp seq ids corresponding to dp_inp_lens_list (i.e., sorted by arrive times and seq ids)
        self.dp_inp_seq_ids_list: List[List[int]] = [list() for _ in range(self.dp_size)]


        # valid_throughput and tep_remainin_lens depend on the plan group, instead of individual exec_plans.
        # self.valid_throughput: float = 0
        # self.tmp_remaining_lens
    


    def copy_the_plan(self):
        return MyExecPlan(
            self.model, 
            self.num_worker, 
            self.wld_degree, 
            self.cache_gpu_num, 
            self.mem_per_comp_gpu, # gpu utilization ratio
            self.dp_size, # data parallelism degree: num of dp workers
            self.param_byte_per_comp_gpu, 
            self.param_byte_per_cache_gpu,
            self.gpu_cache_byte_per_block,
            self.infer_args,
            self.tot_gpu_mem_byte,
        )


    def get_base_model_ids(self):
        return self.model.get_base_model_ids()
    
    # def get_exec_plans(self):
    #     return [self]
    
    def get_base_models(self):
        return [self.model]
    
    def models_not_started(self):
        return self.model.not_started()

    def get_dp_inp_lens_list_for_models(self):
        return self.dp_inp_lens_list


    def get_finish_times_merged_not_support_different_ori_seq_ids_in_different_models(self, seq_ids: List[int], model_ind: int):

        """
            NOTE: this version does not support different models have different original seq ids to answer.
        """

        assert model_ind == 0, f"Wrong model ind: {model_ind}"

        # print(f"in get_finish_times_merged: exec_plan: {str(self)}")
        # print(f"self.dp_inp_seq_ids_list: {self.dp_inp_seq_ids_list}")

        if len(self.finish_times_merged) == 0:
            self.finish_times_merged = np.asarray([-1-self.extra_cost]*self.model.ori_tot_inp_num) 
            for dp_inp_seq_ids, finish_times in zip(self.dp_inp_seq_ids_list, self.finish_times_list):
                self.finish_times_merged[dp_inp_seq_ids] = finish_times
        return self.finish_times_merged[seq_ids]
    


    def get_finish_times_merged(self, seq_ids: List[int], model_ind: int):

        assert model_ind == 0, f"Wrong model ind: {model_ind}"

        # print(f"in get_finish_times_merged: exec_plan: {str(self)}")
        # print(f"self.dp_inp_seq_ids_list: {self.dp_inp_seq_ids_list}")

        if len(self.finish_times_merged) == 0:
            self.finish_times_merged = np.asarray([-1-self.extra_cost]*self.model.ori_tot_inp_num) 
            for dp_inp_seq_ids, finish_times in zip(self.dp_inp_seq_ids_list, self.finish_times_list):
                inds = np.searchsorted(self.model.ori_inp_seq_ids, dp_inp_seq_ids)
                # self.finish_times_merged[dp_inp_seq_ids] = finish_times
                self.finish_times_merged[inds] = finish_times


        # print(f"in get_finish_times_merged: self.finish_times_merged: {self.finish_times_merged}")

        # we need to consider the case where the inp seq id is not the output of the model
        # for example: a model after the chain summary, it depends on different model stages in the summary chain
        #           the total inp seq ids will be the full set of inp seq ids, but they are outputs of different model stages

        return get_infor_given_seq_ids(
            values=self.finish_times_merged, 
            seq_ids_we_have=self.model.ori_inp_seq_ids, 
            seq_ids_requested=seq_ids, 
            default_value=-1-self.extra_cost)

        inds = np.searchsorted(self.model.ori_inp_seq_ids, seq_ids)
        # return self.finish_times_merged[seq_ids]
        return self.finish_times_merged[inds]



    def merge_new_inp_out_lens_of_data_parallel_workers(
            self, new_inp_out_lens_list: List[List[List[int]]]
        )->List[List[int]]:
        # print(f"in merge_new_inp_out_lens_of_data_parallel_workers")
        # print(new_inp_out_lens_list)
        # print(self.dp_inp_seq_ids_list)
        inp_lens_list = [dp_inp_out_lens[0] for dp_inp_out_lens in new_inp_out_lens_list]
        out_lens_list = [dp_inp_out_lens[1] for dp_inp_out_lens in new_inp_out_lens_list]
        inp_seq_ids_list = [dp_inp_seq_ids[dp_inp_out_lens[2]] \
                              for dp_inp_seq_ids, dp_inp_out_lens \
                                in zip(self.dp_inp_seq_ids_list, new_inp_out_lens_list)]
        inp_lens = np.concatenate(inp_lens_list)
        out_lens = np.concatenate(out_lens_list)
        inp_seq_ids = np.concatenate(inp_seq_ids_list)
        # order = np.argsort(-inp_lens)
        # sort by inp seqs ids
        order = np.argsort(inp_seq_ids)
        inp_lens = inp_lens[order]
        out_lens = out_lens[order]
        inp_seq_ids = inp_seq_ids[order]
        return [inp_lens, out_lens, inp_seq_ids]




    def set_gpu_cache_block_num(self):
        '''
            gpu cache block num = (available gpu mem - parameter mem) // mem per block
        '''
        self.gpu_cache_block_num = \
            ((self.tot_gpu_mem_byte * self.mem_per_comp_gpu) \
             - self.param_byte_per_comp_gpu - self.basic_mem_consumption) \
                // self.gpu_cache_byte_per_block
        self.gpu_cache_block_num = int(self.gpu_cache_block_num)
        # assert self.gpu_cache_block_num > 0, f"{self.tot_gpu_mem_byte, self.mem_per_comp_gpu, self.param_byte_per_comp_gpu, self.basic_mem_consumption, self.gpu_cache_byte_per_block, self.model.model_name, self.get_key()}"
        # print(f"self.gpu_cache_block_num: {self.gpu_cache_block_num}")


    def set_extra_cost(self, extra_cost: float):
        self.extra_cost = extra_cost

    
    def estimate_exec_time_no_data_parallel(
            self, cost_table: CostTable):
        '''
            Estimate the total inference of this model for the given inp_lens.
        '''
        # 1. first get the input and output lengths (the out lens are already sampled when initialize the model info obj)
        # inp_lens = list(self.model.get_remaining_seqlens())
        # out_lens = output_length_sampler.sample_out_len_for_given_model(
        #     model=self.model.model_name, inp_lens=inp_lens)

        # print(f"in estimate_exec_time: {str(self)}")

        inp_lens, out_lens = self.model.get_inp_out_seqlens()
        

        # print(f"inp lens: {inp_lens}")
        # print(f"out lens: {out_lens}")

        # 2. do fake scheduling
        key = (self.model.model_name, self.get_key(), self.model.get_inp_out_seqlens())
        if key in _FAKE_SCHEDULING_RES:
            (self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends,
                self.is_prefill_steps) = _FAKE_SCHEDULING_RES[key]
            
            self.total_latency = self.cumsum_latencys[-1]

            # # print(f"self.cumsum_latencys: {self.cumsum_latencys}")
            # print(f"from cache, key={key}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")



            return self.total_latency
        else:
            decode_logs, prefill_logs, is_prefill_steps, infer_progress = fake_scheduling.fake_FCFS_schedule(
                inp_lens=list(inp_lens), out_lens=list(out_lens), 
                max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                block_size=self.infer_args.block_size)
        
            # 3. estimate total latency
            tot_latency, prefill_latencys, decode_latencys = \
                fake_scheduling.estimate_prefill_and_decode_cost_from_predicted_logs(
                    prefill_logs=prefill_logs, decode_logs=decode_logs, cost_table=cost_table, 
                    model_name=self.model.model_path, exec_plan=self.get_key(), sample_config=self.model.sample_config, 
                    trust_remote_code=self.model.trust_remote_code, revision=self.model.revision)

            # 4. get cumulative latencys
            self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends = \
                fake_scheduling.get_cumLatency_inferRng_info(
                    decode_latencys, prefill_latencys, 
                    is_prefill_steps, infer_progress)

            # NOTE: since there is a precision issue, we use self.cumsum_latencys[-1] as tot_latency
            tot_latency = self.cumsum_latencys[-1]


            # self.prefill_latencys = prefill_latencys
            # self.decode_latencys = decode_latencys
            self.is_prefill_steps = is_prefill_steps
            # self.infer_progress = infer_progress
            # store the metadata in the global cache
            _FAKE_SCHEDULING_RES[key] = (self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends, 
                                         self.is_prefill_steps)

            # # print(f"self.cumsum_latencys: {self.cumsum_latencys}")
            # print(f"compute from sketch, key={key}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")         


            self.total_latency = tot_latency
            return tot_latency
        


    def _sort_and_partition_data_parallel(
            self, 
            arrive_times: List[float],
            ):
        inp_lens, out_lens = self.model.get_inp_out_seqlens()
        
        # model-level pipeline
        # SORT the input seqs by their arrival times and their seq ids
        inp_seq_ids = self.model.get_inp_seq_ids()
        # consider the extra cost in arrive times
        arrive_times = np.asarray(arrive_times) - self.extra_cost
        to_sort = list(zip(arrive_times, inp_seq_ids))
        # print(f"arrive_times: {arrive_times}")
        # print(f"inp_lens: {inp_lens}")
        # print(f"out_lens: {out_lens}")
        # print(f"inp_seq_ids: {inp_seq_ids}")
        # print(f"to_sort: {to_sort}")
        order = sorted(range(len(to_sort)), key=lambda i: to_sort[i])
        inp_lens = np.asarray(inp_lens)[order]
        out_lens = np.asarray(out_lens)[order]
        inp_seq_ids = np.asarray(inp_seq_ids)[order]
        arrive_times = arrive_times[order]
        self.inp_arrive_times = to_sort

        for dp_id in range(self.dp_size):
            # divide the requests into dp_size groups evenly
            self.dp_inp_lens_list[dp_id] = inp_lens[dp_id::self.dp_size]
            self.dp_out_lens_list[dp_id] = out_lens[dp_id::self.dp_size]
            self.dp_inp_seq_ids_list[dp_id] = inp_seq_ids[dp_id::self.dp_size]

        return inp_lens, out_lens, inp_seq_ids, arrive_times


    def _get_inp_key(
            self,
            arrive_times: List[float],):
        """
            The key contains the inp_lens, out_lens, and the arrive_times.
        """ 
        def _to_tuple(vs):
            return tuple([tuple([tuple(j) for j in i]) for i in vs])

        # directly return the model inp/out lens
        inp_lens_list = list()
        out_lens_list = list()
        arrive_times_list = list()
        for dp_id in range(self.dp_size):
            inp_lens_list.append(tuple(self.dp_inp_lens_list[dp_id]))
            out_lens_list.append(tuple(self.dp_out_lens_list[dp_id]))
            arrive_times_list.append(tuple(arrive_times[dp_id::self.dp_size]))
        return ((tuple(inp_lens_list), tuple(out_lens_list)), tuple(arrive_times_list))



    # data parallel + model-level pipeline
    """
        Basic idea: 
            1. when generating outputs, we sort the output of all dp workers by (finish time, seq id)
            2. when querying available inputs, whether to sort all the available inputs is controlled by ``sort_input``
    """
    def estimate_exec_time(
            self, cost_table: CostTable,
            # 
            check_gap: int,
            sort_input: bool,
            arrive_times: List[float],
            # extra_cost: float,
            ):
        '''
            Estimate the total inference of this model for the given inp_lens.
            Input:
                check_gap: query whether there are newly available requests every ``check_gap`` inference steps.
                sort_input: whether to sort the waiting requests when we query available requests.
                arrive_times: the arrive times of all input requests, extra_cost considered.
                extra_cost: the extra time before running the model, e.g., loading the LLM. [stored as self property]
            NOTE:
                1. to support data parallelism + model-level pipeline parallelism, we need limit each dp worker to 
                    query dp_id-th available request, i.e., we split arrive_times like we do to inp_lens. 
            NOTE: 
                2. the output total exec time considers the extra_cost (e.g., loading the LLM)
                3. the arrive times are in the order of mode.inp_seq_ids. SO we need to SORT them!!!
        '''
        # 1. first get the input and output lengths (the out lens are already sampled when initialize the model info obj)
        # inp_lens = list(self.model.get_remaining_seqlens())
        # out_lens = output_length_sampler.sample_out_len_for_given_model(
        #     model=self.model.model_name, inp_lens=inp_lens)

        # print(f"in estimate_exec_time: {str(self)}")

        # inp_lens, out_lens = self.model.get_inp_out_seqlens()
        
        # # model-level pipeline
        # # SORT the input seqs by their arrival times and their seq ids
        # inp_seq_ids = self.model.get_inp_seq_ids()
        # # consider the extra cost in arrive times
        # arrive_times = np.asarray(arrive_times) - self.extra_cost
        # to_sort = list(zip(arrive_times, inp_seq_ids))
        # order = sorted(range(len(to_sort)), key=lambda i: to_sort[i])
        # inp_lens = np.asarray(inp_lens)[order]
        # out_lens = np.asarray(out_lens)[order]
        # inp_seq_ids = np.asarray(inp_seq_ids)[order]
        # arrive_times = arrive_times[order]
        # self.inp_arrive_times = to_sort

        # for dp_id in range(self.dp_size):
        #     # divide the requests into dp_size groups evenly
        #     self.dp_inp_lens_list[dp_id] = inp_lens[dp_id::self.dp_size]
        #     self.dp_out_lens_list[dp_id] = out_lens[dp_id::self.dp_size]
        #     self.dp_inp_seq_ids_list[dp_id] = inp_seq_ids[dp_id::self.dp_size]
        

        inp_lens, out_lens, inp_seq_ids, arrive_times = self._sort_and_partition_data_parallel(arrive_times)

        # print(f"inp lens: {inp_lens}")
        # print(f"out lens: {out_lens}")

        # 2. do fake scheduling
        # support model-level pipeline: we add arrive_times to the key
        # key = (self.model.model_name, self.get_key(), self.model.get_inp_out_seqlens(), tuple(arrive_times))
        # NOTE: input ``arrive_times`` may not be sorted, so we sort it
        # key = (self.model.model_name, self.get_key(), (tuple(inp_lens), tuple(out_lens)), tuple(arrive_times))

        # NOTE: 因为可能每个dp worker的inference进度不一样，所以当前每个dp worker剩下的request并不是均衡的，这个就会和我们想要的fake scheduling相矛盾，所以这个地方要把
        # 每个dp worker具体的input 和output length信息也存下来。才能做到正确的reuse scheduling results。
        key = (self.model.model_name, self.get_key(), *self._get_inp_key(arrive_times))

        # print(f"key to check in _FAKE_SCHEDULING_RES: {key}")
        # print(f"_FAKE_SCHEDULING_RES keys: {_FAKE_SCHEDULING_RES.keys()}")
        # for k in _FAKE_SCHEDULING_RES.keys():
        #     print(k)

        if key in _FAKE_SCHEDULING_RES:

            print(f"REUSE EXISTING FAKE SCHEDULING RESULTS!\n")

            (self.cumsum_latencys_list, self.cum_rng_nums_list, self.rng_starts_list, self.rng_ends_list,
                self.is_prefill_steps_list, self.finish_times_list) = _FAKE_SCHEDULING_RES[key]
            
            self.total_latency_list = [cumsum_latencys[-1] if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]


            print(f"total_latency_list: {self.total_latency_list}, self.extra_cost: {self.extra_cost}")

            # # print(f"self.cumsum_latencys: {self.cumsum_latencys}")
            # print(f"from cache, key={key}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")

            # # we need to adjust self.finish_times_merged 
            # # because self.ref_dp_inp_seq_ids_list may not be equal to self.dp_inp_seq_ids_list
            # finish_times_merged = np.asarray([-1]*self.model.ori_tot_inp_num) 
            # for dp_inp_seq_ids, ref_dp_inp_seq_ids in zip(self.dp_inp_seq_ids_list, self.ref_dp_inp_seq_ids_list):
            #     finish_times_merged[dp_inp_seq_ids] = self.finish_times_merged[ref_dp_inp_seq_ids]
            # self.finish_times_merged = finish_times_merged


            return self.total_latency_list
        else:

            print(f"DO FAKE SCHEDULING SEARCH!\n")

            # TODO: 这个地方的assert可能不对，因为我们可能是把request按照arrive时间排序的。相同arrive时间可以按照长度排序，
            # 但是这个排序很奇怪，因为model-level pipeline的情况，可能不会保证时刻都排好序。先不管这个？不管request的排序了。
            # 在写入output request的时候不排序，但是在读取input request的时候要排序？在读取input request的排序也不方便啊，
            # 要不然就不对来自所依赖的所有dp worker的input request进行统一排序了，感觉OK？理论上不同request的总长度分布是一样的。
            # 但是其实我们现在的写法会对output整体进行排序：应该改成对整体按照arrive-time进行排序，没有按长度再排序的过程了。
            # 要不然就把按长度排序这块去掉吧，感觉OK的。<= 因为感觉来自不同dp worker的output长度分布应该是一样的；这样从这些output
            # 中均匀间隔地获取input，得到的input长度分布也应该是一样的。
            # 
            # assert list(inp_lens) == sorted(inp_lens, reverse=True), f"The input lens of model {self.model.model_id} is not sorted!"

            # # prepare finish_times_merged
            # # finish time = 1: request available before this stage
            # self.finish_times_merged = np.asarray([-1]*self.model.ori_tot_inp_num) 
            # inp_seq_ids = self.model.get_inp_seq_ids()

            # order = np.argsort(arrive_times)
            # inp_lens = inp_lens[order]
            # out_lens = out_lens[order]
            # inp_seq_ids = inp_seq_ids[order]
            # arrive_times = arrive_times[order]

            for dp_id in range(self.dp_size):
                # divide the requests into dp_size groups evenly
                # dp_inp_lens = inp_lens[dp_id::self.dp_size]
                # dp_out_lens = out_lens[dp_id::self.dp_size]
                # self.dp_inp_lens_list[dp_id] = dp_inp_lens
                # self.dp_out_lens_list[dp_id] = dp_out_lens
                dp_inp_lens = self.dp_inp_lens_list[dp_id]
                dp_out_lens = self.dp_out_lens_list[dp_id]            
                # dp_seq_ids = inp_seq_ids[dp_id::self.dp_size]
                dp_seq_ids = self.dp_inp_seq_ids_list[dp_id]
                dp_arrive_times = arrive_times[dp_id::self.dp_size]

                # decode_logs, prefill_logs, is_prefill_steps, infer_progress = fake_scheduling.fake_FCFS_schedule(
                #     inp_lens=list(dp_inp_lens), out_lens=list(dp_out_lens), 
                #     max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                #     max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                #     block_size=self.infer_args.block_size)
            
                # # 3. estimate total latency
                # tot_latency, prefill_latencys, decode_latencys = \
                #     fake_scheduling.estimate_prefill_and_decode_cost_from_predicted_logs(
                #         prefill_logs=prefill_logs, decode_logs=decode_logs, cost_table=cost_table, 
                #         model_name=self.model.model_path, 
                #         exec_plan=self.get_key_single_dp_worker(), sample_config=self.model.sample_config, 
                #         trust_remote_code=self.model.trust_remote_code, revision=self.model.revision)

                # # 4. get cumulative latencys
                # (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list[dp_id],
                #     self.rng_starts_list[dp_id], self.rng_ends_list[dp_id]) = \
                #         fake_scheduling.get_cumLatency_inferRng_info(
                #             decode_latencys, prefill_latencys, 
                #             is_prefill_steps, infer_progress)


                # support model-level pipeline
                (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list[dp_id], 
                    self.rng_starts_list[dp_id], self.rng_ends_list[dp_id], self.is_prefill_steps_list[dp_id], 
                    finish_times) = \
                        fake_scheduling.fake_FCFS_schedule(
                            inp_lens=list(dp_inp_lens),out_lens=list(dp_out_lens), 
                            arrive_times=dp_arrive_times, check_gap=check_gap,
                            max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                            max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                            block_size=self.infer_args.block_size, 
                            sort_input=sort_input, 
                            cost_estimate_args={
                                "cost_table":cost_table, 
                                "model_name":self.model.model_path, 
                                "exec_plan":self.get_key_single_dp_worker(), 
                                "sample_config":self.model.sample_config, 
                                "trust_remote_code":self.model.trust_remote_code, 
                                "revision":self.model.revision})

                # NOTE: since there is a precision issue, we use self.cumsum_latencys[-1] as tot_latency
                if len(self.cumsum_latencys_list[dp_id]) == 0:
                    self.total_latency_list[dp_id] = 0
                else:
                    self.total_latency_list[dp_id] = self.cumsum_latencys_list[dp_id][-1]


                # # self.prefill_latencys = prefill_latencys
                # # self.decode_latencys = decode_latencys
                # self.is_prefill_steps_list[dp_id] = is_prefill_steps

                # # organize finish_times_merge
                # # NOTE: here we add the extra cost to the finish times --> CHANGE TO NOT ADDing EXTRA COST
                # self.finish_times_merged[dp_seq_ids] = finish_times
                self.finish_times_list[dp_id] = finish_times
            
            # self.infer_progress = infer_progress
            # store the metadata in the global cache
            _FAKE_SCHEDULING_RES[key] = (self.cumsum_latencys_list, self.cum_rng_nums_list, 
                                         self.rng_starts_list, self.rng_ends_list, 
                                         self.is_prefill_steps_list, self.finish_times_list,) 
                                        #  self.dp_inp_seq_ids_list)

            # # print(f"self.cumsum_latencys: {self.cumsum_latencys}")
            # print(f"compute from sketch, key={key}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")         


            return self.total_latency_list


    def get_total_latency_no_data_parallel(self, cost_table: CostTable):
        if self.total_latency == None:
            self.estimate_exec_time_no_data_parallel(cost_table)

        return self.total_latency


    # data parallel + model-level pipeline
    # change name from get_total_latency to get_min_dp_latency --> get_max_dp_latency_considering_plan_group
    def get_max_dp_latency_considering_plan_group(self, cost_table: CostTable,
            check_gap: int, sort_input: bool, arrive_times: List[float]):
        '''
            Input:
                extra_cost: the time to prepare (e.g., load) the model before running.
        '''

        print(f"exec_plan: {str(self)}")
        # print(f"arrive_times: {arrive_times}")

        if self.total_latency_list[0] == None:
            self.estimate_exec_time(cost_table, 
                check_gap=check_gap, sort_input=sort_input, arrive_times=arrive_times)

        print(f"exec plan latency list: {str(self), self.total_latency_list}")
        
        # print(f"finish times list: {self.finish_times_list}")

        return max(self.total_latency_list) + self.extra_cost


    # data parallel
    def get_max_dp_latency(self, cost_table: CostTable, sort_input: bool):
        """
            This function is only used in the baseline where we select the best exec plan for each LLM independently.
            NOTE: 
                1. as we select the best exec plan for each LLM independently, 
                we assume all input requests are available.
                i.e., no model-level pipeline is considered here.
        """
        if self.total_latency_list[0] == None:
            arrive_times = [-1]*len(self.model.get_inp_seq_ids())
            self.estimate_exec_time(cost_table, 
                check_gap=1, sort_input=sort_input, arrive_times=arrive_times)

        return max(self.total_latency_list)


    def update_inp_out_seqlens_and_throughput_after_an_infer_stage_no_data_parallel(
            self, stop_time: float, cost_table: CostTable):
        '''
            1. compute valid throughput.
            2. Update the remaining seqlens after it finishes the current infer stage (until stop_time).
        '''

        # print(f"in update_inp_out_seqlens_and_throughput_after_an_infer_stage: {str(self)}")
        # NOTE: due to the precision issue, sometimes stop_time > self.cumsum_latencys[-1], but it is impossible
        stop_time = min(self.cumsum_latencys[-1], stop_time)

        # 0. check whether we have run this function for the corresponding stop iter.
        stop_iter_i = np.searchsorted(self.cumsum_latencys, stop_time, side='left')
        if stop_iter_i in self.cache_stop_time_info:

            # print(f"stop_iter_i: {stop_iter_i}, stop time info: {self.cache_stop_time_info[stop_iter_i]}")

            return self.cache_stop_time_info[stop_iter_i], stop_iter_i

        actual_stop_time = self.cumsum_latencys[stop_iter_i]

        # print(self.model.model_name, f"stop_iter_i:{stop_iter_i}")
        # print(self.model.model_name, f"rng_starts:{self.rng_starts}")
        # print(self.model.model_name, f"rng_ends:{self.rng_ends}")
        # print(self.model.model_name, f"cum_rng_nums:{self.cum_rng_nums}")


        # 1. compute the seq infer progress after the infer stage
        finished_lens = fake_scheduling.get_info_at_stop_time(
            self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends, 
            stop_time, stop_iter_i)
        # finished_lens, remaining_lens = fake_scheduling.get_info_at_stop_time(
        #     self.decode_latencys, self.prefill_latencys, 
        #     self.is_prefill_steps, self.infer_progress, 
        #     stop_time)


        # print(f"in  update_inp_out_seqlens_and_throughput_after_an_infer_stage----")
        # print(f"stop_iter_i: {stop_iter_i}")
        # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
        # print(f"self.rng_starts: {self.rng_starts.tolist()}")
        # print(f"self.rng_ends: {self.rng_ends.tolist()}")
        # print(self.model.model_name, f"finished_lens:{finished_lens}")


        # 2. compute the valid throughput in the current infer stage
        inp_lens, out_lens = self.model.get_inp_out_seqlens()

        # print(self.model.model_name, f"old inp_lens:{inp_lens}, old remaining_lens:{out_lens}")
        print(self)

        valid_throughput = fake_scheduling.comp_valid_throughput_at_stop_time(
            inp_lens,
            finished_lens, actual_stop_time, cost_table,
            self.model.model_path, self.model.trust_remote_code, self.model.revision)

        # 3. update the remaining_seqlens
        inp_lens = np.asarray(inp_lens) + np.asarray(finished_lens)
        remaining_lens = np.asarray(out_lens) - finished_lens
        valid_indices = (remaining_lens>0)


        # print(self.model.model_name, f"new inp_lens:{inp_lens[valid_indices]}, new remaining_lens:{remaining_lens[valid_indices]}")

        self.cache_stop_time_info[stop_iter_i] = \
            [(tuple(inp_lens[valid_indices]), tuple(remaining_lens[valid_indices])), valid_throughput]



        # print(f"stop_iter_i: {stop_iter_i}, stop time info: {self.cache_stop_time_info[stop_iter_i]}")

        return self.cache_stop_time_info[stop_iter_i], stop_iter_i
        return (tuple(inp_lens[valid_indices]), tuple(remaining_lens[valid_indices])), valid_throughput





    # data parallel
    # return the results for each data parallel worker seperately
    def update_inp_out_seqlens_and_throughput_after_an_infer_stage(
            self, stop_time: float, cost_table: CostTable):
        '''
            1. compute valid throughput.
            2. Update the remaining seqlens after it finishes the current infer stage (until stop_time).
        '''
        new_inp_out_lens_list, valid_throughput_list, stop_iter_i_list = list(), list(), list()
        stage_stop_time = stop_time

        for dp_id in range(self.dp_size):
            cumsum_latencys = self.cumsum_latencys_list[dp_id]
            cache_stop_time_info = self.cache_stop_time_info_list[dp_id]
            cum_rng_nums = self.cum_rng_nums_list[dp_id]
            rng_starts = self.rng_starts_list[dp_id]
            rng_ends = self.rng_ends_list[dp_id]

            if len(cumsum_latencys) == 0:
                new_inp_out_lens_list.append((tuple(np.asarray([])), tuple(np.asarray([])), np.asarray([], dtype=np.int64)))
                valid_throughput_list.append(0)
                stop_iter_i_list.append(0)
                continue                

            # print(f"in update_inp_out_seqlens_and_throughput_after_an_infer_stage: {str(self)}")
            # NOTE: due to the precision issue, sometimes stop_time > self.cumsum_latencys[-1], but it is impossible
            stop_time = min(cumsum_latencys[-1], stage_stop_time)

            # print(f"in update_inp_out_seqlens_and_throughput_after_an_infer_stage\n")

            # 0. check whether we have run this function for the corresponding stop iter.
            stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')

            # print(f"stop_time: {stop_time}, stop_iter_i: {stop_iter_i}, len(cumsum_latencys): {len(cumsum_latencys)}")
            # print(f"cumsum_latencys: {cumsum_latencys}")
            # print(f"cum_rng_nums: {cum_rng_nums}")
            # print(f"rng_starts: {rng_starts}")
            # print(f"rng_ends: {rng_ends}")

            if stop_iter_i in cache_stop_time_info:

                print(f"reuse stop iter i information\n")

                # print(f"stop_iter_i: {stop_iter_i}, stop time info: {self.cache_stop_time_info[stop_iter_i]}")

                new_inp_out_lens, valid_throughput = cache_stop_time_info[stop_iter_i]
                
                # print(f"stop_iter_i: {stop_iter_i}, len(cumsum_latencys): {len(cumsum_latencys)}")
                # print(f"new_inp_out_lens: {new_inp_out_lens}")

                new_inp_out_lens_list.append(new_inp_out_lens)
                valid_throughput_list.append(valid_throughput)
                stop_iter_i_list.append(stop_iter_i)
                continue

                # return cache_stop_time_info[stop_iter_i], stop_iter_i

            actual_stop_time = cumsum_latencys[stop_iter_i]

            # print(self.model.model_name, f"stop_iter_i:{stop_iter_i}")
            # print(self.model.model_name, f"rng_starts:{self.rng_starts}")
            # print(self.model.model_name, f"rng_ends:{self.rng_ends}")
            # print(self.model.model_name, f"cum_rng_nums:{self.cum_rng_nums}")


            # 1. compute the seq infer progress after the infer stage
            finished_lens = fake_scheduling.get_info_at_stop_time(
                cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, 
                stop_time, stop_iter_i)

            # print(f"finished_lens: {finished_lens}")
            # finished_lens, remaining_lens = fake_scheduling.get_info_at_stop_time(
            #     self.decode_latencys, self.prefill_latencys, 
            #     self.is_prefill_steps, self.infer_progress, 
            #     stop_time)


            # print(f"in  update_inp_out_seqlens_and_throughput_after_an_infer_stage----")
            # print(f"stop_iter_i: {stop_iter_i}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")
            # print(self.model.model_name, f"finished_lens:{finished_lens}")


            # 2. compute the valid throughput in the current infer stage
            # inp_lens, out_lens = self.model.get_inp_out_seqlens()
            dp_inp_lens = self.dp_inp_lens_list[dp_id]
            dp_out_lens = self.dp_out_lens_list[dp_id]


            # print(f"dp_inp_lens: {dp_inp_lens}")

            # print(self.model.model_name, f"old inp_lens:{inp_lens}, old remaining_lens:{out_lens}")
            print(self)

            valid_throughput = fake_scheduling.comp_valid_throughput_at_stop_time(
                dp_inp_lens,
                finished_lens, actual_stop_time, cost_table,
                self.model.model_path, self.model.trust_remote_code, self.model.revision)

            # 3. update the remaining_seqlens
            dp_inp_lens = np.asarray(dp_inp_lens) + np.asarray(finished_lens)
            remaining_lens = np.asarray(dp_out_lens) - finished_lens
            # valid_indices = (remaining_lens>0)
            valid_indices = np.nonzero(remaining_lens>0)[0]

            # print(self.model.model_name, f"new inp_lens:{inp_lens[valid_indices]}, new remaining_lens:{remaining_lens[valid_indices]}")

            # NOTE: we store the valid_indices as well
            cache_stop_time_info[stop_iter_i] = \
                [(tuple(dp_inp_lens[valid_indices]), tuple(remaining_lens[valid_indices]), valid_indices), \
                 valid_throughput]

            new_inp_out_lens_list.append(cache_stop_time_info[stop_iter_i][0])
            valid_throughput_list.append(valid_throughput)
            stop_iter_i_list.append(stop_iter_i)

            # print(f"stop_iter_i: {stop_iter_i}, stop time info: {self.cache_stop_time_info[stop_iter_i]}")
            
            # print(f"stop_iter_i: {stop_iter_i}, len(cumsum_latencys): {len(cumsum_latencys)}")
            # print(f"new_inp_out_lens: {cache_stop_time_info[stop_iter_i][0]}")
            
            # return self.cache_stop_time_info[stop_iter_i], stop_iter_i
            # # return (tuple(inp_lens[valid_indices]), tuple(remaining_lens[valid_indices])), valid_throughput


        # print(f"new_inp_out_lens_list: {new_inp_out_lens_list}")

        return (new_inp_out_lens_list, valid_throughput_list), stop_iter_i_list








    def update_fake_schedule_output_after_an_infer_stage_no_data_parallel(
            self, 
            old_inp_lens: List[int], new_inp_lens: List[int], new_out_lens: List[int], 
            stop_iter_i: int, cost_table: CostTable):
        '''
            This function is called when the exec plan is selected to run for an infer stage.
            Update:
                _FAKE_SCHEDULING_RES[model_name, exec_plan, new_inp_lens, new_out_lens]
        '''
        # 1. update the infer metadata after the infer stage
        cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps = \
            fake_scheduling.update_fake_FCFS_schedule_metadata(
                old_inp_lens, new_inp_lens,
                self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends, 
                self.is_prefill_steps,
                self.infer_args.max_num_batched_tokens, stop_iter_i,
                cost_table, 
                model_name=self.model.model_path, 
                exec_plan=self.get_key(), sample_config=self.model.sample_config, 
                trust_remote_code=self.model.trust_remote_code, revision=self.model.revision
                )
        
        new_key = (self.model.model_name, self.get_key(), (tuple(new_inp_lens), tuple(new_out_lens)))
        _FAKE_SCHEDULING_RES[new_key] = cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps






    def _sort_scheduling_results_by_seq_ids(
            self,
            alive_seq_ids,
            cum_rng_nums, rng_starts, rng_ends, finish_times_of_alive_seqs):
        order = np.argsort(alive_seq_ids)
        new_rng_starts = np.empty_like(rng_starts)
        new_rng_ends = np.empty_like(rng_ends)
        rng_nums = np.diff(cum_rng_nums)[order]
        new_cum_rng_nums = np.cumsum(np.concatenate(([0], rng_nums)))
        for i, ori_ind in enumerate(order):
            new_rng_starts[new_cum_rng_nums[i]:new_cum_rng_nums[i+1]] = \
                rng_starts[cum_rng_nums[ori_ind]:cum_rng_nums[ori_ind+1]]
            new_rng_ends[new_cum_rng_nums[i]:new_cum_rng_nums[i+1]] = \
                rng_ends[cum_rng_nums[ori_ind]:cum_rng_nums[ori_ind+1]]
        new_finish_times_of_alive_seqs = finish_times_of_alive_seqs[order]
        return new_cum_rng_nums, new_rng_starts, new_rng_ends, new_finish_times_of_alive_seqs



    # data parallel + model-level pipeline
    def update_fake_schedule_output_after_an_infer_stage(
            self, 
            old_inp_lens_list: List[List[int]], 
            new_inp_out_lens_list: List[List[List[int]]], 
            new_inp_lens_merged: List[int],
            new_out_lens_merged: List[int],
            new_inp_seq_ids_merged: List[int],
            stop_iter_i_list: List[int], 
            cost_table: CostTable, 
            ):
        '''
            This function is called when the exec plan is selected to run for an infer stage.
            Update:
                _FAKE_SCHEDULING_RES[model_name, exec_plan, new_inp_lens, new_out_lens]
            NOTE:
                we only call this function when after an infer stage, all the requests are available.
            NOTE: 
                1. ``new_inp_lens_merged`` is already sorted by seq ids.
                2. we only do update when there are unfinished requests.
        '''
        # print(f"in update_fake_schedule_output_after_an_infer_stage")
        # print(f"in / out seqlens: {self.model.get_inp_out_seqlens()}")

        # check whether all the input requests are available after this stage is finished
        if (max([i[0] for i in self.inp_arrive_times]) > \
            max([cumsum_latencys[stop_iter_i] if len(cumsum_latencys) > 0 else 0 \
                 for cumsum_latencys, stop_iter_i in \
                 zip(self.cumsum_latencys_list, stop_iter_i_list)])):
            # there are input not available after this stage finishes
            return


        # NOTE: 因为可能每个dp worker的inference进度不一样，所以当前每个dp worker剩下的request并不是均衡的，这个就会和我们想要的fake scheduling相矛盾，
        # 所以对于每个dp worker剩下来的 request 和正常均分workload结果矛盾的情况，我们没有比较进行下面的计算，因为这样的scheduling结果不会被reuse到。
        # 但是这个好麻烦啊，暂时先不管。



        cumsum_latencys_list = [list() for _ in range(self.dp_size)]
        cum_rng_nums_list = [list() for _ in range(self.dp_size)]
        rng_starts_list = [list() for _ in range(self.dp_size)]
        rng_ends_list = [list() for _ in range(self.dp_size)]
        is_prefill_steps_list = [list() for _ in range(self.dp_size)]
        # finish_times_merged = np.asarray([-1]*self.model.ori_tot_inp_num)
        finish_times_list = [list() for _ in range(self.dp_size)]

        for dp_id in range(self.dp_size):
            
            old_inp_lens = old_inp_lens_list[dp_id]
            # new_inp_lens, new_out_lens, new_inp_seq_ids = new_inp_out_lens_list[dp_id]
            stop_iter_i = stop_iter_i_list[dp_id]

            # 1. update the infer metadata after the infer stage
            (cumsum_latencys_list[dp_id], cum_rng_nums_list[dp_id], 
                rng_starts_list[dp_id], rng_ends_list[dp_id], is_prefill_steps_list[dp_id], 
                finish_times_of_alive_seqs, alive_old_indices) = \
                    fake_scheduling.update_fake_FCFS_schedule_metadata(
                        old_inp_lens, # new_inp_lens,
                        self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list[dp_id], 
                        self.rng_starts_list[dp_id], self.rng_ends_list[dp_id], 
                        self.is_prefill_steps_list[dp_id],
                        self.infer_args.max_num_batched_tokens, stop_iter_i,
                        cost_table, 
                        model_name=self.model.model_path, 
                        exec_plan=self.get_key_single_dp_worker(), sample_config=self.model.sample_config, 
                        trust_remote_code=self.model.trust_remote_code, revision=self.model.revision
                        )
            

            # we may need to reorder the scheduling results if the remaining seqs are not in the order of their seq ids
            # 为什么这个地方要sort，fused exec plan的地方不用sort？应该都要sort吧？应该都要sort的，因为我们在准备workload的时候是sort了的。
            # 暂时先把sort全都去掉了，因为我们现在的key并没有进行sort，所以去掉sort后能保证fake scheduling结果和inp reqs的对应。
            # 如果要扩大reuse的范围,应该要把剩余的inp req 的分配情况也放到被reuse的内容里面.
            # cum_rng_nums_list[dp_id], rng_starts_list[dp_id], rng_ends_list[dp_id], finish_times_of_alive_seqs = \
            #     self._sort_scheduling_results_by_seq_ids(
            #         self.dp_inp_seq_ids_list[dp_id][alive_old_indices],
            #         cum_rng_nums_list[dp_id], rng_starts_list[dp_id], rng_ends_list[dp_id], finish_times_of_alive_seqs)


            # dp_seq_ids = self.dp_seq_ids_list[dp_id]
            # finish_times_merged[dp_seq_ids[alive_old_indices]] = finish_times_of_alive_seqs
            finish_times_list[dp_id] = finish_times_of_alive_seqs
            assert (new_inp_out_lens_list[dp_id][2] == alive_old_indices).all(), print(new_inp_out_lens_list[dp_id][2], alive_old_indices)


        # new_key = (self.model.model_name, self.get_key(), (tuple(new_inp_lens_merged), tuple(new_out_lens_merged)), \
        #            tuple([-1 - self.extra_cost]*len(new_inp_lens_merged)))
        # NOTE: 因为可能每个dp worker的inference进度不一样，所以当前每个dp worker剩下的request并不是均衡的，这个就会和我们想要的fake scheduling相矛盾，所以这个地方要把
        # 每个dp worker具体的input 和output length信息也存下来。
        new_key = (self.model.model_name, self.get_key(), 
                   (tuple([tuple(dp_data[0]) for dp_data in new_inp_out_lens_list]), tuple([tuple(dp_data[1]) for dp_data in new_inp_out_lens_list])), \
                   tuple([tuple([-1 - self.extra_cost]*len(dp_data[0])) for dp_data in new_inp_out_lens_list]))
        # print(f"update fake scheduling: new_key: {new_key}")
        # print(f"")
        _FAKE_SCHEDULING_RES[new_key] = \
            cumsum_latencys_list, cum_rng_nums_list, rng_starts_list, rng_ends_list, is_prefill_steps_list, \
            finish_times_list
            # finish_times_merged, new_inp_seq_ids_merged





    def get_key(self):
        # the key of a exec plan is (tp, gpu_ratio, wldeg, cache_gpu_num)
        # data parallel
        # ==> (tp, gpu_ratio, wldeg, cache_gpu_num, dp_size)
        return (self.num_worker, self.mem_per_comp_gpu, self.wld_degree, self.cache_gpu_num, self.dp_size)
    
    def get_key_single_dp_worker(self):
        # the key of the exec plan for a data parallel worker is (tp, gpu_ratio, wldeg, cache_gpu_num)
        return (self.num_worker, self.mem_per_comp_gpu, self.wld_degree, self.cache_gpu_num)


    def __str__(self) -> str:
        return f"{str(self.model)}, "\
            f"{self.get_key()}"
            # f"tp:{self.num_worker}, wld:{self.wld_degree}, "\
            # f"cache_gpu:{self.cache_gpu_num}, mem_r:{self.mem_per_comp_gpu}, "\
            # f"param_byte_per_comp_gpu:{self.param_byte_per_comp_gpu}, param_byte_per_cache_gpu:{self.param_byte_per_cache_gpu}"







class MyVerticalFusedExecPlan(MyExecPlan):
    """ 
        My execution plan definition. There may be multiple models fused vertically in this plan. 
        NOTE: we assume the original complete inp seq ids for each model is range(ori_tot_req_num)!
        有需要的时候再改吧。
    """
    def __init__(
        self,
        # model_list: List[MyModelInfor], 
        fused_model: MyFusedModelInfor,
        # fused_exec_plans: List[MyExecPlan],
        shared_exec_plan: MyExecPlan,
        # 
    ) -> None:
        self.model:MyFusedModelInfor = fused_model
        self.model_list: List[MyModelInfor] = fused_model.model_list
        # self.fused_exec_plans: List[MyExecPlan] = fused_exec_plans      

        exec_plan_0 = shared_exec_plan
        self.num_worker = exec_plan_0.num_worker
        self.wld_degree = exec_plan_0.wld_degree
        self.cache_gpu_num = exec_plan_0.cache_gpu_num
        self.mem_per_comp_gpu = exec_plan_0.mem_per_comp_gpu
        self.dp_size = exec_plan_0.dp_size
        self.param_byte_per_comp_gpu = exec_plan_0.param_byte_per_comp_gpu
        self.param_byte_per_cache_gpu = exec_plan_0.param_byte_per_cache_gpu
        self.gpu_cache_byte_per_block = exec_plan_0.gpu_cache_byte_per_block

        # infer_args: the settings of the inference process
        self.infer_args: InferenceArgs = exec_plan_0.infer_args
        # the total gpu memory available, e.g., 80GB for A100-80GB
        self.tot_gpu_mem_byte: int = exec_plan_0.tot_gpu_mem_byte
        # basic mem consumption includes the intermediate variables
        self.basic_mem_consumption: int = exec_plan_0.basic_mem_consumption
        self.gpu_cache_block_num = exec_plan_0.gpu_cache_block_num
        # self.set_gpu_cache_block_num()


        # store some fake scheduling data
        # this parameters are supposed to be set only once
        # store the fake scheduling data for each dp worker
        self.cumsum_latencys_list: List[List[float]] = [list() for _ in range(self.dp_size)]
        self.cum_rng_nums_list_for_models: List[List[int]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.rng_starts_list_for_models: List[List[int]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.rng_ends_list_for_models: List[List[int]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.is_prefill_steps_list: List[List[bool]] = [list() for _ in range(self.dp_size)]

        self.total_latency_list: List[Optional[float]] = [None for _ in range(self.dp_size)]

        self.cache_stop_time_info_list: List[Dict[int, Tuple[Tuple[List[int], List[int]], float]]] \
            = [dict() for _ in range(self.dp_size)]
        
        # NOTE: we have one dp_inp_lens_list for each model in the mode_list
        self.dp_inp_lens_list_for_models: List[List[List[int]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.dp_out_lens_list_for_models: List[List[List[int]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        # self.dp_inp_lens_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        # self.dp_out_lens_list: List[List[int]] = [list() for _ in range(self.dp_size)]

        # model-level pipeline 
        # ``finish_times_merged`` stores the finish time of each input in the order of input id (i.e., sequence id)
        # len(self.finish_times_merged) == the model's ori total inp number
        self.finish_times_merged_for_models: List[List[float]] = [list() for _ in self.model_list]
        self.finish_times_list_for_models: List[List[List[float]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        # # NOTE: ``ref_dp_inp_seq_ids_list`` is the inp seq ids sorted by arrive times in the reused fake scheduling result
        # self.ref_dp_inp_seq_ids_list: List[int] = list() 
        # self.inp_arrive_times does not consider extra_cost
        self.inp_arrive_times_for_models: List[List[Tuple[float, int]]] = [list() for _ in self.model_list] # [(arrive_time, seq_id)]
        self.extra_cost: float = 0.0
        # store the inp seq ids corresponding to dp_inp_lens_list (i.e., sorted by arrive times and seq ids)
        self.dp_inp_seq_ids_list_for_models: List[List[List[int]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.dp_arrive_times_list_for_models: List[List[List[float]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]


        # valid_throughput and tep_remainin_lens depend on the plan group, instead of individual exec_plans.
        # self.valid_throughput: float = 0
        # self.tmp_remaining_lens


    def copy_the_plan(self):
        shared_exec_plan = MyExecPlan(
            self.model, 
            self.num_worker, 
            self.wld_degree, 
            self.cache_gpu_num, 
            self.mem_per_comp_gpu, # gpu utilization ratio
            self.dp_size, # data parallelism degree: num of dp workers
            self.param_byte_per_comp_gpu, 
            self.param_byte_per_cache_gpu,
            self.gpu_cache_byte_per_block,
            self.infer_args,
            self.tot_gpu_mem_byte,
        )
        return MyVerticalFusedExecPlan(self.model, shared_exec_plan)




    def get_base_model_ids(self):
        return self.model.get_base_model_ids()

    # def get_exec_plans(self):
    #     return self.fused_exec_plans
    
    def get_base_models(self):
        return self.model_list
    
    def models_not_started(self):
        # return False not in [model.not_started() for model in self.model_list]
        return self.model.not_started()
        
    def get_dp_inp_lens_list_for_models(self):
        return self.dp_inp_lens_list_for_models

    def get_finish_times_merged_limited_version(self, seq_ids: List[int]):
        """
            NOTE: this version only support fused model 是严格线性依赖关系，在fused model之外的model只会依赖于fused model的最后一个base model
            的output。
        """

        if len(self.finish_times_merged_for_models[-1]) == 0:
            self.finish_times_merged_for_models[-1] = np.asarray([-1-self.extra_cost]*self.model_list[-1].ori_tot_inp_num) 
            
            for dp_inp_seq_ids_for_models, finish_times_for_models in \
                zip(self.dp_inp_seq_ids_list_for_models, self.finish_times_list_for_models):
                inds = np.searchsorted(self.model_list[-1].ori_inp_seq_ids, dp_inp_seq_ids_for_models[-1])
                self.finish_times_merged_for_models[-1][inds] = finish_times_for_models[-1]
        
        inds = np.searchsorted(self.model_list[-1].ori_inp_seq_ids, seq_ids)
        return self.finish_times_merged_for_models[-1][inds]




    def get_finish_times_merged(self, seq_ids: List[int], model_ind: int):
        """
            Input: 
                model_ind: we want to get the seq finish times of the ``model_ind``-th base model of the fused model 
        """

        if len(self.finish_times_merged_for_models[model_ind]) == 0:
            self.finish_times_merged_for_models[model_ind] = np.asarray([-1-self.extra_cost]*self.model_list[model_ind].ori_tot_inp_num) 
            
            for dp_inp_seq_ids_for_models, finish_times_for_models in \
                zip(self.dp_inp_seq_ids_list_for_models, self.finish_times_list_for_models):
                inds = np.searchsorted(self.model_list[model_ind].ori_inp_seq_ids, dp_inp_seq_ids_for_models[model_ind])
                self.finish_times_merged_for_models[model_ind][inds] = finish_times_for_models[model_ind]

        # we need to consider the case where the inp seq id is not the output of the model
        # for example: a model after the chain summary, it depends on different model stages in the summary chain
        #           the total inp seq ids will be the full set of inp seq ids, but they are outputs of different model stages

        # print(f"in get_finish_times_merged: self.finish_times_merged_for_models: {self.finish_times_merged_for_models}")


        return get_infor_given_seq_ids(
            values=self.finish_times_merged_for_models[model_ind], 
            seq_ids_we_have=self.model_list[model_ind].ori_inp_seq_ids, 
            seq_ids_requested=seq_ids, 
            default_value=-1-self.extra_cost)


        inds = np.searchsorted(self.model_list[model_ind].ori_inp_seq_ids, seq_ids)
        return self.finish_times_merged_for_models[model_ind][inds]




    def merge_new_inp_out_lens_of_data_parallel_workers(
            self, new_inp_out_lens_list_for_models: List[List[List[List[int]]]]
        )->List[List[int]]:
        """
            Support vertical fusion of models.
        """
        # print(f"in merge_new_inp_out_lens_of_data_parallel_workers")
        # print(new_inp_out_lens_list)
        # print(self.dp_inp_seq_ids_list)

        print(f"{str(self)}")
        
        # data_num = 3: there are 3 kinds of data: inp_lens, out_lens, valid_indices
        data_num = len(new_inp_out_lens_list_for_models[0])
        new_inp_lens_for_models = list()
        new_out_lens_for_models = list()
        new_inp_seq_ids_for_models = list()
        for i in range(len(self.model_list)):
            new_inp_out_lens_list = [[dp_data[data_i][i] for data_i in range(data_num)] for dp_data in new_inp_out_lens_list_for_models]
            dp_inp_seq_ids_list = [dp_data[i] for dp_data in self.dp_inp_seq_ids_list_for_models]

            # print(f"new_inp_out_lens_list:{new_inp_out_lens_list}")
            # print(f"dp_inp_seq_ids_list:{dp_inp_seq_ids_list}")

            inp_lens_list = [dp_inp_out_lens[0] for dp_inp_out_lens in new_inp_out_lens_list]
            out_lens_list = [dp_inp_out_lens[1] for dp_inp_out_lens in new_inp_out_lens_list]
            inp_seq_ids_list = [dp_inp_seq_ids[dp_inp_out_lens[2]] \
                                for dp_inp_seq_ids, dp_inp_out_lens \
                                    in zip(dp_inp_seq_ids_list, new_inp_out_lens_list)]
            inp_lens = np.concatenate(inp_lens_list)
            out_lens = np.concatenate(out_lens_list)
            inp_seq_ids = np.concatenate(inp_seq_ids_list)
            # order = np.argsort(-inp_lens)
            # sort by inp seqs ids
            # print(f"inp_lens: {inp_lens}")
            # print(f"out_lens: {out_lens}")
            # print(f"inp_seq_ids: {inp_seq_ids}")

            order = np.argsort(inp_seq_ids)

            # print(f"order: {order}")

            inp_lens = inp_lens[order]
            out_lens = out_lens[order]
            inp_seq_ids = inp_seq_ids[order]
            new_inp_lens_for_models.append(inp_lens)
            new_out_lens_for_models.append(out_lens)
            new_inp_seq_ids_for_models.append(inp_seq_ids)            
        return [new_inp_lens_for_models, new_out_lens_for_models, new_inp_seq_ids_for_models]



    def set_gpu_cache_block_num(self):
        '''
            gpu cache block num = (available gpu mem - parameter mem) // mem per block
        '''

        """
            We will not call this method directly on a fused exec plan object.
        """
        assert False




    # def set_extra_cost(self, extra_cost: float):
    #     """
    #         We will not call this method directly on a fused exec plan object.
    #         We use the same method as in MyExecPlan.
    #     """
    #     assert False

    

    def _sort_and_partition_data_parallel(
            self, 
            arrive_times_list: List[List[float]],
            ):
        """
            Update:
                self.dp_inp_seq_ids_list_for_models, 
                self.dp_inp_lens_list_for_models,
                self.dp_out_lens_list_for_models,
                self.dp_arrive_times_list_for_models
        """

        def _sort_and_assign(seq_infos, model_i: int, dp_size: int):
            """
                sort the seqs and assign them to the dp workers.
            """
            inp_lens = [i[0] for i in seq_infos]
            out_lens = [i[1] for i in seq_infos]
            inp_seq_ids = [i[2] for i in seq_infos]
            arrive_times = [i[3] for i in seq_infos]

            to_sort = list(zip(arrive_times, inp_seq_ids))
            order = sorted(range(len(to_sort)), key=lambda i: to_sort[i])
            inp_lens = np.asarray(inp_lens)[order]
            out_lens = np.asarray(out_lens)[order]
            inp_seq_ids = np.asarray(inp_seq_ids)[order]
            arrive_times = np.asarray(arrive_times)[order]

            # print(f"inp_lens: {inp_lens}, dp_size: {dp_size}")

            for dp_id in range(dp_size):
                # divide the requests into dp_size groups evenly                
                self.dp_inp_lens_list_for_models[dp_id][model_i].extend(inp_lens[dp_id::dp_size])
                self.dp_out_lens_list_for_models[dp_id][model_i].extend(out_lens[dp_id::dp_size])
                self.dp_inp_seq_ids_list_for_models[dp_id][model_i].extend(inp_seq_ids[dp_id::dp_size])
                self.dp_arrive_times_list_for_models[dp_id][model_i].extend(arrive_times[dp_id::dp_size])
        
        def _assign_in_consistent_with_previous_assignment(seq_infos, model_i: int, dp_size: int):
            """
                assign seqs to the dp worker which generates the their previous versions.
            """
            inp_lens = np.asarray([i[0] for i in seq_infos])
            out_lens = np.asarray([i[1] for i in seq_infos])
            inp_seq_ids = np.asarray([i[2] for i in seq_infos])
            arrive_times = np.asarray([i[3] for i in seq_infos])
            order = np.argsort(inp_seq_ids)
            for dp_id in range(dp_size):
                last_model_seq_ids = self.dp_inp_seq_ids_list_for_models[dp_id][model_i-1]
                seq_ids_to_add = sorted(set(inp_seq_ids).intersection(last_model_seq_ids))
                inds = np.searchsorted(inp_seq_ids[order], seq_ids_to_add)
                self.dp_inp_lens_list_for_models[dp_id][model_i].extend(inp_lens[order][inds])
                self.dp_out_lens_list_for_models[dp_id][model_i].extend(out_lens[order][inds])
                self.dp_inp_seq_ids_list_for_models[dp_id][model_i].extend(inp_seq_ids[order][inds])
                self.dp_arrive_times_list_for_models[dp_id][model_i].extend(arrive_times[order][inds])


        def _sort_by_seq_ids(dp_size: int):
            """
                Sort the seq info lists by seq ids.
            """
            for dp_id in range(dp_size):
                for model_i in range(len(self.model_list)):
                    order = np.argsort(self.dp_inp_seq_ids_list_for_models[dp_id][model_i])
                    self.dp_inp_lens_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_inp_lens_list_for_models[dp_id][model_i])[order]
                    self.dp_out_lens_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_out_lens_list_for_models[dp_id][model_i])[order]
                    self.dp_inp_seq_ids_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_inp_seq_ids_list_for_models[dp_id][model_i])[order]
                    self.dp_arrive_times_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_arrive_times_list_for_models[dp_id][model_i])[order]




        seq_ids_visited = list()
        # dp_size = self.dp_size       

        for i in range(len(self.model_list)):
            # 1. get the seq whose inp arrive times are known
            known = list()
            unknown = list()
            model = self.model_list[i]
            inp_lens, out_lens = model.get_inp_out_seqlens()
            inp_seq_ids = model.get_inp_seq_ids()
            for inp_len, out_len, seq_id, arrive_time in zip(inp_lens, out_lens, inp_seq_ids, arrive_times_list[i]):
                if seq_id not in seq_ids_visited:
                    unknown.append((inp_len, out_len, seq_id, arrive_time - self.extra_cost))
                    seq_ids_visited.append(seq_id)
                else:
                    known.append((inp_len, out_len, seq_id, arrive_time - self.extra_cost))
            
            # print(f"unknown: {unknown}")
            # print(f"known: {known}")

            # 2. partition the known and unknown among the dp workers
            # all the seq ids are sorted
            assert sorted(inp_seq_ids) == list(inp_seq_ids)
            # for unknown seqs, partition them evenly 
            _sort_and_assign(unknown, i, self.dp_size)
            # for known seqs, partition them according to the previous partition solution
            _assign_in_consistent_with_previous_assignment(known, i, self.dp_size)


        # sort the seq infos by seq ids
        _sort_by_seq_ids(self.dp_size)


    def _get_inp_key_merged_version(
            self,
            arrive_times_list: List[List[float]],):
        """
            The key contains the inp_lens, out_lens, and the arrive_times.
        """ 
        def _to_tuple(vs):
            return tuple([tuple([tuple(j) for j in i]) for i in vs])

        # directly return the model inp/out lens
        inp_lens = list()
        out_lens = list()
        arrive_times = list()
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            inps, outs = model.get_inp_out_seqlens()
            inp_lens.append(tuple(inps))
            out_lens.append(tuple(outs))
            arrive_times.append(tuple(np.asarray(arrive_times_list[i]) - self.extra_cost))
        return ((tuple(inp_lens), tuple(out_lens)), tuple(arrive_times))

        tuple_inp_lens = _to_tuple(self.dp_inp_lens_list_for_models)
        tuple_out_lens = _to_tuple(self.dp_out_lens_list_for_models)
        tuple_arrive_times = _to_tuple(self.dp_arrive_times_list_for_models)
        return ((tuple_inp_lens, tuple_out_lens), tuple_arrive_times)




    def _get_inp_key(
            self,
            arrive_times_list: List[List[float]],):
        """
            The key contains the inp_lens, out_lens, and the arrive_times.
        """ 
        def _to_tuple(vs):
            return tuple([tuple([tuple(j) for j in i]) for i in vs])

        # # directly return the model inp/out lens
        # inp_lens = list()
        # out_lens = list()
        # arrive_times = list()
        # for i in range(len(self.model_list)):
        #     model = self.model_list[i]
        #     inps, outs = model.get_inp_out_seqlens()
        #     inp_lens.append(tuple(inps))
        #     out_lens.append(tuple(outs))
        #     arrive_times.append(tuple(np.asarray(arrive_times_list[i]) - self.extra_cost))
        # return ((tuple(inp_lens), tuple(out_lens)), tuple(arrive_times))

        tuple_inp_lens = _to_tuple(self.dp_inp_lens_list_for_models)
        tuple_out_lens = _to_tuple(self.dp_out_lens_list_for_models)
        tuple_arrive_times = _to_tuple(self.dp_arrive_times_list_for_models)
        return ((tuple_inp_lens, tuple_out_lens), tuple_arrive_times)




    # data parallel + model-level pipeline
    """
        Basic idea: 
            1. when generating outputs, we sort the output of all dp workers by (finish time, seq id)
            2. when querying available inputs, whether to sort all the available inputs is controlled by ``sort_input``
    """
    def estimate_exec_time(
            self, cost_table: CostTable,
            # 
            check_gap: int,
            sort_input: bool,
            # arrive_times: List[float],
            arrive_times_list: List[List[float]],
            # extra_cost: float,
            ):
        '''
            Estimate the total inference of this model for the given inp_lens.
            Input:
                check_gap: query whether there are newly available requests every ``check_gap`` inference steps.
                sort_input: whether to sort the waiting requests when we query available requests.
                arrive_times: the arrive times of all input requests, extra_cost considered.
                extra_cost: the extra time before running the model, e.g., loading the LLM. [stored as self property]
            NOTE:
                1. to support data parallelism + model-level pipeline parallelism, we need limit each dp worker to 
                    query dp_id-th available request, i.e., we split arrive_times like we do to inp_lens. 
            NOTE: 
                2. the output total exec time considers the extra_cost (e.g., loading the LLM)
                3. the arrive times are in the order of mode.inp_seq_ids. SO we need to SORT them!!!
        '''
        # 1. first get the input and output lengths (the out lens are already sampled when initialize the model info obj)
        # inp_lens = list(self.model.get_remaining_seqlens())
        # out_lens = output_length_sampler.sample_out_len_for_given_model(
        #     model=self.model.model_name, inp_lens=inp_lens)

        # print(f"in estimate_exec_time: {str(self)}")

        # 1. first get the input and output lengths
        # NOTE: for model 0, we divide inputs evenly; for other models, we assign divide the inputs 
        # according to the dp input assignment for model 0


        time1 = time.perf_counter()

        self._sort_and_partition_data_parallel(arrive_times_list)
       
        time2 = time.perf_counter()
        print(f"TIME--_sort_and_partition_data_parallel: {time2 - time1}")

        # print(f"inp lens: {inp_lens}")
        # print(f"out lens: {out_lens}")

        # 2. do fake scheduling
        # support model-level pipeline: we add arrive_times to the key
        # key = (self.model.model_name, self.get_key(), self.model.get_inp_out_seqlens(), tuple(arrive_times))
        # NOTE: input ``arrive_times`` may not be sorted, so we sort it
        # key = (self.model.model_name, self.get_key(), (tuple(inp_lens), tuple(out_lens)), tuple(arrive_times))
        # key = (self.model.model_name, self.get_key(), *(self._get_inp_key(arrive_times_list)))

        # NOTE: 因为可能每个dp worker的inference进度不一样，所以当前每个dp worker剩下的request并不是均衡的，这个就会和我们想要的fake scheduling相矛盾，所以这个地方要把
        # 每个dp worker具体的input 和output length信息也存下来。
        key = (self.model.model_name, self.get_key(), *(self._get_inp_key(arrive_times_list)))

        # print(f"key to check in _FAKE_SCHEDULING_RES: {key}")
        if key in _FAKE_SCHEDULING_RES:

            print(f"Reuse fake scheduling results")

            (self.cumsum_latencys_list, 
             self.cum_rng_nums_list_for_models, self.rng_starts_list_for_models, self.rng_ends_list_for_models,
             self.is_prefill_steps_list, 
             self.finish_times_list_for_models) = _FAKE_SCHEDULING_RES[key]
            
            self.total_latency_list = [cumsum_latencys[-1] if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]

            # # print(f"self.cumsum_latencys: {self.cumsum_latencys}")
            # print(f"from cache, key={key}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")
            # for k in _FAKE_SCHEDULING_RES.keys():
            #     print(k)
            # print(f"self.cum_rng_nums_list_for_models: {self.cum_rng_nums_list_for_models}")
            # print(f"self.rng_starts_list_for_models: {self.rng_starts_list_for_models}")
            # print(f"self.rng_ends_list_for_models: {self.rng_ends_list_for_models}")

            time3 = time.perf_counter()
            print(f"TIME--reuse fake scheduling: {time3 - time2}")

            # print(f"self.finish_times_list_for_models: {self.finish_times_list_for_models}")


            return self.total_latency_list
        else:

            print(f"DO FAKE SCHEDULING SEARCH!\n")

            # TODO: 这个地方的assert可能不对，因为我们可能是把request按照arrive时间排序的。相同arrive时间可以按照长度排序，
            # 但是这个排序很奇怪，因为model-level pipeline的情况，可能不会保证时刻都排好序。先不管这个？不管request的排序了。
            # 在写入output request的时候不排序，但是在读取input request的时候要排序？在读取input request的排序也不方便啊，
            # 要不然就不对来自所依赖的所有dp worker的input request进行统一排序了，感觉OK？理论上不同request的总长度分布是一样的。
            # 但是其实我们现在的写法会对output整体进行排序：应该改成对整体按照arrive-time进行排序，没有按长度再排序的过程了。
            # 要不然就把按长度排序这块去掉吧，感觉OK的。<= 因为感觉来自不同dp worker的output长度分布应该是一样的；这样从这些output
            # 中均匀间隔地获取input，得到的input长度分布也应该是一样的。
            # 
            # assert list(inp_lens) == sorted(inp_lens, reverse=True), f"The input lens of model {self.model.model_id} is not sorted!"


            for dp_id in range(self.dp_size):
                # divide the requests into dp_size groups evenly
                dp_inp_lens_for_models = self.dp_inp_lens_list_for_models[dp_id]
                dp_out_lens_for_models = self.dp_out_lens_list_for_models[dp_id]            
                dp_seq_ids_for_models = self.dp_inp_seq_ids_list_for_models[dp_id]
                dp_arrive_times_for_models = self.dp_arrive_times_list_for_models[dp_id]


                time3 = time.perf_counter()


                # support model-level pipeline
                # # NOTE: here we add the extra cost to the finish times --> CHANGE TO NOT ADDing EXTRA COST
                # exec_plan_0 = self.fused_exec_plans[0]
                (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list_for_models[dp_id], 
                    self.rng_starts_list_for_models[dp_id], self.rng_ends_list_for_models[dp_id], 
                    self.is_prefill_steps_list[dp_id], 
                    self.finish_times_list_for_models[dp_id]) = \
                        fake_scheduling.fake_FCFS_schedule_vertical_fuse(
                            inp_lens=list(dp_inp_lens_for_models[0]),out_lens=list(dp_out_lens_for_models[0]), 
                            arrive_times=dp_arrive_times_for_models[0], 
                            ref_seq_ids=dp_seq_ids_for_models[0],
                            # 
                            ref_seq_ids_list=dp_seq_ids_for_models[1:],
                            inp_lens_list=dp_inp_lens_for_models[1:],
                            out_lens_list=dp_out_lens_for_models[1:],
                            arrive_times_list=dp_arrive_times_for_models[1:],
                            # 
                            check_gap=check_gap,
                            max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                            max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                            block_size=self.infer_args.block_size, 
                            sort_input=sort_input, 
                            cost_estimate_args={
                                "cost_table":cost_table, 
                                "model_name":self.model.model_path, 
                                "exec_plan":self.get_key_single_dp_worker(), 
                                "sample_config":self.model.sample_config, 
                                "trust_remote_code":self.model.trust_remote_code, 
                                "revision":self.model.revision}
                            )

                # NOTE: since there is a precision issue, we use self.cumsum_latencys[-1] as tot_latency
                if len(self.cumsum_latencys_list[dp_id]) == 0:
                    self.total_latency_list[dp_id] = 0
                else:
                    self.total_latency_list[dp_id] = self.cumsum_latencys_list[dp_id][-1]

                time4 = time.perf_counter()
                print(f"TIME--fake scheduling vertical: {time4 - time3}")

            
            # self.infer_progress = infer_progress
            # store the metadata in the global cache
            _FAKE_SCHEDULING_RES[key] = (self.cumsum_latencys_list, self.cum_rng_nums_list_for_models, 
                                         self.rng_starts_list_for_models, self.rng_ends_list_for_models, 
                                         self.is_prefill_steps_list, self.finish_times_list_for_models,) 
                                        #  self.dp_inp_seq_ids_list)

            # # print(f"self.cumsum_latencys: {self.cumsum_latencys}")
            # print(f"compute from sketch, key={key}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")       


            time5 = time.perf_counter()  
            print(f"TIME--estimate time cost: {time5 - time1}")

            # print(f"self.finish_times_list_for_models: {self.finish_times_list_for_models}")


            return self.total_latency_list



    # data parallel + model-level pipeline
    # change name from get_total_latency to get_min_dp_latency --> get_max_dp_latency_considering_plan_group
    def get_max_dp_latency_considering_plan_group(self, cost_table: CostTable,
            check_gap: int, sort_input: bool, arrive_times_list: List[List[float]]):
        '''
            Input:
                extra_cost: the time to prepare (e.g., load) the model before running.
        '''

        print(f"exec_plan: {str(self)}")
        # print(f"arrive_times: {arrive_times_list}")
        

        if self.total_latency_list[0] == None:
            self.estimate_exec_time(cost_table, 
                check_gap=check_gap, sort_input=sort_input, arrive_times_list=arrive_times_list)

        print(f"exec plan latency list: {str(self), self.total_latency_list}")
        
        # print(f"finish times list: {self.finish_times_list}")

        return max(self.total_latency_list) + self.extra_cost


    # data parallel
    def get_max_dp_latency(self, cost_table: CostTable, sort_input: bool):
        """
            This function is only used in the baseline where we select the best exec plan for each LLM independently.
            NOTE: 
                1. as we select the best exec plan for each LLM independently, 
                we assume all input requests are available.
                i.e., no model-level pipeline is considered here.
        """
        if self.total_latency_list[0] == None:
            arrive_times_list = [[-1]*len(model.get_inp_seq_ids()) for model in self.model_list]
            self.estimate_exec_time(cost_table, 
                check_gap=1, sort_input=sort_input, arrive_times_list=arrive_times_list)

        return max(self.total_latency_list)



    # data parallel
    # return the results for each data parallel worker seperately
    def update_inp_out_seqlens_and_throughput_after_an_infer_stage(
            self, stop_time: float, cost_table: CostTable):
        '''
            1. compute valid throughput.
            2. Update the remaining seqlens after it finishes the current infer stage (until stop_time).
        '''
        new_inp_out_lens_list_for_models, valid_throughput_list, stop_iter_i_list = list(), list(), list()
        stage_stop_time = stop_time

        for dp_id in range(self.dp_size):
            cumsum_latencys = self.cumsum_latencys_list[dp_id]
            cache_stop_time_info = self.cache_stop_time_info_list[dp_id]
            cum_rng_nums_for_models = self.cum_rng_nums_list_for_models[dp_id]
            rng_starts_for_models = self.rng_starts_list_for_models[dp_id]
            rng_ends_for_models = self.rng_ends_list_for_models[dp_id]

            if len(cumsum_latencys) == 0:
                new_inp_out_lens_list_for_models.append(
                    (tuple(np.asarray([[] for _ in range(len(self.model_list))])), 
                     tuple(np.asarray([[] for _ in range(len(self.model_list))])), 
                     np.asarray([[] for _ in range(len(self.model_list))], dtype=np.int64)))
                valid_throughput_list.append(0)
                stop_iter_i_list.append(0)
                continue

            # print(f"in update_inp_out_seqlens_and_throughput_after_an_infer_stage: {str(self)}")
            # NOTE: due to the precision issue, sometimes stop_time > self.cumsum_latencys[-1], but it is impossible
            stop_time = min(cumsum_latencys[-1], stage_stop_time)

            # 0. check whether we have run this function for the corresponding stop iter.
            stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')
            if stop_iter_i in cache_stop_time_info:

                # print(f"stop_iter_i: {stop_iter_i}, stop time info: {self.cache_stop_time_info[stop_iter_i]}")

                new_inp_out_lens_for_models, valid_throughput = cache_stop_time_info[stop_iter_i]
                
                # print(f"stop_iter_i: {stop_iter_i}, len(cumsum_latencys): {len(cumsum_latencys)}")
                # print(f"new_inp_out_lens: {new_inp_out_lens}")

                new_inp_out_lens_list_for_models.append(new_inp_out_lens_for_models)
                valid_throughput_list.append(valid_throughput)
                stop_iter_i_list.append(stop_iter_i)
                continue

                # return cache_stop_time_info[stop_iter_i], stop_iter_i

            actual_stop_time = cumsum_latencys[stop_iter_i]

            # print(self.model.model_name, f"stop_iter_i:{stop_iter_i}")
            # print(self.model.model_name, f"rng_starts:{self.rng_starts}")
            # print(self.model.model_name, f"rng_ends:{self.rng_ends}")
            # print(self.model.model_name, f"cum_rng_nums:{self.cum_rng_nums}")


            # print(f"cum_rng_nums_for_models: {cum_rng_nums_for_models}")
            # print(f"rng_starts_for_models: {rng_starts_for_models}")
            # print(f"rng_ends_for_models: {rng_ends_for_models}")

            # 1. compute the seq infer progress after the infer stage

            # first flatten all metadata
            # indptr = np.cumsum([0]+[len(cum_rng_nums) for cum_rng_nums in cum_rng_nums_for_models])

            finished_lens = np.concatenate([fake_scheduling.get_info_at_stop_time(
                cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, 
                stop_time, stop_iter_i) for cum_rng_nums, rng_starts, rng_ends \
                    in zip(cum_rng_nums_for_models, rng_starts_for_models, rng_ends_for_models)])

            # finished_lens, remaining_lens = fake_scheduling.get_info_at_stop_time(
            #     self.decode_latencys, self.prefill_latencys, 
            #     self.is_prefill_steps, self.infer_progress, 
            #     stop_time)


            # print(f"in  update_inp_out_seqlens_and_throughput_after_an_infer_stage----")
            # print(f"stop_iter_i: {stop_iter_i}")
            # print(f"self.cum_rng_nums: {self.cum_rng_nums.tolist()}")
            # print(f"self.rng_starts: {self.rng_starts.tolist()}")
            # print(f"self.rng_ends: {self.rng_ends.tolist()}")
            print(str(self), f"finished_lens: len: {len(finished_lens)}: {finished_lens}")


            # 2. compute the valid throughput in the current infer stage
            # inp_lens, out_lens = self.model.get_inp_out_seqlens()
            # dp_inp_lens = self.dp_inp_lens_list[dp_id]
            # dp_out_lens = self.dp_out_lens_list[dp_id]

            dp_inp_lens_flattened = np.concatenate( self.dp_inp_lens_list_for_models[dp_id] )
            dp_out_lens_flattened = np.concatenate( self.dp_out_lens_list_for_models[dp_id] )

            # print(str(self), f"dp_inp_lens_flattened: len: {len(dp_inp_lens_flattened)}: {dp_inp_lens_flattened}")

            # print(self.model.model_name, f"old inp_lens:{inp_lens}, old remaining_lens:{out_lens}")
            print(self)

            # exec_plan_0 = self.fused_exec_plans[0]
            valid_throughput = fake_scheduling.comp_valid_throughput_at_stop_time(
                dp_inp_lens_flattened,
                finished_lens, actual_stop_time, cost_table,
                self.model.model_path, self.model.trust_remote_code, self.model.revision)

            # 3. update the remaining_seqlens
            dp_inp_lens_flattened = np.asarray(dp_inp_lens_flattened) + np.asarray(finished_lens)
            remaining_lens = np.asarray(dp_out_lens_flattened) - np.asarray(finished_lens)
            valid_indices = (remaining_lens>0)

            indptr = np.cumsum([0]+[len(_) for _ in self.dp_inp_lens_list_for_models[dp_id]])
            # convert flattened metadata back
            dp_inp_lens_for_models = [dp_inp_lens_flattened[indptr[i]:indptr[i+1]] for i in range(len(self.model_list))]
            remaining_lens_for_models = [remaining_lens[indptr[i]:indptr[i+1]] for i in range(len(self.model_list))]
            valid_indices_for_models = [valid_indices[indptr[i]:indptr[i+1]] for i in range(len(self.model_list))]

            # delete finished reqs
            dp_inp_lens_for_models = [tuple(vs[inds]) for vs, inds in zip(dp_inp_lens_for_models, valid_indices_for_models)]
            remaining_lens_for_models = [tuple(vs[inds]) for vs, inds in zip(remaining_lens_for_models, valid_indices_for_models)]

            # print(self.model.model_name, f"new inp_lens:{inp_lens[valid_indices]}, new remaining_lens:{remaining_lens[valid_indices]}")
            # print(f"valid_indices_for_models: {valid_indices_for_models}")

            # NOTE: we store the valid_indices as well
            cache_stop_time_info[stop_iter_i] = \
                [(tuple(dp_inp_lens_for_models), tuple(remaining_lens_for_models), valid_indices_for_models), \
                 valid_throughput]

            new_inp_out_lens_list_for_models.append(cache_stop_time_info[stop_iter_i][0])
            valid_throughput_list.append(valid_throughput)
            stop_iter_i_list.append(stop_iter_i)

            # print(f"stop_iter_i: {stop_iter_i}, stop time info: {self.cache_stop_time_info[stop_iter_i]}")
            
            # print(f"stop_iter_i: {stop_iter_i}, len(cumsum_latencys): {len(cumsum_latencys)}")
            # print(f"new_inp_out_lens: {cache_stop_time_info[stop_iter_i][0]}")
            
            # return self.cache_stop_time_info[stop_iter_i], stop_iter_i
            # # return (tuple(inp_lens[valid_indices]), tuple(remaining_lens[valid_indices])), valid_throughput


        return (new_inp_out_lens_list_for_models, valid_throughput_list), stop_iter_i_list






    # data parallel + model-level pipeline
    def update_fake_schedule_output_after_an_infer_stage(
            self, 
            old_inp_lens_list: List[List[int]], 
            new_inp_out_lens_list: List[List[List[int]]], 
            new_inp_lens_merged: List[int],
            new_out_lens_merged: List[int],
            new_inp_seq_ids_merged: List[int],
            stop_iter_i_list: List[int], 
            cost_table: CostTable, 
            ):
        '''
            This function is called when the exec plan is selected to run for an infer stage.
            Update:
                _FAKE_SCHEDULING_RES[model_name, exec_plan, new_inp_lens, new_out_lens]
            NOTE:
                we only call this function when after an infer stage, all the requests are available.
            NOTE: 
                1. ``new_inp_lens_merged`` is already sorted by seq ids.
                2. we only do update when there are unfinished requests.
        '''
        def to_tuple(vs):
            return tuple([tuple(_) for _ in vs])

        # print(f"in update_fake_schedule_output_after_an_infer_stage")
        # print(f"in / out seqlens: {self.model.get_inp_out_seqlens()}")

        # check whether all the input requests are available after this stage is finished
        # in the case where models are fused vertically, all the input requests are available only if 
        # all the models but the last one has finished and the last one's other inputs are available
        if sum([len(model.get_inp_seq_ids()) for model in self.model_list[:-1]]) == 0:
            # all the models but the last one has finished
            last_inp_available_time = max(\
                [max(self.dp_arrive_times_list_for_models[dp_id][-1]) \
                 for dp_id in range(self.dp_size)])
            stop_time = max([cumsum_latencys[stop_iter_i] if len(cumsum_latencys) > 0 else 0 \
                             for cumsum_latencys, stop_iter_i in \
                                zip(self.cumsum_latencys_list, stop_iter_i_list)])
            if last_inp_available_time > stop_time:
                # there are input not available after this stage finishes
                return
        else:
            # there are models except the last one not finished
            return


        cumsum_latencys_list = [list() for _ in range(self.dp_size)]
        cum_rng_nums_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        rng_starts_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        rng_ends_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        is_prefill_steps_list = [list() for _ in range(self.dp_size)]
        # finish_times_merged = np.asarray([-1]*self.model.ori_tot_inp_num)
        finish_times_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]


        # exec_plan_0 = self.fused_exec_plans[0]

        for dp_id in range(self.dp_size):
            
            old_inp_lens_for_models = old_inp_lens_list[dp_id]
            old_inp_lens = old_inp_lens_for_models[-1]
            # new_inp_lens, new_out_lens, new_inp_seq_ids = new_inp_out_lens_list[dp_id]
            stop_iter_i = stop_iter_i_list[dp_id]

            # 1. update the infer metadata after the infer stage
            (cumsum_latencys_list[dp_id], cum_rng_nums_list_for_models[dp_id][-1], 
                rng_starts_list_for_models[dp_id][-1], rng_ends_list_for_models[dp_id][-1], is_prefill_steps_list[dp_id], 
                finish_times_list_for_models[dp_id][-1], alive_old_indices) = \
                    fake_scheduling.update_fake_FCFS_schedule_metadata(
                        old_inp_lens, # new_inp_lens,
                        self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list_for_models[dp_id][-1], 
                        self.rng_starts_list_for_models[dp_id][-1], self.rng_ends_list_for_models[dp_id][-1], 
                        self.is_prefill_steps_list[dp_id],
                        self.infer_args.max_num_batched_tokens, stop_iter_i,
                        cost_table, 
                        model_name=self.model.model_path, 
                        exec_plan=self.get_key_single_dp_worker(), sample_config=self.model.sample_config, 
                        trust_remote_code=self.model.trust_remote_code, revision=self.model.revision
                        )
            
            # dp_seq_ids = self.dp_seq_ids_list[dp_id]
            # finish_times_merged[dp_seq_ids[alive_old_indices]] = finish_times_of_alive_seqs
            # finish_times_list[dp_id] = finish_times_of_alive_seqs
            assert (np.nonzero(new_inp_out_lens_list[dp_id][2][-1])[0] == alive_old_indices).all()





        # new_key = (self.model.model_name, self.get_key(), (to_tuple(new_inp_lens_merged), to_tuple(new_out_lens_merged)), \
        #            tuple([tuple([-1 - self.extra_cost]*len(new_inps)) for new_inps in new_inp_lens_merged]))

        # NOTE: 因为可能每个dp worker的inference进度不一样，所以当前每个dp worker剩下的request并不是均衡的，这个就会和我们想要的fake scheduling相矛盾，所以这个地方要把
        # 每个dp worker具体的input 和output length信息也存下来。
        new_key = (self.model.model_name, self.get_key(), 
                   (tuple([to_tuple(dp_data[0]) for dp_data in new_inp_out_lens_list]), tuple([to_tuple(dp_data[1]) for dp_data in new_inp_out_lens_list])), \
                   tuple([to_tuple([[-1 - self.extra_cost]*len(_) for _ in dp_data[0]]) for dp_data in new_inp_out_lens_list]))


        # print(f"update fake scheduling 2: new_key: {new_key}")
        _FAKE_SCHEDULING_RES[new_key] = \
            cumsum_latencys_list, cum_rng_nums_list_for_models, rng_starts_list_for_models, rng_ends_list_for_models, \
                is_prefill_steps_list, finish_times_list_for_models
            # finish_times_merged, new_inp_seq_ids_merged





    # def get_key(self):
    #     # the key of a exec plan is (tp, gpu_ratio, wldeg, cache_gpu_num)
    #     # data parallel
    #     # ==> (tp, gpu_ratio, wldeg, cache_gpu_num, dp_size)
    #     return self.fused_exec_plans[0].get_key()
    #     # return (self.num_worker, self.mem_per_comp_gpu, self.wld_degree, self.cache_gpu_num, self.dp_size)
    
    # def get_key_single_dp_worker(self):
    #     # the key of the exec plan for a data parallel worker is (tp, gpu_ratio, wldeg, cache_gpu_num)
    #     return self.fused_exec_plans[0].get_key_single_dp_worker()
    #     # return (self.num_worker, self.mem_per_comp_gpu, self.wld_degree, self.cache_gpu_num)


    def __str__(self) -> str:
        return f"{[str(model) for model in self.model_list]}, "\
            f"{self.get_key()}"
        # return f"{str(self.model)}, "\
        #     f"{self.get_key()}"
            # f"tp:{self.num_worker}, wld:{self.wld_degree}, "\
            # f"cache_gpu:{self.cache_gpu_num}, mem_r:{self.mem_per_comp_gpu}, "\
            # f"param_byte_per_comp_gpu:{self.param_byte_per_comp_gpu}, param_byte_per_cache_gpu:{self.param_byte_per_cache_gpu}"













class MyExecPlanGroup:
    """ My execution plan group definition. """
    def __init__(
        self,
        exec_plans: List[MyExecPlan], 
        cost_table: CostTable,
        last_stage_exec_plans: List[MyExecPlan],
        # model-level pipeline
        check_gap: int,
        sort_input: bool,
    ) -> None:
        
        print(f"building exec plan group: {[(_.model.get_base_model_ids(), _.get_key()) for _ in exec_plans]}\n", flush=True)

        self.exec_plans = exec_plans
        self.throughput = None
        self.infer_stage_latency = None
        self.comp_throughput = None
        # self.tot_worker_num = None

        # model-level pipeline
        self.inp_exec_plan_dict: Dict[MyExecPlan, List[MyExecPlan]] = defaultdict(list)
        self._topological_sort()
        # --------------------------

        self.valid_throughputs: List[float] = list()
        # self.extra_prepare_costs: Dict[MyExecPlan, float] = dict()
        # self.tmp_remaining_lens_list: List[List[int]] = list()
        self.tmp_inp_out_lens_list: \
            List[Union[Tuple[List[int], List[int], List[int]],  Tuple[List[int], List[int], List[int]]]] = list()
        self.tmp_remaining_decode_flops_after_infer_stage: List[float] = list()
        self.tmp_stop_iter_i_list: List[int] = list()
        self.compute_infer_stage_data(cost_table=cost_table, 
                                      last_stage_exec_plans=last_stage_exec_plans, 
                                      check_gap=check_gap, sort_input=sort_input)



    def get_involved_base_model_num(self):
        return sum([len(exec_plan.get_base_model_ids()) for exec_plan in self.exec_plans])


    def get_involved_fused_models(self):
        return [exec_plan.model for exec_plan in self.exec_plans if isinstance(exec_plan, MyVerticalFusedExecPlan)]


    # data parallel
    # TODO: 感觉这个函数的写法有问题，没有考虑到一个exec stage中可能有两个相同model的情况，比如这两个model有依赖关系；
    # 或者这两个model的input不同。但是感觉就先这样吧。因为schedule的阶段还没提供相应支持。
    def comp_extra_prepare_costs(
            self, cost_table: CostTable, last_stage_exec_plans: List[MyExecPlan]):
        '''
            Compute the extra prepare costs for each exec plan in this group.
            There will be extra prepare cost if:
                (1) the exec plan of the same model changes.
                NOTE: if only the memory changes, will there be extra cost? Yes, but we currently do not consider this.
        '''
        model_exec_plans = { exec_plan.model : exec_plan for exec_plan in last_stage_exec_plans }
        for exec_plan in self.exec_plans:
            model = exec_plan.model
            if model in model_exec_plans:
                last_exec_plan = model_exec_plans[model]
                if exec_plan.get_key() == last_exec_plan.get_key():
                    # we did not change the exec_plan, so there is no extra cost
                    # self.extra_prepare_costs.append(0.0)
                    # self.extra_prepare_costs[exec_plan] = 0.0
                    exec_plan.set_extra_cost(0.0)
                else:
                    # we change the exec_plan for the model, there is extra cost
                    # self.extra_prepare_costs.append(
                    #     cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                    # )
                    # self.extra_prepare_costs[exec_plan] = \
                    #     cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                    exec_plan.set_extra_cost(
                        cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                    )
            else:
                # there is extra cost to load the model
                # self.extra_prepare_costs.append(
                #     cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                # )
                # self.extra_prepare_costs[exec_plan] = \
                #     cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                exec_plan.set_extra_cost(
                    cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                )




    def _topological_sort(self):
        """
            This function returns the list of exec_plans in topological order.
            Update:
                1. self.exec_plans: 
                    sorted list of exec plans;
                2. self.inp_exec_plan_dict: 
                    the dependent input exec plans of each exec plan in this exec plan group.
            NOTE:
                This function is wrong because a model's input models may be finished (so not in the current plan group)
        """
        def get_mapped_model_ids_in_group(node_mapping, model_ids):
            """
                Get the mapped values of the input model_ids accoding to the node mapping.
                I.e., get the corresponding model ids in the new system when accepting the newly fused models.
                NOTE: only get the mapped model id if the id is in the current exec plan group.
            """
            # print(f"node_mapping: {node_mapping}, model_ids: {model_ids}")
            return set([node_mapping[i] for i in model_ids if i in node_mapping])

        model_exec_plan_mapping: Dict[int, MyExecPlan] = \
            {exec_plan.model.model_id:exec_plan for exec_plan in self.exec_plans}
        
        # get the model_id mapping from base model id to fused/base model id (or from fuse to fuse)
        # NOTE: 这里一个需要注意的点是：我们现在允许合并两个fused model成一个更大的fused model，但是这两个fused model需要都还没有开始inference
        # node_mapping = {exec_plan.model.model_id:exec_plan.model.model_id for exec_plan in self.exec_plans}
        node_mapping = {ori:exec_plan.model.model_id \
                             for exec_plan in self.exec_plans for ori in exec_plan.model.get_base_model_ids() }

        # the model ids are in the new system with newly fused models
        # all_model_ids = set(model_exec_plan_mapping.keys())

        inp_model_ids_dict: Dict[MyExecPlan, List[int]] = dict()

        # print(f"all exec plans in this group:\n")
        # print(f"{[(exec_plan.model.model_id, exec_plan.model.get_base_model_ids(), exec_plan.get_key()) for exec_plan in self.exec_plans]}")

        # get self.inp_exec_plan_dict
        for exec_plan in self.exec_plans:
            inp_model_ids_this_stage = get_mapped_model_ids_in_group(node_mapping, exec_plan.model.inp_base_model_ids)
            # inp_model_ids_this_stage = set.intersection(set(mapped_inp_model_ids), all_model_ids)
            inp_exec_plans_this_stage = [model_exec_plan_mapping[model_id] for model_id in inp_model_ids_this_stage]
            self.inp_exec_plan_dict[exec_plan] = inp_exec_plans_this_stage
            inp_model_ids_dict[exec_plan] = inp_model_ids_this_stage
        
        # NOTE: we do not need to sort the exec plans, as we already ensure the topological order when we generate them
        # NOTE: but we keep the sort below anyway

        sorted_plans: List[MyExecPlan] = list()
        sorted_model_ids = set()
        # iteratively add LLMs which are ready
        while len(sorted_plans) < len(self.exec_plans):
            for model_id, exec_plan in model_exec_plan_mapping.items():
                if model_id in sorted_model_ids:
                    continue
                if inp_model_ids_dict[exec_plan].issubset(sorted_model_ids):
                    sorted_plans.append(exec_plan)
                    sorted_model_ids.add(model_id)

        # reorder self.exec_plans 
        self.exec_plans = sorted_plans




    def _get_arrive_times_base_model_limited_version(self, inp_seq_ids: List[int], inp_exec_plans: List[MyExecPlan]):
        """
            Compute the input arrive times for this exec plan.
            NOTE: the finish_times are in the order of model.inp_seq_ids. [感觉这个影响不会太大？]
        """
        # inp_seq_ids: List[int] = exec_plan.model.get_inp_seq_ids()
        # inp_exec_plans = self.inp_exec_plan_dict[exec_plan]

        # print(f"inp_exec_plans: {[str(exec_plan) for exec_plan in inp_exec_plans]}")

        # 这个地方很奇怪，因为inp_plan 可能是个fused plan
        # NOTE：但是我们只支持线性的依赖关系 (linear dependency) 的model vertical fusion，
        # 所以可以直接假设依赖的是inp_plan里的最后一个base plan。
        # 要支持别的更复杂的情况也不是不行。

        # print(f"in _get_arrive_times_base_model: inp_seq_ids: {inp_seq_ids}")

        if len(inp_exec_plans) > 0:
            # NOTE: here we directly use inp_plan.get_exec_plans()[-1]
            finish_times = np.asarray([inp_plan.get_finish_times_merged(inp_seq_ids) \
                                       + inp_plan.extra_cost \
                                    for inp_plan in inp_exec_plans])
            finish_times = np.max(finish_times, axis=0)
        else:
            # no input exec plan, i.e., all inputs are available at the beginning
            finish_times = np.asarray([-1]*len(inp_seq_ids))
        return finish_times



    # support vertical fusion of models
    def _get_arrive_times_limited_version(self, exec_plan: MyExecPlan):

        """
            NOTE: 我们现在假设只支持自环的vertical fuse, i.e., fused 的models作为整体，其input与第一个model的input相同，
            其output与最后一个model的output相同。
        """

        print("base model information: ", [(model.model_name, model.model_id, model.get_base_model_ids(), isinstance(model, MyFusedModelInfor)) for model in exec_plan.get_base_models()])

        inp_seq_ids: List[int] = exec_plan.get_base_models()[0].get_inp_seq_ids()
        inp_exec_plans = self.inp_exec_plan_dict[exec_plan]
        
        finish_times_list = [self._get_arrive_times_base_model(inp_seq_ids, inp_exec_plans)] + \
            [self._get_arrive_times_base_model(model.get_inp_seq_ids(), []) for model in exec_plan.get_base_models()[1:]]
        
        if isinstance(exec_plan, MyVerticalFusedExecPlan):
            # print(f"is MyVerticalFusedExecPlan: finish_times_list: {finish_times_list}")

            return finish_times_list
        else:
            # print(f"is Basic exec plan: finish_times_list: {finish_times_list}")

            return finish_times_list[0]




    def _get_arrive_times_base_model(self, inp_seq_ids: List[int], inp_info: List[Tuple[MyExecPlan, int]]):
        """
            Compute the input arrive times for this exec plan.
            NOTE: the finish_times are in the order of model.inp_seq_ids. [感觉这个影响不会太大？]
            Input:
                inp_info: list of tuple (inp exec plan, the base model id in the inp exec plan)
        """
        # inp_seq_ids: List[int] = exec_plan.model.get_inp_seq_ids()
        # inp_exec_plans = self.inp_exec_plan_dict[exec_plan]

        # print(f"inp_exec_plans: {[str(exec_plan) for exec_plan in inp_exec_plans]}")

        # 这个地方很奇怪，因为inp_plan 可能是个fused plan
        # NOTE：但是我们只支持线性的依赖关系 (linear dependency) 的model vertical fusion，
        # 所以可以直接假设依赖的是inp_plan里的最后一个base plan。
        # 要支持别的更复杂的情况也不是不行。

        # print(f"in _get_arrive_times_base_model: inp_seq_ids: {inp_seq_ids}")
        # print(f"inp_info: {[(_[0].model.model_id, _[0].model.get_base_model_ids(), _[1]) for _ in inp_info]}")

        if len(inp_info) > 0:
            # NOTE: here we directly use inp_plan.get_exec_plans()[-1]
            finish_times = np.asarray([inp_plan.get_finish_times_merged(inp_seq_ids, model_ind) \
                                       + inp_plan.extra_cost \
                                    for inp_plan, model_ind in inp_info])

            # print(f"the original finish_times lists we get: {finish_times}")

            finish_times = np.max(finish_times, axis=0)
        else:
            # no input exec plan, i.e., all inputs are available at the beginning
            finish_times = np.asarray([-1]*len(inp_seq_ids))
        return finish_times



    # support vertical fusion of models
    def _get_arrive_times(self, exec_plan: MyExecPlan):

        """
            NOTE: 这个版本支持fused model中各个base model存有自己的output，都可以被外部的model利用。
        """

        print("base model information: ", [(model.model_name, model.model_id, model.get_base_model_ids(), isinstance(model, MyFusedModelInfor)) for model in exec_plan.get_base_models()])

        inp_exec_plans = self.inp_exec_plan_dict[exec_plan]
        in_exec_plans_base_model_ids = [in_exec_plan.get_base_model_ids() for in_exec_plan in inp_exec_plans]
        print(f"in_exec_plans_base_model_ids: {in_exec_plans_base_model_ids}")
        finish_times_list: List[List[float]] = list()

        # get the inp arrive times for each base model of the exec_plan
        for base_model in exec_plan.get_base_models():
            
            inp_info: List[MyExecPlan, int] = list()

            print(f"base_model: {base_model.get_base_model_ids()} its input model ids: {base_model.input_model_ids}")

            # 1. get the inp plans and the corresponding base model ids in the inp plans that this base model depends on
            for in_model_id in base_model.inp_base_model_ids:
                for i, model_ids in enumerate(in_exec_plans_base_model_ids):
                    if in_model_id in model_ids:
                        for j, model_id in enumerate(model_ids):
                            if in_model_id == model_id:
                                inp_info.append((inp_exec_plans[i], j))
            
            # 2. get the inp arrive times
            inp_seq_ids: List[int] = base_model.get_inp_seq_ids()
            finish_times_list.append(self._get_arrive_times_base_model(inp_seq_ids, inp_info))

        
        if isinstance(exec_plan, MyVerticalFusedExecPlan):
            # print(f"is MyVerticalFusedExecPlan: finish_times_list: {finish_times_list}")

            return finish_times_list
        else:
            # print(f"is Basic exec plan: finish_times_list: {finish_times_list}")

            return finish_times_list[0]









    def compute_infer_stage_data(
            self, cost_table: CostTable, last_stage_exec_plans: List[MyExecPlan], 
            check_gap: int, sort_input: bool):
        '''
            1. Compute the infer stage time.
            2. Update the infer progress of each model involved.
            Modify:
                (1) self.valid_throughputs, (2) self.tmp_inp_out_lens_list
                (3) self.tmp_remaining_decode_flops_after_infer_stage [deleted]
                (4) self.tmp_stop_iter_i_list
                (5) the total latency of the current infer stage.
        '''

        print(f"INIT plan group: {[(str(exec_plan), exec_plan.model.model_id, isinstance(exec_plan, MyVerticalFusedExecPlan)) for exec_plan in self.exec_plans]}")

        # 0. Compute the extra preparation time.
        self.comp_extra_prepare_costs(cost_table, last_stage_exec_plans)
        
        # 1. Compute the infer stage time.
        # latencys = [exec_plan.get_total_latency(cost_table) \
        #         for exec_plan in self.exec_plans]
        # 
        # latencys = [exec_plan.get_total_latency(cost_table) + extra_cost \
        #         for exec_plan, extra_cost in zip(self.exec_plans, self.extra_prepare_costs)]
        # data parallel
        # latencys = [exec_plan.get_min_dp_latency(cost_table) + extra_cost \
        #         for exec_plan, extra_cost in zip(self.exec_plans, self.extra_prepare_costs)]
        latencys = [exec_plan.get_max_dp_latency_considering_plan_group(
                        cost_table, check_gap, sort_input, self._get_arrive_times(exec_plan),
                        # self.extra_prepare_costs[exec_plan],
                    ) for exec_plan in self.exec_plans]
        latency = min(latencys)

        print(f"stage latency: {latency}")
        print(f"{[exec_plan.model.get_base_model_ids() for exec_plan in self.exec_plans]}")

        # 2. Update the infer progress of each model involved.
        # for exec_plan in self.exec_plans:
        # for exec_plan, extra_cost in zip(self.exec_plans, self.extra_prepare_costs):
        for exec_plan in self.exec_plans:
            extra_cost = exec_plan.extra_cost
            # data parallel
            # new_inp_out_lens stores a list of all dp workers' results (list length is dp_size)
            (new_inp_out_lens, valid_throughput), stop_iter_i = \
                exec_plan.update_inp_out_seqlens_and_throughput_after_an_infer_stage(
                    stop_time=latency-extra_cost, cost_table=cost_table)
            self.tmp_inp_out_lens_list.append(new_inp_out_lens)
            # self.tmp_remaining_decode_flops_after_infer_stage.append(
            #     exec_plan.model.get_remaining_decode_flops_after_infer_stage(cost_table, *new_inp_out_lens))
            self.valid_throughputs.append(valid_throughput)
            self.tmp_stop_iter_i_list.append(stop_iter_i)
        
        self.infer_stage_latency = latency

        print(f"throughput: {self.get_throughput()}")


    def get_throughput_no_data_parallel(self):
        '''
        Get the total throughput of the given plan group.
        NOTE: we also consider the extra preparation cost here.
        '''    
        if self.throughput == None:
            assert len(self.valid_throughputs) > 0
            # self.throughput = sum(self.valid_throughputs)
            throughputs = [v*exec_plan.cumsum_latencys[stop_iter_i]/self.infer_stage_latency \
                for v, exec_plan, stop_iter_i \
                    in zip(self.valid_throughputs, self.exec_plans, self.tmp_stop_iter_i_list)]
            self.throughput = sum(throughputs)

        return self.throughput
    

    # data parallel
    def get_throughput(self):
        '''
        Get the total throughput of the given plan group.
        NOTE: we also consider the extra preparation cost here.
        '''    
        if self.throughput == None:
            assert len(self.valid_throughputs) > 0
            # self.throughput = sum(self.valid_throughputs)
            throughputs = [sum([v*cumsum_latencys[stop_iter_i]/self.infer_stage_latency \
                            for v, cumsum_latencys, stop_iter_i in \
                                zip(v_list, exec_plan.cumsum_latencys_list, stop_iter_i_list) \
                                    if len(cumsum_latencys) > 0]) \
                for v_list, exec_plan, stop_iter_i_list \
                    in zip(self.valid_throughputs, self.exec_plans, self.tmp_stop_iter_i_list)]
            self.throughput = sum(throughputs)

        return self.throughput
    

    def get_comp_throughput_only_no_data_parallel(self):
        '''
        Get the total throughput of the given plan group, do not consider the extra preparation cost.
        '''
        if self.comp_throughput == None:
            assert len(self.valid_throughputs) > 0
            self.comp_throughput = sum(self.valid_throughputs)
        return self.comp_throughput

    # data parallel
    def get_comp_throughput_only(self):
        '''
        Get the total throughput of the given plan group, do not consider the extra preparation cost.
        '''
        if self.comp_throughput == None:
            assert len(self.valid_throughputs) > 0
            self.comp_throughput = sum([sum(v_list) for v_list in self.valid_throughputs])
        return self.comp_throughput

    
    def get_infer_stage_latency(self):
        return self.infer_stage_latency
    

    # TODO: 其实这些地方的key应该是(inp,out) pair的集合，以及每个pair出现的次数
    def get_model_states_after_infer_stage(self, cost_table: CostTable):
        # the model state: (model name, model remaining decode flops after the infer stage)
        # return tuple(sorted([(exec_plan.model.model_name, remaining_decode_flops) \
        #         for exec_plan, remaining_decode_flops \
        #             in zip(self.exec_plans, self.tmp_remaining_decode_flops_after_infer_stage)]))

        # return tuple(sorted([(exec_plan.model.model_name, exec_plan.model.model_id, inp_out_lens) \
        #                      for exec_plan, inp_out_lens \
        #                         in zip(self.exec_plans, self.tmp_inp_out_lens_list)]))
        
        # print(f"self.tmp_inp_out_lens_list: {self.tmp_inp_out_lens_list}")
        # print([str(exec_plan) for exec_plan in self.exec_plans])

        def get_model_state(exec_plan, inp_out_lens):
            if isinstance(exec_plan, MyVerticalFusedExecPlan):
                print(f"MyVerticalFusedExecPlan")
                return (
                [exec_plan.model.model_name]*len(exec_plan.model.get_base_model_ids()), 
                tuple(exec_plan.model.get_base_model_ids()), 
                tuple([tuple(_) for _ in exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[0]]),
                tuple([tuple(_) for _ in exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[1]]),
                )
                return (
                exec_plan.model.model_name, tuple(exec_plan.model.get_base_model_ids()), 
                tuple(np.concatenate([tuple(np.concatenate(inps)) for inps, outs, indices in inp_out_lens])), 
                tuple(np.concatenate([tuple(np.concatenate(outs)) for inps, outs, indices in inp_out_lens])) 
                )
            elif isinstance(exec_plan, MyExecPlan):
                print(f"MyExecPlan")
                return (
                [exec_plan.model.model_name], 
                tuple(exec_plan.model.get_base_model_ids()), 
                [tuple(exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[0])], 
                [tuple(exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[1])] 
                )
                return (
                exec_plan.model.model_name, tuple(exec_plan.model.get_base_model_ids()), 
                tuple(np.concatenate([inps for inps, outs, indices in inp_out_lens])), 
                tuple(np.concatenate([outs for inps, outs, indices in inp_out_lens])) 
                )

        ret = list()
        for exec_plan, inp_out_lens in zip(self.exec_plans, self.tmp_inp_out_lens_list):
            states = get_model_state(exec_plan, inp_out_lens)
            names, model_ids, inp_lens, out_lens = states
            ret.extend(zip(names, model_ids, inp_lens, out_lens))

        return tuple(sorted(ret))
        
        return tuple(sorted([get_model_state(exec_plan, inp_out_lens) for exec_plan, inp_out_lens \
                in zip(self.exec_plans, self.tmp_inp_out_lens_list)]))


        # return tuple(sorted([(
        #     exec_plan.model.model_name, tuple(exec_plan.model.get_base_model_ids()), 
        #     tuple(np.concatenate([inps for inps, outs, indices in inp_out_lens])), 
        #     tuple(np.concatenate([outs for inps, outs, indices in inp_out_lens])) 
        #     ) for exec_plan, inp_out_lens \
        #         in zip(self.exec_plans, self.tmp_inp_out_lens_list)]))
    
    def get_model_states_before_infer_stage(self):
        ret = list()
        for exec_plan in self.exec_plans:
            ret.extend([model.get_state() for model in exec_plan.get_base_models()])
        return tuple(sorted(ret))
        # return tuple(sorted([(exec_plan.model.get_state()) for exec_plan in self.exec_plans]))
    

    def update_model_inp_out_lens_no_data_parallel(self, cost_table: CostTable):
        # print(f"WHEN APPLY A PLAN GROUP: {self}")

        # formally change the model's remaining seq lens
        for exec_plan, inp_out_lens, stop_iter_i in zip(self.exec_plans, self.tmp_inp_out_lens_list, self.tmp_stop_iter_i_list):
            old_inp_lens, _ = exec_plan.model.get_inp_out_seqlens()

            
            exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table)
        
            # update the fake FCFS schedule metadata as well when the model is not finished
            if not exec_plan.model.is_finished():
                # print(f"old out_lens: {_}")
                exec_plan.update_fake_schedule_output_after_an_infer_stage_no_data_parallel(
                    old_inp_lens, *inp_out_lens,
                    stop_iter_i, cost_table)
            
            # print(f"applying exec plan: {str(exec_plan)}")
            # print(f"old inp lens: {old_inp_lens}")
            # print(f"old out lens: {_}")
            # print(f"new inp lens: {exec_plan.model.get_inp_out_seqlens()[0]}")
            # print(f"new out lens: {exec_plan.model.get_inp_out_seqlens()[1]}")
        
        return
        for i in range(len(self.exec_plans)):
            exec_plan = self.exec_plans[i]
            inp_out_lens = self.tmp_inp_out_lens_list[i]
            remaining_decode_flops = self.tmp_remaining_decode_flops_after_infer_stage[i]
            exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)




    # data parallel + model-level pipeline
    def update_model_inp_out_lens(self, cost_table: CostTable):
        # print(f"WHEN APPLY A PLAN GROUP: {self}")

        # formally change the model's remaining seq lens
        for exec_plan, inp_out_lens, stop_iter_i in \
            zip(self.exec_plans, self.tmp_inp_out_lens_list, self.tmp_stop_iter_i_list):
            # old_inp_lens, _ = exec_plan.model.get_inp_out_seqlens()
            old_inp_lens = exec_plan.get_dp_inp_lens_list_for_models()

            # we sort the inps by their seq ids
            # merged_inp_out_lens = merge_inp_out_lens_of_data_parallel_workers(inp_out_lens)
            merged_inp_out_lens = exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)
            
            print(f"UPDATE MODELS AFTER SELECTING AN EXEC PLAN-------------\n")
            print(f"exec_plan: {str(exec_plan)}, model_ids: {exec_plan.model.get_base_model_ids()}")
            # print(f"inp_out_lens: {inp_out_lens}")
            # print(f"merged_inp_out_lens: {merged_inp_out_lens}")

            # exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table)
            exec_plan.model.update_inp_out_seqlens(*merged_inp_out_lens, cost_table)
        
            # update the fake FCFS schedule metadata as well when the model is not finished
            if not exec_plan.model.is_finished():
                # print(f"old out_lens: {_}")
                exec_plan.update_fake_schedule_output_after_an_infer_stage(
                    old_inp_lens, inp_out_lens, 
                    *merged_inp_out_lens,
                    stop_iter_i, cost_table)
            
            # print(f"applying exec plan: {str(exec_plan)}")
            # print(f"old inp lens: {old_inp_lens}")
            # print(f"old out lens: {_}")
            # print(f"new inp lens: {exec_plan.model.get_inp_out_seqlens()[0]}")
            # print(f"new out lens: {exec_plan.model.get_inp_out_seqlens()[1]}")
        
        return
        for i in range(len(self.exec_plans)):
            exec_plan = self.exec_plans[i]
            inp_out_lens = self.tmp_inp_out_lens_list[i]
            remaining_decode_flops = self.tmp_remaining_decode_flops_after_infer_stage[i]
            exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)



    
    def __str__(self):
        '''
        Get the string to represent a plan group: 
            the exec_plan settings + the remaining_lens after the current infer stage.
        '''
        return str(sorted([f"{str(exec_plan)}: {str(exec_plan.total_latency_list)}s" #+ str(inp_lens) + str(out_lens) \
                           for exec_plan in self.exec_plans])) \
                                + ' ' + str(self.infer_stage_latency) \
                                + ' ' + str(self.get_throughput())
    
        return str(sorted([str(exec_plan) #+ str(inp_lens) + str(out_lens) \
                           for exec_plan, (inp_lens, out_lens) \
                            in zip(self.exec_plans, self.tmp_inp_out_lens_list)])) \
                                + ' ' + str(self.infer_stage_latency) \
                                + ' ' + str(self.get_throughput())
    
    
    
    def __len__(self):
        return len(self.exec_plans)




class MyExecPlanGroupSeq:
    """ My execution plan group sequence definition. """
    def __init__(
        self,
        tot_flops: float,
        plan_group_seq: List[MyExecPlanGroup], 
        time_seq: List[float]
    ) -> None:
        self.tot_flops = tot_flops
        self.plan_group_seq = plan_group_seq
        self.time_seq = time_seq
    
    def get_last_stage_exec_plans(self) -> List[MyExecPlan]:
        if len(self.plan_group_seq) == 0:
            return []
        else:
            return self.plan_group_seq[-1].exec_plans
    
    def get_tot_time(self):
        return sum(self.time_seq)
    

    def get_valid_throughput(self):
        # in fact, this is throughput, not valid (only computation) throughput
        if len(self.time_seq) == 0:
            return 0
        else:
            return self.tot_flops / self.get_tot_time()
    
    def get_tmp_throughput_after_adding_a_plan_group(self, plan_group: MyExecPlanGroup):
        flops = sum([group.get_throughput()*group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])
        latency = sum([group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])        
        return flops / latency
    

    def get_tmp_only_comp_throughput_after_adding_a_plan_group(self, plan_group: MyExecPlanGroup):
        flops = sum([group.get_throughput()*group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])
        
        print("DEBUG: ", [group.valid_throughputs for group in self.plan_group_seq+[plan_group]])
        
        latency = sum([group.get_throughput()*group.get_infer_stage_latency()/group.get_comp_throughput_only() \
                       for group in self.plan_group_seq+[plan_group]])        
        return flops / latency
    


    def append_plan_group(self, plan_group):
        self.plan_group_seq.append(plan_group)
    
    def append_exec_time(self, comp_time):
        self.time_seq.append(comp_time)


    def pop_one_stage(self):
        self.plan_group_seq = self.plan_group_seq[:-1]
        self.time_seq = self.time_seq[:-1]
    def set_plan_group_and_time(self, plan_group_seq, time_seq):
        self.plan_group_seq = plan_group_seq
        self.time_seq = time_seq
    def __str__(self) -> str:
        if (len(self.plan_group_seq) == 0) or (self.plan_group_seq[0] == None):
            return f"{self.plan_group_seq, self.time_seq}"
        else:
            return f"{[[str(exec_plan) for exec_plan in group.exec_plans] for group in self.plan_group_seq]} "\
                f"{self.time_seq} "\
                f"{sum(self.time_seq)} "\
                f"{self.get_valid_throughput()}"
        








class MyModelSystem:
    """ My multi-model system class. Contains the computation graph of this system. """
    def __init__(
        self,
        model_list: List[MyModelInfor],
        out_edge_dict: Dict[int, List[int]],
        # parameters for inp/out lens correction
        cost_table: CostTable, inp_merger, outlen_generator, 
        need_correct_inp_out_lens: bool,
        out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]] = None,
    ) -> None:
        # here we have self.model_list[i].model_id = i --> change to use model dict now
        self.model_dict: Dict[int, MyModelInfor] = {model.model_id: model for model in model_list}
        # model-level pipeline
        self.all_level_model_ids: List[List[int]] = list()
        # here we use edge_dict[i] to store the output edges of model i
        self.out_edge_dict = defaultdict(list, out_edge_dict)
        self.cost_table = cost_table
        self.inp_merger = inp_merger
        self.outlen_generator = outlen_generator
        # # generate inp_edge_dict for use
        # self.inp_edge_dict = defaultdict(list)
        # for inp in out_edge_dict:
        #     for out in out_edge_dict[inp]:
        #         self.inp_edge_dict[out].append(inp)

        
        # set the inp_edges information for each LLM
        for model in self.model_dict.values():
            model.input_model_ids = list()
        for inp in out_edge_dict:
            for out in out_edge_dict[inp]:
                self.model_dict[out].input_model_ids.append(inp)

        print(f"out_edge_dict: {out_edge_dict}")
        for model_id, model in self.model_dict.items():
            print(f"inp model ids of model {model_id} = {model.get_base_model_ids()}: {model.input_model_ids}")
        print("original inp/out lens of each model: ")
        for model_id, model in self.model_dict.items():
            print(f"{model_id}")
            print(f"{model.get_inp_out_seqlens()[0]}")
            print(f"{model.get_inp_out_seqlens()[1]}")

        # correct inp/out lens of models considering LLM dependency
        # TODO: 这个地方可以优化一下，一步到位生成所有模型的正确的inp/out lens
        if need_correct_inp_out_lens:
            self._get_inp_out_lens_considering_LLM_dependency(
                cost_table=cost_table, inp_merger=inp_merger, outlen_generator=outlen_generator, out_req_id_mapping=out_req_id_mapping)
        
        print("correct inp/out lens of each model: ")
        for model_id, model in self.model_dict.items():
            print(f"{model_id}")
            print(f"{model.get_inp_out_seqlens()[0]}")
            print(f"{model.get_inp_out_seqlens()[1]}")




    def check_finish_states_accuracy(self):
        for model in self.model_dict.values():
            print(f"model {model.model_id}, inp/out_lens: {model.get_inp_out_seqlens()}")
        for model in self.model_dict.values():
            inp_model_states = [self.model_dict[i].is_finished() for i in model.input_model_ids]
            if model.is_finished() and (False in inp_model_states):
                assert False








    def fuse_similar_models_in_a_chain(
            self,
            tot_gpu_num, byte_per_gpu, cost_table: CostTable,
            # baseline: str, 
            check_gap: int, sort_input: bool,
            similar_threshold: float):
        """
            This function tries to fuse models in a chain (these models can be fused vertically) which has similar performance given the same GPU resources.
            Output: a new model system with fused models.

            NOTE: call this function before the search starts.
        """

        print(f"\n\nTRYING FUSING SOME MODELS AT THE BEGINNING!\n\n")

        # NOTE: assume self.get_all_level_models is already called right before this function is called.

        visit_model_level = -1
        
        # { the model id: the comp throughputs of different tp_size setting for the model }
        comp_throughputs_dict: Dict[int, List[float]] = dict()
        # { the first model id in the fused model: all the model ids in the fused model }
        fused_models: Dict[int, List[int]] = dict()

        while True:
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)
            if len(cand_models) == 0:
                break

            for to_fuse_model in cand_models:
                
                # 1. compute its comp throughput vector
                exec_plans = get_possible_exec_plans(to_fuse_model, tot_gpu_num, byte_per_gpu, cost_table, baseline='ours', sort_input=sort_input)
                plan_groups = [MyExecPlanGroup([exec_plan], cost_table=cost_table, last_stage_exec_plans=[],
                                                check_gap=check_gap, sort_input=sort_input,) for exec_plan in exec_plans]
                comp_throughput_vecs = np.asarray([plan_group.get_comp_throughput_only() for plan_group in plan_groups])


                is_fused: bool = False
                # 2. check whether this model can be fused vertically
                # to_fuse_inp_model_ids must only contain base model ids
                to_fuse_inp_model_ids = to_fuse_model.input_model_ids
                for first_model_id, model_ids_fused in fused_models.items():
                    fused_model_inp_base_model_ids = self.model_dict[first_model_id].inp_base_model_ids
                    if _meet_vertical_fuse_condition(to_fuse_inp_model_ids, model_ids_fused, fused_model_inp_base_model_ids):
                        # this model can be fused topologically

                        # 3. check whether the comp_throughput_vecs is similar enough
                        diff = np.abs((comp_throughput_vecs-comp_throughputs_dict[first_model_id])/comp_throughputs_dict[first_model_id])
                        if (diff < similar_threshold).all():
                            # we can directly fuse this model with model_ids_fused
                            fused_models[first_model_id].append(to_fuse_model.model_id)
                            is_fused = True
                            break
            

                # there is no fused model to join
                if not is_fused:
                    comp_throughputs_dict[to_fuse_model.model_id] = comp_throughput_vecs
                    fused_models[to_fuse_model.model_id] = [to_fuse_model.model_id]
            
        
        
        print(f"\n\n FINISH FUSING SOME MODELS AT THE BEGINNING!\n\n")
        print(f"comp_throughputs_dict: {comp_throughputs_dict}")
        print(f"fused_models: {fused_models}")
        
        
        # generate a new model system with the fused models
        fused_model_objs = list()
        for model_ids in fused_models.values():
            if len(model_ids) > 1:
                fused_model_objs.append( MyFusedModelInfor([self.model_dict[model_id] for model_id in model_ids]) )
        
        return self.gen_new_model_sys_with_fused_models(fused_model_objs)
        















    def gen_new_model_sys_with_fused_models(self, fused_model_list: List[MyFusedModelInfor]):
        node_mapping = {ori:fused_model.model_id for fused_model in fused_model_list for ori in fused_model.get_base_model_ids()}
        # NOTE: we need to support the case where we fuse some models at the beginning before the search process
        #       i.e., we may fuse two fused models into a bigger fused model
        for model_id, model in self.model_dict.items():
            base_model_ids = model.get_base_model_ids()
            if base_model_ids[0] in node_mapping:
                assert False not in [node_mapping[_] == node_mapping[base_model_ids[0]] for _ in base_model_ids]
                node_mapping[model_id] = node_mapping[base_model_ids[0]]

        # now node_mapping's keys contains all models in the old model system that are fused into models in ``fused_model_list``
        new_model_dict = {model_id:model for model_id, model in self.model_dict.items() if model_id not in node_mapping}
        new_model_dict.update({model.model_id:model for model in fused_model_list})
        
        for model_id in self.model_dict:
            if model_id not in node_mapping:
                node_mapping[model_id] = model_id
        new_out_edge_dict = defaultdict(set)
        for src, tgts in self.out_edge_dict.items():
            new_out_edge_dict[node_mapping[src]].update([node_mapping[tgt] for tgt in tgts])
        for k in new_out_edge_dict:
            # we need to delete the self circle introduced in by fusing models
            new_out_edge_dict[k] = list(new_out_edge_dict[k].difference({k}))

        
        new_model_sys = MyModelSystem(new_model_dict.values(), new_out_edge_dict, self.cost_table, self.inp_merger, self.outlen_generator,
                                      need_correct_inp_out_lens=False)
        return new_model_sys




    def _get_inp_out_lens_considering_LLM_dependency(
            self, 
            cost_table: CostTable,
            inp_merger, outlen_generator, 
            out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]],
        ):
        """
            Compute the input and output sequence lengths for all LLMs in the system according to the given ``inp_merger``.
            NOTE: the model dependency is considered here.
            NOTE: this function is called when initializing an LLM system.
            INPUT:
                inp_merger: a function whose input is 
                    (1) the output seq lengths from the input models of an LLM (if any),
                    (2) the original inp seq lengths of the LLM (if any)
                    and generates the length of the fused input based on the given inplens.
                outlen_generator: a function which gereates an outlen given an inplen.

            Update:
                update the input and output seq lengths of all LLMs in the system.
        """

        def get_required_outputs_from_inp_model(inp_seq_ids, inp_model_id, out_req_id_mapping):
            outputs = self.model_dict[inp_model_id].get_inp_out_seqlens()[1]
            outputs_inds = self.model_dict[inp_model_id].get_inp_seq_ids()

            if inp_model_id in out_req_id_mapping:
                # we need to concat some output of this model's outputs as new outputs that other models can use
                new_output_ids = [out_req_id_mapping[inp_model_id][output_id][0] for output_id in outputs_inds]
                print(f"new_output_ids: {new_output_ids}")
                # sort the items by the order of new output req ids
                order = np.argsort(new_output_ids)
                new_output_ids, counts = np.unique(new_output_ids, return_counts=True)
                print(f"new_output_ids: {new_output_ids}, counts: {counts}")
                cum_chunk_nums = np.cumsum(np.concatenate(([0], counts)))
                new_outputs = np.asarray(outputs)[order]
                print(f"cum_chunk_nums: {cum_chunk_nums}")
                print(f"new_outputs: {new_outputs}")
                new_outputs = [sum(new_outputs[cum_chunk_nums[i]:cum_chunk_nums[i+1]]) for i in range(len(counts))]
                print(f"new_outputs: {new_outputs}")
                
                outputs = new_outputs
                outputs_inds = new_output_ids

                # NOTE: TODO: 此处我们暂时修改一下output inds，等horizontal fusion的支持完善了之后可以把这个去掉。
                outputs_inds = cum_chunk_nums[1:]-1
                print(f"outputs_inds: {outputs_inds}")

            # we need to consider the case where the inp seq id is not the output of the model
            # for example: a model after the chain summary, it depends on different model stages in the summary chain
            #           the total inp seq ids will be the full set of inp seq ids, but they are outputs of different model stages

            return get_infor_given_seq_ids(
                values=outputs, 
                seq_ids_we_have=outputs_inds, 
                seq_ids_requested=inp_seq_ids, 
                default_value=0)
            
            inds = np.searchsorted(outputs_inds, inp_seq_ids)
            
            # we need to consider the case where the inp seq id is not the output of the model
            # for example: a model after the chain summary, it depends on different model stages in the summary chain
            #           the total inp seq ids will be the full set of inp seq ids, but they are outputs of different model stages
            ret = np.zeros(len(inp_seq_ids),dtype=inp_seq_ids.dtype)
            valid_indices1 = inds<len(outputs_inds)
            valid_indices2 = (outputs_inds[inds[valid_indices1]] == inp_seq_ids[valid_indices1])
            inds = inds[valid_indices1][valid_indices2]
            ret_inds = np.arange(len(ret))[valid_indices1][valid_indices2]
            ret[ret_inds] = np.asarray(outputs)[inds]
            
            return ret
            return np.asarray(outputs)[inds]

        # sort the models by the topological order first
        self.get_all_level_models()
        for model_ids in self.all_level_model_ids:
            for model_id in model_ids:
                model = self.model_dict[model_id]
                if len(model.input_model_ids) == 0:
                    # we do not need to change the inp seq lengths of this model
                    continue
                ori_inp_lens = model.get_inp_out_seqlens()[0]
                # NOTE: all the inp seq ids are sorted
                inp_seq_ids = model.get_inp_seq_ids()
                new_inp_lens = inp_merger(
                    [ori_inp_lens] + \
                        [get_required_outputs_from_inp_model(inp_seq_ids, inp_model_id, out_req_id_mapping) for inp_model_id in model.input_model_ids]
                        )
                # new_inp_lens = inp_merger(
                #     [ori_inp_lens] + \
                #         [self.model_dict[inp_model_id].get_inp_out_seqlens()[1] for inp_model_id in model.input_model_ids]
                #         )
                new_out_lens = outlen_generator(model.model_name, new_inp_lens)
                model.update_inp_out_seqlens(new_inp_lens, new_out_lens, model.get_inp_seq_ids(), cost_table)

                






    
    def get_runnable_models(self, running_model_ids: List[int]):
        """
            A model can be started when all its input models are started.
            Return the models which can be started given the list of running models and the finished models.
            NOTE: the returned models may not depend directly on the running models.
        """
        def is_finished_or_running(model: MyModelInfor, running_model_ids: List[int]):
            return (model.is_finished()) or \
                (model.model_id in running_model_ids)

        to_run: List[MyModelInfor] = list()
        for model in self.model_dict.values():
            if is_finished_or_running(model, running_model_ids):
                continue

            # inps = self.inp_edge_dict[model.model_id]
            inps = model.input_model_ids
            inps_status = [is_finished_or_running(self.model_dict[inp], running_model_ids) for inp in inps]
            if False in inps_status:
                continue

            to_run.append(model)
        
        return to_run
    

    # def get_next_level_models(self, last_level_running_model_ids: List[int], all_running_model_ids: List[int]):
    #     """
    #         A model can be started when all its input models are started.
    #         Return the models which can be started given the list of running models and the finished models.
    #         NOTE: we should ensure the next level models depend directly on the newly added running models.
    #     """
    #     to_run: List[MyModelInfor] = list()
    #     for model in self.model_list:
    #         if (model.is_finished()) or (model.model_id in all_running_model_ids):
    #             continue

    #         if (len(last_level_running_model_ids) > 0) and (len(model.input_model_ids) == 0):
    #             # this model should appear in the first level of running models
    #             continue

    #         # inps = self.inp_edge_dict[model.model_id]
    #         inps_status = [(inp in last_level_running_model_ids) for inp in model.input_model_ids]
    #         if False in inps_status:
    #             continue

    #         to_run.append(model)
        
    #     return to_run
    
    def get_models_at_given_level(self, level_num: int):
        if level_num >= len(self.all_level_model_ids):
            return list()
        return [self.model_dict[model_id] for model_id in self.all_level_model_ids[level_num]]



    def get_runnable_plans_from_cand_plans(
            self, 
            running_plan_group: List[MyExecPlan], 
            cand_models: List[MyModelInfor], cand_exec_plans: List[List[MyExecPlan]]):
        

        print(f"in get_runnable_plans_from_cand_plans: running_plan_group {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in running_plan_group]} cand_models: {[_.get_base_model_ids() for _ in cand_models]}")



        def is_finished_or_running(model: MyModelInfor, running_model_ids: List[int]):
            return (model.is_finished()) or \
                (set(model.get_base_model_ids()).issubset(running_model_ids))
                # (model.get_base_model_ids() in running_model_ids)

        runnable_exec_plans: List[List[MyExecPlan]] = list()
        running_model_ids: List[int] = [exec_plan.model.model_id for exec_plan in running_plan_group]
        # we need to support fused models
        for exec_plan in running_plan_group:
            running_model_ids.extend(exec_plan.model.get_base_model_ids())
            print(f"running model ids: {exec_plan.model.model_id} = {exec_plan.get_base_model_ids()}")
        
        for model, exec_plans in zip(cand_models, cand_exec_plans):
            inps = model.input_model_ids
            inps_status = [is_finished_or_running(self.model_dict[inp], running_model_ids) for inp in inps]

            print(f"checking model: {model.get_base_model_ids()}  inp model ids: {inps} = {[self.model_dict[_].get_base_model_ids() for _ in inps]}, inps_status: {inps_status}")

            if False in inps_status:
                continue

            # print(f"in get_runnable_plans_from_cand_plans: model {model.model_id} is runnable")

            runnable_exec_plans.append(exec_plans)
        return runnable_exec_plans



    def get_all_level_models(self):
        """
            Update the model ids on all levels according to the current model status.
            Update: self.all_level_model_ids
        """
        self.all_level_model_ids = list()
        # visited: List[bool] = np.asarray([False] * len(self.model_dict))
        visited: Dict[int, bool] = {model_id: False for model_id in self.model_dict}

        # while sum(visited) < len(self.model_dict):
        while False in visited.values():
            # get the model ids on this level
            # newly_visited: List[bool] = np.asarray([False] * len(self.model_dict))
            newly_visited: Dict[int, bool] = {model_id: False for model_id in self.model_dict}
            new_model_ids: List[int] = list()
            for model in self.model_dict.values():
                if visited[model.model_id]:
                    continue
                if model.is_finished():
                    newly_visited[model.model_id] = True
                    continue
                inp_status = [visited[inp] for inp in model.input_model_ids]
                if False in inp_status:
                    continue
                newly_visited[model.model_id] = True
                new_model_ids.append(model.model_id)
            # visited = visited + newly_visited
            visited = {k: visited[k] or newly_visited[k] for k in visited}
            if len(new_model_ids) > 0:
                self.all_level_model_ids.append(new_model_ids)





    def get_model_num(self)->int:
        return len(self.model_dict)




    def get_not_finished_base_model_num(self) -> int:
        not_finished_model_num = sum([not base_model.is_finished() \
                                      for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()])
        return not_finished_model_num


    def is_finished(self) -> bool:
        """
            Return True if all the models in the system is finished.
        """
        # return False not in [model.is_finished() for model in self.model_dict.values()]
        return self.get_not_finished_base_model_num() == 0



    # def get_model_states(self):
    #     '''
    #         Get the current inference progress of the given list of models.
    #         NOTE: the returned progress should be able to be added to a set.
    #     '''
    #     return tuple(sorted([model.get_state() for model in self.model_dict.values()]))

    def get_base_model_states(self):
        '''
            Get the current inference progress of the given list of models.
            NOTE: the returned progress should be able to be added to a set.
        '''
        return tuple(sorted([base_model.get_state() for model in self.model_dict.values() \
                             for base_model in model.get_base_models()]))
    
    
    def get_model_inp_out_lens(self):
        ori_inp_out_lens_list = [model.get_inp_out_seqlens() for model in self.model_dict.values()]
        return ori_inp_out_lens_list
    
    def get_model_remaining_decode_flops(self):
        ori_remaining_decode_flops_list = [model.get_remaining_flops() for model in self.model_dict.values()]
        return ori_remaining_decode_flops_list

    def get_model_inp_seq_ids(self):
        ori_inp_seq_ids_list = [model.get_inp_seq_ids() for model in self.model_dict.values()]
        return ori_inp_seq_ids_list


    def get_model_inp_model_ids(self):
        ori_inp_model_ids_list = [model.input_model_ids for model in self.model_dict.values()]
        return ori_inp_model_ids_list


    
    
    def recover_model_state(
            self,
            inp_seq_ids_list: List[List[int]],
            inp_out_lens_list: List[Tuple[List[int], List[int]]], 
            cost_table: CostTable, remaining_decode_flops_list: List[float],
            inp_model_ids_list: List[List[int]]):
        # for model, inp_out_lens in zip(model_list, inp_out_lens_list):
        # for i in range(len(self.model_list)):
        #     model = self.model_list[i]
        #     inp_out_lens = inp_out_lens_list[i]
        #     remaining_decode_flops = remaining_decode_flops_list[i]
        #     model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)
        for model, inp_seq_ids, inp_out_lens, remaining_decode_flops, inp_model_ids in \
            zip(self.model_dict.values(),inp_seq_ids_list,inp_out_lens_list,remaining_decode_flops_list,inp_model_ids_list):
            model.update_inp_out_seqlens(*inp_out_lens, inp_seq_ids, cost_table, remaining_decode_flops)
            model.input_model_ids = inp_model_ids


    def print_model_list(self):
        print(f"model_list: {[(str(model), model.model_id, model.get_base_model_ids()) for model in self.model_dict.values()]}")




    def get_candidate_plan_groups(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
        """
        # running_model_ids are models running in this exec stage
        tot_plan_groups = [[]]
        new_plan_groups = [[]]

        uniq_exec_plan_mapping = dict()
        # stores the best plan group given a set of models and the GPU num to use --> to add pruning in the plan group generation process
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()

        # we first divide the unfinished model into different levels
        self.get_all_level_models()

        print(f"all_level_model_ids: {self.all_level_model_ids}")

        visit_model_level = -1
        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)

            # print(f"cand_models in this round: {[model.model_id for model in cand_models]}")

            # 1. first get the candidate exec plans for each model
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input)
                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")

            # print(f"New Round get_candidate_plan_groups ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            for cand_plan_group in new_plan_groups:
                # running_model_ids = [exec_plan.model.model_id for exec_plan in cand_plan_group]
                # cand_models = self.get_next_level_models(last_level_running_model_ids=, all_running_model_ids=running_model_ids)

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)

                # print(f"runnable_exec_plans_list:  ----------------------")
                # for plans in runnable_exec_plans_list:
                #     print(f"{len(plans)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plans]}")
                
                # print(f"root cand_plan_group: ------------------------")
                # print(f"{len(cand_plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in cand_plan_group]}")
                
                # 2. second combine exec plans for different models into a group
                plan_groups = [cand_plan_group]
                _append_exec_plan(plan_groups, runnable_exec_plans_list, 0, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping)
                # plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans) \
                #             for plan_group in plan_groups if len(plan_group) > 0]
                
                if len(plan_groups) == 1:
                    # no new model is added to cand_plan_group
                    tot_plan_groups.extend(plan_groups)
                else:
                    # we first update the good_plan_group_dict
                    # print(f"the groups we found a round: ")
                    # for plan_group in plan_groups[1:]:
                    #     print([(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group])
                    good_plan_group_keys = [_update_good_plan_group_dict(
                        cost_table=cost_table, check_gap=check_gap, sort_input=sort_input, last_stage_exec_plans=last_stage_exec_plans,
                        plan_group=plan_group, good_plan_group_dict=good_plan_group_dict
                        ) for plan_group in plan_groups[1:]]
                    good_plan_group_keys = [_ for _ in good_plan_group_keys if _ != None]
                    to_compare = sorted([_[0] for _ in good_plan_group_dict.values()], reverse=True)[top_k-1] if top_k <= len(good_plan_group_dict) else -1
                    good_plan_groups = [good_plan_group_dict[_][1].exec_plans for _ in good_plan_group_keys if good_plan_group_dict[_][0] > to_compare]
                    # print(f"only keep good plan groups: num: {len(good_plan_groups)}, keys: {[_ for _ in good_plan_group_keys if good_plan_group_dict[_][0] > to_compare]}")
                    # print(f"to_compare: {to_compare}, top_k: {top_k}")
                    # print(f"only keep good plan groups: num: {len(good_plan_groups)}, keys: {[[(plan.model.get_base_model_ids(), plan.get_key()) for plan in _] for _ in good_plan_groups]}")
                    # 
                    # good_plan_groups = [_ for _ in good_plan_groups if len(_)!=0]
                    tmp_new_plan_groups.extend(good_plan_groups)
                    # 
                    # tmp_new_plan_groups.extend(plan_groups[1:])

                # print(f"tot_plan_groups: {[[str(plan) for plan in plan_group] for plan_group in tot_plan_groups]}")
                # print(f"tmp_new_plan_groups: {[[str(plan) for plan in plan_group] for plan_group in tmp_new_plan_groups]}")
                # print(f"cand_plan_group: {[str(plan) for plan in cand_plan_group]}")
                # print(f"tot_plan_groups: ----------------------")
                # for plan_group in tot_plan_groups:
                #     print(f"{len(plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                # print(f"tmp_new_plan_groups: ------------------")
                # for plan_group in tmp_new_plan_groups:
                #     print(f"{len(plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


        # print(f"in get_candidate_plan_groups: the plan groups we generated: ")
        # for plan_group in tot_plan_groups:
        #     print(f"{len(plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")



        # convert plan_groups to MyExecPlanGroup objects
        # plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
        #                 check_gap=check_gap, sort_input=sort_input,) \
        #             for plan_group in tot_plan_groups if len(plan_group) > 0]

        plan_groups = [_[1] for _ in good_plan_group_dict.values()]

        return plan_groups




    def get_candidate_plan_groups_greedy_baseline_adapted_from_MuxServe_best_model_first(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3))->List[MyExecPlanGroup]:
        """
            Greedily select exec plans to run in a exec stage.
            NOTE: sort models by their sizes first, and then select their best exec plans.
            NOTE: when there are multi-level models in the system, we sort the models level by level.
        """
        new_plan_groups = [[]]
        checked_model_ids = list()
        while True:
            cand_plan_group = new_plan_groups[0]
            running_model_ids = [exec_plan.model.model_id for exec_plan in cand_plan_group]
            cand_models = self.get_runnable_models(running_model_ids=running_model_ids)
            cand_models = [model for model in cand_models if model.model_id not in checked_model_ids]

            # sort cand_models           
            cand_models: List[MyModelInfor] = [get_sorted_models_by_model_size(cand_models)[0]]

            # 1. first get the candidate exec plans for each model
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input)
                exec_plans_list.append(exec_plans)
            
            # 2. second combine exec plans for different models into a group
            plan_groups = [cand_plan_group]
            # _append_exec_plan(plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu)
            _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
                plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu,
                cost_table, last_stage_exec_plans, 
                check_gap, sort_input,)
                           
            new_plan_groups = plan_groups
            if len(cand_models) == 0:
                break

            checked_model_ids.append(cand_models[0].model_id)
        
        # convert plan_groups to MyExecPlanGroup objects
        plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                        check_gap=check_gap, sort_input=sort_input) \
                    for plan_group in new_plan_groups if len(plan_group) > 0]
        return plan_groups




    def get_candidate_plan_groups_greedy_baseline_adapted_from_MuxServe_best_exec_plan_first(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3))->List[MyExecPlanGroup]:
        """
            Greedily select exec plans to run in a exec stage.
            NOTE: sort exec plans from all candidate models by their throughputs.
        """
        new_plan_groups = [[]]
        while True:
            cand_plan_group = new_plan_groups[0]
            ori_group_size = len(cand_plan_group)
            running_model_ids = [exec_plan.model.model_id for exec_plan in cand_plan_group]
            cand_models = self.get_runnable_models(running_model_ids=running_model_ids)

            # 1. first get the candidate exec plans for each model
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input)
                exec_plans_list.extend(exec_plans)
            
            # 2. second combine exec plans for different models into a group
            plan_groups = [cand_plan_group]
            # _append_exec_plan(plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu)
            _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
                plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu,
                cost_table, last_stage_exec_plans, 
                check_gap, sort_input,)
                           
            new_plan_groups = plan_groups
            if len(plan_groups[0]) == ori_group_size:
                break
        
        # convert plan_groups to MyExecPlanGroup objects
        plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                        check_gap=check_gap, sort_input=sort_input) \
                    for plan_group in new_plan_groups if len(plan_group) > 0]
        return plan_groups







    def remaining_models_are_on_the_last_layer(self)->bool:
        remaining_model_ids = [model_id for model_id, model in self.model_dict.items() if not model.is_finished()]
        output_model_num = sum([len(self.out_edge_dict[model_id]) for model_id in remaining_model_ids])
        fused_model_num = sum([isinstance(self.model_dict[model_id], MyFusedModelInfor) for model_id in remaining_model_ids])
        return (output_model_num == 0) and (fused_model_num == 0)






def get_infor_given_seq_ids(
        values, seq_ids_we_have: List[int], seq_ids_requested: List[int], default_value):
    """
        1. ``values`` containing the values of corresponding to the ``seq_ids_we_have``;
        2. default_value: the value assigned to the requested seq ids which are not in seq_ids_we_have;
    """
    # we need to consider the case where the inp seq id is not the output of the model
    # for example: a model after the chain summary, it depends on different model stages in the summary chain
    #           the total inp seq ids will be the full set of inp seq ids, but they are outputs of different model stages
    values = np.asarray(values)
    seq_ids_we_have = np.asarray(seq_ids_we_have)
    seq_ids_requested = np.asarray(seq_ids_requested)
    # 
    ret = np.full(len(seq_ids_requested), default_value, dtype=values.dtype)
    inds = np.searchsorted(seq_ids_we_have, seq_ids_requested)
    valid_indices1 = inds<len(seq_ids_we_have)
    valid_indices2 = (seq_ids_we_have[inds[valid_indices1]] == seq_ids_requested[valid_indices1])
    inds = inds[valid_indices1][valid_indices2]
    ret_inds = np.arange(len(ret))[valid_indices1][valid_indices2]
    ret[ret_inds] = np.asarray(values)[inds]
    # 
    return ret




def get_factors(v, start_from, smaller_than):
    return [i for i in range(start_from, smaller_than) if v%i==0]



def is_valid_exec_plan(exec_plan: MyExecPlan, cost_table: CostTable):
    '''
    Check whether this exec plan itself is valid, 
    without considering the exec plan combination to be applied together on the GPU cluster.
    Input:
        exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    wld_degree = exec_plan.wld_degree
    mem_per_comp_gpu = exec_plan.mem_per_comp_gpu
    param_byte_per_comp_gpu = exec_plan.param_byte_per_comp_gpu
    param_byte_per_cache_gpu = exec_plan.param_byte_per_cache_gpu
    byte_per_gpu = exec_plan.tot_gpu_mem_byte
    # 

    # print(f"param_byte_per_comp_gpu:{param_byte_per_comp_gpu}, param_byte_per_cache_gpu:{param_byte_per_cache_gpu}, byte_per_gpu:{byte_per_gpu}")

    # 0. is mem_per_comp_gpu and wld_degree consistent with each other
    if (wld_degree > 0) and (mem_per_comp_gpu < 0.9):
        # print(f"mem_per_comp_gpu and wld_degree inconsistent")
        return False
    # 1. whether mem is enough
    # check comp gpu mem
    if mem_per_comp_gpu * byte_per_gpu < param_byte_per_comp_gpu:
        # print(f"not enough comp gpu mem")
        return False
    # check cache gpu mem
    if byte_per_gpu < param_byte_per_cache_gpu:
        # print(f"not enough cache gpu mem")
        return False
    # 2. whether weight loading bandwidth is enough
    # compare the peak comp throughput with the weight loading speed, leave it to the cost model? I think it is ok.
    
    # 3. whether we have the corresponding cost table for this model and exec plan
    # data parallel
    if not cost_table.can_estimate_cost(exec_plan.model.model_path, exec_plan.get_key_single_dp_worker()):
        # print(f"cannot estimate cost")
        return False
    
    return True




def _get_possible_exec_plans(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    import math
    exec_plans = list()
    
    # 1. do not need to generate exec plans for finished models
    # if model.left_flops_per_token == 0:
    if model.is_finished():
        print(f"model: {model}, model is finished")
        return exec_plans
    
    
    # enumerate possible tensor parallel degrees (i.e., parallel worker numbers)
    for i in range(int(math.log(tot_gpu_num, 2)+1)):
        num_worker = 2**i

        # 1. first get necessary engine args
        if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
            _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
                model_path=model.model_path, tensor_parallel_size=num_worker)
        (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]
        
        infer_args = InferenceArgs(scheduler_config, cache_config)
        
        # 2. get cache byte per block and param_byte_per_layer
        # gpu_cache_byte_per_block = (hidden_dimension * infer_args.block_size) // num_worker
        gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
        # param_byte_per_layer = get_param_byte_per_layer(model, num_worker)
        param_byte_per_layer, extra_byte = \
            get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)

        # for wld_degree in [2, 8, 10, 16, 20, 40]: # get_factors(layer_num): # TODO
        # for wld_degree in get_factors(model.layer_num, 2, model.layer_num):
        for wld_degree in [2]: # do not consider cache gpu temporarily
            if wld_degree < 2:
                # if wld_degree <= 2, then we do not need to use cache gpu, so <2 will be the same as ==2.
                continue
            
            # param_byte_per_comp_gpu = get_extra_param_byte(model, num_worker) + \
            #     param_byte_per_layer * (model.layer_num - wld_degree + 2)
            param_byte_per_comp_gpu = extra_byte + \
                param_byte_per_layer * (model.layer_num - wld_degree + 2)

            # TODO (jingzhi): we allow different possoble cache_gpu_num? strange
            for cache_gpu_num in range(tot_gpu_num-num_worker+1):
                if (wld_degree > 2) and (cache_gpu_num == 0):
                    # no cache gpu but have layers cached on other gpus, inconsistent
                    continue
                if (wld_degree == 2) and (cache_gpu_num > 0):
                    # no layer cached but has cache gpus
                    continue
                
                # we can compute param_byte_per_cache_gpu
                param_byte_per_cache_gpu = 0
                if cache_gpu_num > 0:
                    # param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / num_worker / cache_gpu_num
                    # we do not multiple num_worker here because two tp workers may not have the same cache gpus
                    # we do not multiple dp_size here because two dp workers may not have the same cache gpus
                    param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / cache_gpu_num
                
                # for mem_per_comp_gpu in [0.9]: # TODO [j/10 for j in range(1, 10)]:
                for mem_per_comp_gpu in [j/10 for j in range(1, 10)]:

                    dp_size = 1
                    exec_plan = MyExecPlan(model,
                        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                        param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                        gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                    
                    if isinstance(model, MyFusedModelInfor):
                        exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                    
                    # print(f"gen an exec plan: {str(exec_plan)}")

                    # check whether exec_plan is valid
                    if is_valid_exec_plan(exec_plan, cost_table):
                        exec_plans.append(exec_plan)

                        # print(f"valid")

                        # support data parallel
                        for dp_size in range(2, tot_gpu_num // num_worker + 1):
                            if dp_size * num_worker + cache_gpu_num > tot_gpu_num:
                                # each dp worker occupies num_worker GPUs for computation
                                # all dp workers can share cache_gpu_num GPUs for cache (but it is not necessary)
                                continue

                            # add exec_plan with dp_size
                            exec_plan = MyExecPlan(model,
                                num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                                param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                                gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                            
                            if isinstance(model, MyFusedModelInfor):
                                exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                            
                            # do not need to call is_valid_exec_plan for dp_size
                            exec_plans.append(exec_plan)

    return exec_plans






# NOTE: we set dp_size to 1 here, but maybe we should select the best exec plan for the model?
# ==> such function is implemented in "_get_possible_exec_plans_naive_baseline_2()"
# Change the default version to the version where dp_size can be any value.
# 但是其实两种思路都有测试的价值，说到底关键还是我们的cost model发挥了作用，没有cost model的话或许就真的只能选择dp_size=1的这种可能
def _get_possible_exec_plans_naive_baseline_1(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable, sort_input: bool):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    NOTE: generate the exec plan which uses all gpu for computation.
    '''
    exec_plans = list()
    
    # 1. do not need to generate exec plans for finished models
    # if model.left_flops_per_token == 0:
    if model.is_finished():
        print(f"This model is finished.")
        return exec_plans
    

    num_worker = tot_gpu_num

    # 1. first get necessary engine args
    if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
        _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
            model_path=model.model_path, tensor_parallel_size=num_worker)
    (model_config, cache_config, parallel_config, scheduler_config,
        device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]
    
    infer_args = InferenceArgs(scheduler_config, cache_config)
    
    # 2. get cache byte per block and param_byte_per_layer
    # gpu_cache_byte_per_block = (hidden_dimension * infer_args.block_size) // num_worker
    gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
    # param_byte_per_layer = get_param_byte_per_layer(model, num_worker)
    param_byte_per_layer, extra_byte = \
        get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)

    wld_degree = 2

    param_byte_per_comp_gpu = extra_byte + \
        param_byte_per_layer * (model.layer_num - wld_degree + 2)


    cache_gpu_num = 0
    mem_per_comp_gpu = 0.9
    param_byte_per_cache_gpu = 0
    dp_size = 1

    exec_plan = MyExecPlan(model,
        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
        param_byte_per_comp_gpu, param_byte_per_cache_gpu,
        gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)

    if isinstance(model, MyFusedModelInfor):
        exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
    
    print(f"gen an exec plan: {str(exec_plan)}")

    # check whether exec_plan is valid
    if is_valid_exec_plan(exec_plan, cost_table):
        exec_plans.append(exec_plan)


    return exec_plans





def _get_possible_exec_plans_naive_baseline(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable, sort_input: bool):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    NOTE: only generate the exec plan which has the highest throughput for the model
    '''
    import math
    exec_plans: List[MyExecPlan] = list()
    
    # 1. do not need to generate exec plans for finished models
    # if model.left_flops_per_token == 0:
    if model.is_finished():
        return exec_plans
    
    
    # enumerate possible tensor parallel degrees (i.e., parallel worker numbers)
    for i in range(int(math.log(tot_gpu_num, 2)+1)):
        num_worker = 2**i

        # 1. first get necessary engine args
        if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
            _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
                model_path=model.model_path, tensor_parallel_size=num_worker)
        (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]
        
        infer_args = InferenceArgs(scheduler_config, cache_config)
        
        # 2. get cache byte per block and param_byte_per_layer
        # gpu_cache_byte_per_block = (hidden_dimension * infer_args.block_size) // num_worker
        gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
        # param_byte_per_layer = get_param_byte_per_layer(model, num_worker)
        param_byte_per_layer, extra_byte = \
            get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)

        # for wld_degree in [2, 8, 10, 16, 20, 40]: # get_factors(layer_num): # TODO
        # for wld_degree in get_factors(model.layer_num, 2, model.layer_num):
        for wld_degree in [2]:
            if wld_degree < 2:
                # if wld_degree <= 2, then we do not need to use cache gpu, so <2 will be the same as ==2.
                continue
            
            # param_byte_per_comp_gpu = get_extra_param_byte(model, num_worker) + \
            #     param_byte_per_layer * (model.layer_num - wld_degree + 2)
            param_byte_per_comp_gpu = extra_byte + \
                param_byte_per_layer * (model.layer_num - wld_degree + 2)

            # TODO (jingzhi): we allow different possoble cache_gpu_num? strange
            for cache_gpu_num in range(tot_gpu_num-num_worker+1):
                if (wld_degree > 2) and (cache_gpu_num == 0):
                    # no cache gpu but have layers cached on other gpus, inconsistent
                    continue
                if (wld_degree == 2) and (cache_gpu_num > 0):
                    # no layer cached but has cache gpus
                    continue
                
                # we can compute param_byte_per_cache_gpu
                param_byte_per_cache_gpu = 0
                if cache_gpu_num > 0:
                    # param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / num_worker / cache_gpu_num
                    # we do not multiple num_worker here because two tp workers may not have the same cache gpus
                    # we do not multiple dp_size here because two dp workers may not have the same cache gpus
                    param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / cache_gpu_num
                
                # for mem_per_comp_gpu in [0.9]: # TODO [j/10 for j in range(1, 10)]:
                for mem_per_comp_gpu in [j/10 for j in range(1, 10)]:

                    dp_size = 1
                    exec_plan = MyExecPlan(model,
                        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                        param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                        gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                    
                    if isinstance(model, MyFusedModelInfor):
                        exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                    
                    # print(f"gen an exec plan: {str(exec_plan)}")

                    # check whether exec_plan is valid
                    if is_valid_exec_plan(exec_plan, cost_table):
                        exec_plans.append(exec_plan)

                        # print(f"valid")

                        # support data parallel
                        for dp_size in range(2, tot_gpu_num // num_worker + 1):
                            if dp_size * num_worker + cache_gpu_num > tot_gpu_num:
                                # each dp worker occupies num_worker GPUs for computation
                                # all dp workers can share cache_gpu_num GPUs for cache (but it is not necessary)
                                continue

                            # add exec_plan with dp_size
                            exec_plan = MyExecPlan(model,
                                num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                                param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                                gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                            
                            if isinstance(model, MyFusedModelInfor):
                                exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                            
                            # do not need to call is_valid_exec_plan for dp_size
                            exec_plans.append(exec_plan)
    
    # compute the latency for each exec plan
    # consider the max data parallel worker latency
    exec_plans = sorted(exec_plans, \
        key=lambda exec_plan: \
            exec_plan.get_max_dp_latency(cost_table, sort_input)+\
                cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
    )
    exec_plans = [exec_plans[0]]

    return exec_plans






def get_possible_exec_plans(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable,
        baseline: str, sort_input: bool):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    if baseline == 'ours':
        return _get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table)
    else:
        # return _get_possible_exec_plans_naive_baseline_2(model, tot_gpu_num, byte_per_gpu, cost_table, sort_input)
        return _get_possible_exec_plans_naive_baseline(model, tot_gpu_num, byte_per_gpu, cost_table, sort_input)




# def get_failed_gpu_request_pair_num(resources: np.ndarray, requests):
#     return sum([sum(resources < r_group[0]) * len(r_group) for r_group in requests])





def select_best_gpus_for_cache_request(request: np.ndarray, resources: np.ndarray, requests: List[List[int]]):
    '''
    Input: 
        request: list of ints: each int is the cache mem required on a cache gpu.
        resources: the available gpu mems on each candidate cache gpu.
        requests: the remaining cache mem requests.
    NOTE: we deal with all cache requirement for a model at a time.
    现在的这种计算fragment ratio的方式是按照ATC23里面来的，感觉有点奇怪，还可以有别的计算fragment ratio的方式，有需要的话可以尝试。
    '''
    best_choice = None
    best_fail_ratio = float('inf')
    tmp_resources = resources.copy()
    choices = itertools.combinations(range(len(resources)), len(request))
    for choice in choices:
        # check the choice has enough mem
        choice_list = list(choice)
        if (resources[choice_list] < request).any():
            continue
        # compute the fragment ratio of this choice
        tmp_resources[choice_list] = tmp_resources[choice_list] - request
        # TODO (jingzhi) the definition of fail ratio here is also strange.
        # TODO (jingzhi) cannot select the best choice for the last request in all requests, i.e., when requests=[]
        fail_ratio = sum([sum((tmp_resources < r[0])*tmp_resources) for r in requests]) / sum(tmp_resources)
        if fail_ratio < best_fail_ratio:
            best_fail_ratio = fail_ratio
            best_choice = choice_list
        tmp_resources[choice_list] = tmp_resources[choice_list] + request
    return best_choice, best_fail_ratio





def get_tot_worker_num(exec_plans: List[MyExecPlan]):
    # return sum([exec_plan.num_worker for exec_plan in exec_plans])
    # data parallel
    return sum([exec_plan.num_worker * exec_plan.dp_size for exec_plan in exec_plans])


def is_valid_exec_plan_combination(exec_plans: List[MyExecPlan], tot_gpu_num, byte_per_gpu):
    '''
    Check whether the exec plan combination can be applied to the GPU cluster.
    Input: 
        exec_plans: a list of exec plans: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
        tot_gpu_num: the total GPU number.
        byte_per_gpu: the total memory of a GPU (in bytes).
    '''
    '''
    要怎么判断某种execution 组合在资源上是可行的？有没有什么快速判断方法？不能简单求和因为需要的资源在不同的卡上，也不能
    简单分多个维度，因为各个维度之间是等价的，放在哪个维度上都行。好像也是一个hard problem。
    就参考ATC23里的方法（仅是参考idea，因为我们没有确认我们的方法和他们的方法是一样的），
    根据选择后的无法满足某一内存需求的GPU数量总和的大小，来判断应该把Request对应到哪个GPU上。
    '''

    # print(f"checking plan group 2: {[str(_) for _ in exec_plans]}")

    use_cache_gpu_plans = []
    without_cache_gpu_plans = []
    cache_gpu_remaining_bytes = []
    cache_gpu_required_bytes = []
    for exec_plan in exec_plans:
        num_worker = exec_plan.num_worker
        cache_gpu_num = exec_plan.cache_gpu_num
        mem_per_comp_gpu = exec_plan.mem_per_comp_gpu
        param_byte_per_cache_gpu = exec_plan.param_byte_per_cache_gpu
        dp_size = exec_plan.dp_size
        if cache_gpu_num > 0:
            use_cache_gpu_plans.append(exec_plan)
            # data parallel
            # cache_gpu_required_bytes.append(np.asarray([param_byte_per_cache_gpu]*cache_gpu_num))
            # for each tp-dp worker, it has its own cache request list
            cache_gpu_required_bytes.extend(
                [np.asarray([param_byte_per_cache_gpu]*cache_gpu_num) for _ in range(num_worker*dp_size)]
                )
        else:
            without_cache_gpu_plans.append(exec_plan)
            # TODO (jingzhi): consider to set the max available gpu mem ratio to be 0.9 instead of 1 <- I think we should keep this to 1?
            # cache_gpu_remaining_bytes.extend([byte_per_gpu * (1 - mem_per_comp_gpu)] * num_worker)
            # data parallel
            cache_gpu_remaining_bytes.extend([byte_per_gpu * (1 - mem_per_comp_gpu)] * num_worker * dp_size)
    
    # 1. check comp resources
    # print([str(exec_plan) for exec_plan in exec_plans])
    tot_worker_num = get_tot_worker_num(exec_plans)
    if tot_worker_num > tot_gpu_num:
        # print(f"tot_worker_num > tot_gpu_num: {tot_worker_num, tot_gpu_num}")
        return False
    
    
    # 2. check cache gpu mem resources
    # sort the cache request by size
    # requests = sorted(cache_gpu_required_bytes, key=lambda i: i[0], reverse=True)
    # resources = np.asarray(cache_gpu_remaining_bytes)
    # for i, r_group in enumerate(requests):
    #     for j, r in enumerate(r_group):
    #         remaining_requests = []
    #         if j < len(r_group) - 1:
    #             remaining_requests = [r_group[j+1:]]
    #         if i < len(requests) - 1:
    #             remaining_requests.extend(requests[i+1:])
    #         # get the best gpu for r
    #         resource_i, _ = select_best_gpu_for_cache_request(r, resources, remaining_requests)
    #         if resource_i == None:
    #             # this exec plan combination is not valid
    #             # print(f"cache gpu mem is not enough: resources: {resources}, requests: {requests}")
    #             return False
    #         # update the resources
    #         resources[resource_i] = resources[resource_i] - r
    
    # 2. check cache gpu mem resources
    # sort the cache gpu mem request by cache_mem-per-gpu: from high to low
    requests = sorted(cache_gpu_required_bytes, key=lambda i: i[0], reverse=True)
    resources = np.asarray(cache_gpu_remaining_bytes)

    # print(f"requests: {requests}, resources: {resources}")

    for i, request in enumerate(requests):

        # update the remaining req list
        # remaining_requests = []
        # if i < len(requests) - 1:
        #     remaining_requests = requests[i+1:]
        remaining_requests = requests[i+1:]
        
        # get the best choice of cache gpus for request
        gpus_choice, _ = select_best_gpus_for_cache_request(request, resources, remaining_requests)

        # print(f"requests: {requests}, resources: {resources}, gpus_choice: {gpus_choice}")

        if gpus_choice == None:
            # this exec plan combination is not valid
            # print(f"there is no valid gpu choice for cache weight")
            return False
        # update the resources
        resources[gpus_choice] = resources[gpus_choice] - request


    # 3. delete the exec plan combinations which does not fully utilize the resources, i.e., the mem usage is less than 90%.
    if (resources > 0.1 * byte_per_gpu).any():
        # print(f"memory is not fully utilized")
        return False
    

    # print(f"valid!!!!!!")
    # print(f"checking plan group 3: {[str(_) for _ in exec_plans]}")
    return True



def _can_be_fused_vertically_linear_dependency(plan_group: List[MyExecPlan], to_fuse: MyExecPlan):
    """
        The condition to do vertical fusion:
        1. linear dependency: model 1 -> model 2.
        2. about the same model, has the same exec setting.
        3. the models in both plans have not been started.
    """

    to_fuse_inp_model_ids = to_fuse.model.input_model_ids
    for plan_i, plan in enumerate(plan_group):
        if (len(to_fuse_inp_model_ids) == 1) and (to_fuse_inp_model_ids[0] == plan.get_base_model_ids()[-1]):
            # to_fuse directly depend on the last exec_plan in the plan (which may be fused)
            if plan.model.get_name() == to_fuse.model.get_name():
                # the two plan are about the same model
                if plan.get_key() == to_fuse.get_key():
                    # the two plan has the same exec setting
                    
                    # check whether these models have been started
                    if plan.models_not_started() and to_fuse.models_not_started():
                        # we can fuse them vertically

                        print(f"in generate fused model:\n")
                        print([(_.model_id, _) for _ in plan.get_base_models()+[to_fuse.model]])

                        to_fuse_models = sorted(plan.get_base_models()+[to_fuse.model], \
                                                key=lambda model: model.model_id)
                        
                        print([(_.model_id, _) for _ in to_fuse_models])

                        fused_model = MyFusedModelInfor(to_fuse_models)
                        fused = MyVerticalFusedExecPlan(fused_model, to_fuse)
                        return [plan_group[:plan_i]+[fused]+plan_group[plan_i+1:]]
    return []






def _get_path_key(plan_group:List[MyExecPlan], exec_plan):
    """
        Get the information of all models that exec_plan depends directly or indirectly on in the current plan_group.
    """
    inp_model_ids = sorted(exec_plan.model.input_model_ids)
    key = list()
    for inp in inp_model_ids:
        for plan in plan_group:
            if (inp == plan.model.model_id) or (inp in plan.model.get_base_model_ids()):
                # key.append((inp, tuple(plan.model.get_base_model_ids()), plan.get_key() ))
                # NOTE: it seems we do not need to store the model_id itself in the key
                key.append(( tuple(plan.model.get_base_model_ids()), plan.get_key() ))
                break
    # add the key part for exec_plan
    # key.append(( exec_plan.model.model_id, tuple(exec_plan.model.get_base_model_ids()), exec_plan.get_key() ))
    key.append(( tuple(exec_plan.model.get_base_model_ids()), exec_plan.get_key() ))
    return tuple(key)



def _update_good_plan_group_dict(
        cost_table: CostTable,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        plan_group: List[MyExecPlan], good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]], 
    ) -> Optional[Tuple[List[int], int]]:
    """
        Return None if the given plan group is not good enough, else return the input plan_group key.
    """

    # 1. get MyExecPlanGroup based on plan_group
    group_obj = MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                    check_gap=check_gap, sort_input=sort_input,)
    
    # 2. get key in good_plan_group_dict
    # NOTE: here we regard fused models and base models as the same if their base model sets are the same
    # NOTE: 更保守一点的话，感觉可以把 fused model和base model 不认为包含的base model相同就是初始状态相同，这个应该也OK。
    # NOTE：两种都可以试试，感觉？先试试只要base model相同就相同的版本吧，先这么试试吧，实在不行也可以每个几轮prune一次，而不是每一轮都prune一次。
    base_model_ids = sorted(np.concatenate([exec_plan.model.get_base_model_ids() for exec_plan in group_obj.exec_plans]))
    tot_gpu_num = get_tot_worker_num(group_obj.exec_plans)
    key = ( tuple(base_model_ids),  tot_gpu_num)

    # 3. check whether this plan group is good
    ret = key
    if key not in good_plan_group_dict:
        print(f"key not in good_plan_group_dict: {key}-{group_obj.get_throughput()}")
        good_plan_group_dict[key] = (group_obj.get_throughput(), group_obj)
    else:
        if (group_obj.get_throughput() > good_plan_group_dict[key][0]) or \
            ( (group_obj.get_throughput() == good_plan_group_dict[key][0]) \
             and (len(group_obj.exec_plans) < len(good_plan_group_dict[key][1].exec_plans)) ):
            # if 1. the throughput of the new group obj is larger OR 
            # 2. (2.1) their throughputs are the same and (2.2) the new group has fewer exec plans, 
            # i.e., we prefer fusing models rather than model-level pipeline parallelism.
            good_plan_group_dict[key] = (group_obj.get_throughput(), group_obj)
            print(f"key in good_plan_group_dict, update: {key}-{group_obj.get_throughput()}")
        else:
            ret = None
            print(f"key in good_plan_group_dict, discard: ori-{key}-{good_plan_group_dict[key][0]} vs new-{group_obj.get_throughput()}")

    return ret





def _meet_vertical_fuse_condition(to_fuse_inp_base_model_ids: List[int], model_ids_fused: List[int], fused_model_inp_base_model_ids: List[int]):
    """
        NOTE: we assume after vertical fusion, 
            the input models of the fused model is the same as the input models of the FIRST BASE model in the fused model.
        INPUT:
            1. to_fuse_inp_base_model_ids: the input base model ids of the model (may already be a fused model) to be fused.
            2. model_ids_fused: the base model ids of the fused model we want to add more models to.
            3. fused_model_inp_base_model_ids: the input base model ids of the fused model we want to add more models to.
    """
    cond1 = (len(to_fuse_inp_base_model_ids) == 1) and \
        (to_fuse_inp_base_model_ids[0] == model_ids_fused[-1])
    # cond2 is wrong, because we may fuse some models in the chain starting from the middle of the chain
    # cond2 = (sorted(to_fuse_inp_base_model_ids) == sorted(model_ids_fused))
    cond2 = sorted(to_fuse_inp_base_model_ids) == sorted(fused_model_inp_base_model_ids+model_ids_fused)
    return (cond1 or cond2)



def _can_be_fused_vertically(plan_group: List[MyExecPlan], to_fuse: MyExecPlan, uniq_exec_plan_mapping):
    """
        The condition to do vertical fusion:
        1. supported dependency:
            (a) linear dependency: model 1 -> model 2.
            (b) model i depends on all models before it: (model 1, ..., model i-1) -> model i.
        2. about the same model, has the same exec setting.
        3. the models in both plans have not been started.
    """

    # NOTE: to fuse successfully, plan and to_fuse must not have been started, so to_fuse.model.input_model_ids == to_fuse.model.input_base_model_ids here
    # NOTE: in the case where we fuse some models in the beginning before the search, the above assumption does not hold
    # NOTE: ==> we check whether the first base model of to_fuse can be fused to a plan, if can, then even if to_fuse is a fused model, all its base models can be fused
    # successfully
    # to_fuse_inp_model_ids = to_fuse.model.input_model_ids
    to_fuse_inp_base_model_ids = to_fuse.get_base_models()[0].inp_base_model_ids
    for plan_i, plan in enumerate(plan_group):
        # cond1 = (len(to_fuse_inp_model_ids) == 1) and \
        #     (to_fuse_inp_model_ids[0] == plan.get_base_model_ids()[-1])
        # cond2 = (sorted(to_fuse_inp_model_ids) == sorted(plan.get_base_model_ids()))
        # # if (len(to_fuse_inp_model_ids) == 1) and (to_fuse_inp_model_ids[0] == plan.get_base_model_ids()[-1]):
        # if cond1 or cond2:
        # print(f"in _can_be_fused_vertically: to_fuse_inp_model_ids: {to_fuse_inp_base_model_ids}, to_fuse_base_model_ids: {to_fuse.get_base_model_ids()}, model_ids_fused: {plan.get_base_model_ids()}")
        if _meet_vertical_fuse_condition(to_fuse_inp_base_model_ids, 
                                         model_ids_fused=plan.get_base_model_ids(), 
                                         fused_model_inp_base_model_ids=plan.get_base_models()[0].inp_base_model_ids):
            # to_fuse directly depend on the last exec_plan in the plan (which may be fused)
            if plan.model.get_name() == to_fuse.model.get_name():
                # the two plan are about the same model
                if plan.get_key() == to_fuse.get_key():
                    # the two plan has the same exec setting
                    
                    # check whether these models have been started
                    if plan.models_not_started() and to_fuse.models_not_started():
                        # we can fuse them vertically

                        # print(f"in generate fused model:\n")
                        # print([(_.model_id, _) for _ in plan.get_base_models()+[to_fuse.model]])

                        # to_fuse_models = sorted(plan.get_base_models()+[to_fuse.model], \
                        #                         key=lambda model: model.model_id)
                        # NOTE: seems here we should not sort the base models but keep them in the dependency order
                        # to_fuse_models = plan.get_base_models()+[to_fuse.model]
                        # NOTE: to support the case where we fuse some models at the beginning before the search process
                        to_fuse_models = plan.get_base_models()+to_fuse.get_base_models()
                        
                        # print([(_.model_id, _) for _ in to_fuse_models])

                        # -------------------------------------------------------------------------------------------------
                        # check whether there is available fused exec plan to reuse
                        path_key = _get_path_key(plan_group, plan)[:-1]
                        path_key = path_key+( (tuple(plan.model.get_base_model_ids()+to_fuse.model.get_base_model_ids()), to_fuse.get_key() ) ,)
                        # print(f"plan group: {[str(plan) for plan in plan_group]}")
                        # print(f"path_key: {path_key}")
                        fused = None
                        if path_key in uniq_exec_plan_mapping:
                            # print(f"reuse fused exec plan")
                            fused = uniq_exec_plan_mapping[path_key]
                            # print(f"Reuse exec plan: {str(exec_plan_to_use), exec_plan_to_use.model.model_id}")
                        else:
                            # print(f"generate new exec plan")
                            fused_model = MyFusedModelInfor(to_fuse_models)
                            fused = MyVerticalFusedExecPlan(fused_model, to_fuse)
                            uniq_exec_plan_mapping[path_key] = fused
                            # print(f"New exec plan: {str(exec_plan_to_use), exec_plan_to_use.model.model_id}")

                        # -------------------------------------------------------------------------------------------------
                                        
                        # fused_model = MyFusedModelInfor(to_fuse_models)
                        # fused = MyVerticalFusedExecPlan(fused_model, to_fuse)
                        return [plan_group[:plan_i]+[fused]+plan_group[plan_i+1:]]
    return []












# support vertical fusion of models
def _append_exec_plan(plan_groups, exec_plans_list, depth_i, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping):
    '''
    Get all the possible exec plans with depth-first search.
    The initial plan_groups is [[]], i.e., containing a group with no exec plan.
    All plan groups are valid if they are put into plan_groups and returned.
    NOTE:
        1. here we use good plen group to add pruning the plan group generation process.
    '''
    # stop condition
    if depth_i == len(exec_plans_list):
        return
    
    new_plan_groups = list()
    for plan_group in plan_groups:
        # try to append the exec plans for the current model (depth_i) to the plan group

        for exec_plan in exec_plans_list[depth_i]:

            # check whether we need to copy a new exec plan
            path_key = _get_path_key(plan_group, exec_plan)
            # print(f"path_key: {path_key},    exec_plan: {str(exec_plan)}")
            exec_plan_to_use = None
            if path_key in uniq_exec_plan_mapping:
                exec_plan_to_use = uniq_exec_plan_mapping[path_key]
                # print(f"Reuse exec plan: {str(exec_plan_to_use), exec_plan_to_use.model.model_id}")
            else:
                # exec_plan_to_use = copy.deepcopy(exec_plan)
                exec_plan_to_use = exec_plan.copy_the_plan()
                uniq_exec_plan_mapping[path_key] = exec_plan_to_use
                # print(f"New exec plan: {str(exec_plan_to_use), exec_plan_to_use.model.model_id}")


            # support vertical fusion
            # check whether this plan can be fused vertically with its inp exec plan
            new_plan_groups = new_plan_groups+_can_be_fused_vertically(plan_group, exec_plan_to_use, uniq_exec_plan_mapping)

            if get_tot_worker_num(plan_group) == tot_gpu_num:
                # no model can be added
                continue

            tmp_plan_group = plan_group + [exec_plan_to_use]
            # check valid
            # print(f"uniq_exec_plan_mapping: {[(k,str(v)) for k,v in uniq_exec_plan_mapping.items()]}")
            # print(f"checking plan group: {[str(_) for _ in tmp_plan_group]}")
            if is_valid_exec_plan_combination(tmp_plan_group, tot_gpu_num, byte_per_gpu):
                new_plan_groups.append(tmp_plan_group)

                # print(f"valid")

    plan_groups.extend(new_plan_groups)
    _append_exec_plan(plan_groups, exec_plans_list, depth_i+1, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping)




# TODO: 这个地方可能还是要把model vertical fusion的功能加进来，因为这个函数也会在根据exec plan greedy地做选择的时候被调用，而这种情况
# 我们并不会无脑vertical fuse model
def _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
        plan_groups, exec_plans_list, depth_i, tot_gpu_num, byte_per_gpu,
        cost_table: CostTable,
        last_stage_exec_plans: List[MyExecPlan],
        check_gap: int, sort_input: bool,):
    '''
    Get all the possible exec plans with depth-first search.
    The initial plan_groups is [[]], i.e., containing a group with no exec plan.
    All plan groups are valid if they are put into plan_groups and returned.
    '''

    # print(f"depth_i: {depth_i}")

    # stop condition
    if depth_i == len(exec_plans_list):
        return
    
    new_plan_groups = list()
    for plan_group in plan_groups:
        # try to append the exec plans for the current model (depth_i) to the plan group
        if get_tot_worker_num(plan_group) == tot_gpu_num:
            # no model can be added
            continue
        for exec_plan in exec_plans_list[depth_i]:
            tmp_plan_group = plan_group + [exec_plan]
            # check valid
            # print(f"checking plan group 1: {[str(_) for _ in tmp_plan_group]}")
            if is_valid_exec_plan_combination(tmp_plan_group, tot_gpu_num, byte_per_gpu):
                # print(f"new_plan_groups: {new_plan_groups}", flush=True)
                new_plan_groups.append(tmp_plan_group)

                # print(f"valid")

    # plan_groups.extend(new_plan_groups)
    # print(f"new_plan_groups: {[[str(_) for _ in tmp_plan_group] for tmp_plan_group in new_plan_groups]}")
    # print(f"plan_groups 1: {[[str(_) for _ in tmp_plan_group] for tmp_plan_group in plan_groups]}")

    if len(new_plan_groups) > 0:
        new_plan_groups = [
            MyExecPlanGroup(exec_plans, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                check_gap=check_gap, sort_input=sort_input)\
            for exec_plans in new_plan_groups]
        exec_plans = sorted(new_plan_groups, key=lambda i: i.get_throughput(), reverse=True)[0].exec_plans
        # print(f"plan_groups: {[str(i) for i in exec_plans]}")
        plan_groups[0] = exec_plans

    _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
        plan_groups, exec_plans_list, depth_i+1, tot_gpu_num, byte_per_gpu,
        cost_table, last_stage_exec_plans,
        check_gap, sort_input,)

    # print(f"plan_groups 2: {[[str(_) for _ in tmp_plan_group] for tmp_plan_group in plan_groups]}")




# def get_plan_group_str(plan_group):
#     '''
#     Get the string to represent a plan group
#     '''
#     return str(sorted([str(exec_plan) + str(exec_plan.model.get_left_flops_per_token()) for exec_plan in plan_group]))



def get_one_stage_exec_plans_sorted(
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        model_sys: MyModelSystem, 
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        top_k=float('inf')):
    '''
    Get a set of exec plans which can work corrently on the given multi-GPU environment.
    '''

    # print(f"model_list: {[str(model) for model in model_list]}")
    model_sys.print_model_list()

    # 1. first get the candidate exec plans for each model
    # exec_plans_list = list()
    # for model in model_list:
    #     exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline)
    #     exec_plans_list.append(exec_plans)
    # # 
    # plan_groups = [[]]
    # _append_exec_plan(plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu)
    # plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans) \
    #                for plan_group in plan_groups if len(plan_group) > 0]
    
    # allow models with dependency to be selected into the same group
    plan_groups = model_sys.get_candidate_plan_groups(
        gen_execplans_baseline, check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num, byte_per_gpu, top_k)
    

    print(f"in get_one_stage_exec_plans_sorted: the plan groups we generated: ")
    for plan_group in plan_groups:
        print(f"{len(plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans]}")


    # print(f"\nfinish plan group gen\n")

    # 2. delete plan_groups which do not occupy all comp gpus when there are models not executed
    # also for cases where there are idle comp resources, check whether increase comp resources can improve throughput, if yes, delete it.
    # TODO (jingzhi) it seems not_finished_model_num is not used
    # not_finished_model_num = sum([not model.is_finished() for model in model_list])
    not_finished_model_num = model_sys.get_not_finished_base_model_num()
    useful_plan_groups = list()
    idle_comp_plan_groups = dict() # {models executed: (best_throughput, best_plan_group)}
    # comp_throughput_given_plan_groups(plan_groups, gpu_name)


    # print(f"\nfinish plan group gen 2\n")

    # 2.1 first delete inefficient plan groups for each possible set of involved models
    for plan_group in plan_groups:
        
        # print(f"check redundancy 1 of: {str(plan_group)}")

        key = plan_group.get_model_states_before_infer_stage()

        # print(f"before key: {key}")

        if key not in idle_comp_plan_groups:
            # print(f"new key (before infer stage): {key}")
            idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
        else:
            # print(f"old key (before infer stage): {key}")
            if plan_group.get_throughput() > idle_comp_plan_groups[key][0]:
                idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)


    # print(f"\nfinish plan group gen 3\n")

    plan_groups: List[MyExecPlanGroup] = [plan_group for _, plan_group in idle_comp_plan_groups.values()]
    idle_comp_plan_groups = dict() # {models executed: (best_throughput, best_plan_group)}

    # 2.2 then delete inefficient plan groups for each possible set of model states after this infer stage
    for plan_group in plan_groups:

        # print(f"check redundancy 2 of: {str(plan_group)}")


        # 1. if there are models available but the comp gpus are not fully utilized
        if (not_finished_model_num > plan_group.get_involved_base_model_num()) and (get_tot_worker_num(plan_group.exec_plans)<tot_gpu_num):
            continue

        # print(f"check redundancy 2 -- fully resource utilization!")

        # the key of a plan group is [(model_i name, model_i progressing status)]
        
        # key = tuple(\
        #     sorted([\
        #         (str(exec_plan.model), exec_plan.model.get_left_flops_per_token()) \
        #             for exec_plan in plan_group.exec_plans\
        #             ]))
        
        # the key of a plan group now is [(model_i name, model_i remaining seqlens)]
        key = plan_group.get_model_states_after_infer_stage(cost_table)
        
        # print(f"after key: {key}")
        
        # print(f"key of plan group: {key}")
        # if key in idle_comp_plan_groups:
        #     print(f"current best equivalent plan groups: {idle_comp_plan_groups[key]}")
        # else:
        #     print(f"current best equivalent plan groups: {None}")




        if key not in idle_comp_plan_groups:
            
            # print(f"new key (after infer stage): {key}")
            # print(f"plan_group.get_throughput(): {plan_group.get_throughput()}")

            idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
        else:

            # print(f"old key (after infer stage): {key}")
            # print(f"plan_group.get_throughput(): {plan_group.get_throughput()} > {idle_comp_plan_groups[key][0]} ")

            if plan_group.get_throughput() > idle_comp_plan_groups[key][0]:
                idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
    for _, plan_group in idle_comp_plan_groups.values():
        useful_plan_groups.append(plan_group)

        # else:
        #     print(f"inefficient plan group: {str(plan_group)}")

    
    # print(f"\nfinish plan group gen 5\n")

    # TODO (jingzhi) strange, the check below seems to be useless
    # 3. delete redundant plan_groups (in case there are models that are the same)
    # uniq_plan_groups_strs = set()
    # uniq_plan_groups = list()
    # for plan_group in useful_plan_groups:
    #     plan_group_str = str(plan_group)
    #     if plan_group_str not in uniq_plan_groups_strs:
    #         uniq_plan_groups_strs.add(plan_group_str)
    #         uniq_plan_groups.append(plan_group)
    #     # else:
    #     #     print(f"redundant plan group: {str(plan_group)}")

    uniq_plan_groups = useful_plan_groups

    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    print(f"the uniq plan groups we get:")
    for plan_group in uniq_plan_groups:
        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans], plan_group.infer_stage_latency, plan_group.get_throughput()}")


    # sort plan groups according to the overall throughput
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.get_throughput(), reverse=True)

    return uniq_plan_groups




def get_one_stage_exec_plans_sorted_greedy_baseline_adapted_from_MuxServe(
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        model_sys: MyModelSystem, 
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    '''
    Get a set of exec plans which can work corrently on the given multi-GPU environment.
    NOTE: this function select exec plans in a stage 
        (1) in the order of models from large to small OR 
        (2) in the order of model's remaining flops from large to small [bad idea].
    '''

    uniq_plan_groups = model_sys.get_candidate_plan_groups_greedy_baseline_adapted_from_MuxServe_best_model_first(
        gen_execplans_baseline,
        check_gap, sort_input,
        last_stage_exec_plans,
        cost_table,
        tot_gpu_num, byte_per_gpu)

    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    print(f"the uniq plan groups we get:")
    for plan_group in uniq_plan_groups:
        print(str(plan_group))


    # sort plan groups according to the overall throughput
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.get_throughput(), reverse=True)

    return uniq_plan_groups



    # sort models
    # in the order of parameter size
    sorted_models = list()
    for model in model_list:
        # here the original tp_size is set to 2, but it is strange, so change it to 1
        # param_byte_per_layer, extra_byte = \
        #         get_per_layer_and_extra_param_and_buffer_byte(model, 2)
        param_byte_per_layer, extra_byte = \
                get_per_layer_and_extra_param_and_buffer_byte(model, tp_size=1)
        sorted_models.append((model, param_byte_per_layer*model.layer_num + extra_byte))
    
    model_list = sorted(sorted_models, key=lambda i: i[1], reverse=True)
    model_list = [i[0] for i in model_list]

    print(f"model_list: {[str(model) for model in model_list]}")

    # 1. first get the candidate exec plans for each model
    exec_plans_list = list()
    for model in model_list:
        exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline)
        exec_plans_list.append(exec_plans)
        # print(f"exec_plans: {exec_plans}")
    # 
    plan_groups = [[]]
    # _append_exec_plan(plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu)
    _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
        plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu,
        cost_table, last_stage_exec_plans)

    # print(f"plan_groups: {[[str(_) for _ in tmp_plan_group] for tmp_plan_group in plan_groups]}")


    plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans) \
                   for plan_group in plan_groups if len(plan_group) > 0]

    
    
    uniq_plan_groups = plan_groups

    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    print(f"the uniq plan groups we get:")
    for plan_group in uniq_plan_groups:
        print(str(plan_group))


    # sort plan groups according to the overall throughput
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.get_throughput(), reverse=True)

    return uniq_plan_groups







# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# helper functions

# def recover_model_state(
#         model_list:List[MyModelInfor], inp_out_lens_list: List[Tuple[List[int], List[int]]], 
#         cost_table: CostTable, remaining_decode_flops_list: List[float]):
#     # for model, inp_out_lens in zip(model_list, inp_out_lens_list):
#     for i in range(len(model_list)):
#         model = model_list[i]
#         inp_out_lens = inp_out_lens_list[i]
#         remaining_decode_flops = remaining_decode_flops_list[i]
#         model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)



# def models_are_finished(model_list: List[MyModelInfor]):
#     return False not in [model.is_finished() for model in model_list]


# def get_model_states(model_list: List[MyModelInfor]):
#     '''
#         Get the current inference progress of the given list of models.
#         NOTE: the returned progress should be able to be added to a set.
#     '''
#     return tuple(sorted([model.get_state() for model in model_list]))



# def get_model_left_decode_flops(model_list: List[MyModelInfor], cost_table: CostTable):
#     return sum([model.get_remaining_flops() for model in model_list])


def get_total_model_flops(model_list: List[MyModelInfor], cost_table: CostTable):
    flops = 0
    for model in model_list:
        assert not isinstance(model, MyFusedModelInfor)
        i = fake_scheduling.comp_flops_from_seqlens(
            model.inp_lens, model.out_lens, only_decode=False, cost_table=cost_table, 
            model_path=model.model_path, trust_remote_code=model.trust_remote_code, revision=model.revision)
        flops += i
    return flops


# def merge_inp_out_lens_of_data_parallel_workers(inp_out_lens_list: List[List[List[int]]])->List[List[int]]:
#     inp_lens_list = [dp_inp_out_lens[0] for dp_inp_out_lens in inp_out_lens_list]
#     out_lens_list = [dp_inp_out_lens[1] for dp_inp_out_lens in inp_out_lens_list]
#     inp_seq_ids_list = [dp_inp_out_lens[2] for dp_inp_out_lens in inp_out_lens_list]
#     inp_lens = np.concatenate(inp_lens_list)
#     out_lens = np.concatenate(out_lens_list)
#     inp_seq_ids = np.concatenate(inp_seq_ids_list)
#     # order = np.argsort(-inp_lens)
#     # sort by inp seqs ids
#     order = np.argsort(inp_seq_ids)
#     inp_lens = inp_lens[order]
#     out_lens = out_lens[order]
#     inp_seq_ids = inp_seq_ids[order]
#     return [inp_lens, out_lens, inp_seq_ids]



def get_sorted_models_by_model_size(model_list: List[MyModelInfor]):
    sorted_models = list()
    for model in model_list:
        # here the original tp_size is set to 2, but it is strange, so change it to 1
        # param_byte_per_layer, extra_byte = \
        #         get_per_layer_and_extra_param_and_buffer_byte(model, 2)
        param_byte_per_layer, extra_byte = \
                get_per_layer_and_extra_param_and_buffer_byte(model, tp_size=1)
        sorted_models.append((model, param_byte_per_layer*model.layer_num + extra_byte))
    
    sorted_models = sorted(sorted_models, key=lambda i: i[1], reverse=True)
    sorted_models = [i[0] for i in sorted_models]
    return sorted_models

# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================











# we compute the best model execution plan for the given model list.
# assumption: the output lengths of all the models are the same.
# we can directly use a table for the cost model.
def _get_best_model_schedule(
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        cost_table: CostTable, 
        model_sys: MyModelSystem, 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: dict,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''
    # check the max seq-to-check number
    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    if _CHECKED_SEQ_NUM > _MAX_SEQ_NUM:
        return


    print(f"CURRENT PLAN GROUP SEQ: {str(curr_group_seq)}, {[[_.model.get_base_model_ids() for _ in plan_group.exec_plans] for plan_group in curr_group_seq.plan_group_seq if plan_group!=None]}")
    print(f"CURRENT BEST PLAN GROUP SEQ: {str(best_group_seq)}, {[[_.model.get_base_model_ids() for _ in plan_group.exec_plans] for plan_group in best_group_seq.plan_group_seq if plan_group!=None]}")



    # 1. check stop condition
    # NOTE: after introducing data parallel, in each stage maybe only a part of a model will finish
    # 但是感觉这样的话，整个scheduling的过程会变成砍一刀砍一刀，被砍得很细，效率很低，其实，虽然会自动重新均匀分配dp workload
    # solution 1：就这么做；2：一个model的运行时间按照时间最长的dp worker来算，在此基础上估计plan group的运行时间。
    # 先试试solution1 ==> solution1 搜索空间太大了，换成solution2。
    # stop condition of the depth-first search: there are <= len(model_list) exec stages 
    # since in each stage >= 1 models will finish
    if len(curr_group_seq.plan_group_seq) > model_sys.get_model_num():
        # assert False, f'{[model.left_flops_per_token for model in model_list]},'\
        #             f'{[[str(_) for _ in group] for group in curr_group_seq]}'
        assert False, f'{[[str(_) for _ in group.exec_plans] for group in curr_group_seq.plan_group_seq]}'


    print(f"finish step 1:      curr depth is within limit")

    # 1. check stop condition
    # exit the loop when all model are finished
    # if models_are_finished(model_list):
    if model_sys.is_finished():
        
        print(f"all models finished")
        
        _CHECKED_SEQ_NUM += 1

        # update best plan seq info
        # print(f"curr_plan_seq return: {curr_plan_seq}")
        if curr_group_seq.get_tot_time() < best_group_seq.get_tot_time():
            best_group_seq.set_plan_group_and_time(curr_group_seq.plan_group_seq, curr_group_seq.time_seq)
        return 
    # 

    if curr_group_seq.get_tot_time() >= best_group_seq.get_tot_time():
        # the current group seq is definitely slower than the currently best group seq, no need to search deeper
        return


    print(f"finish step 2")

    # 2. get the model states before the current infer stage, check its redundancy
    # model_states = model_sys.get_model_states()
    model_states = model_sys.get_base_model_states()


    if model_states in uniq_model_states:
        # do not need to check this plan group, as we have check an equivalent one
        # print(f"redundant plan group due to model states: {model_states, str(curr_plan_seq)}")
        # return

        # only when the latency of the current group seq is higher, we do not check it further
        if curr_group_seq.get_tot_time() > uniq_model_states[model_states]:

            print(f"the state has been checked and curr is not the best choice for it")
            print(f"model_states: {model_states}")
            print(f"uniq_model_states: {uniq_model_states}")

            return
        else:
            uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    else:
        # update uniq_model_states
        # uniq_model_states.add(model_states)
        uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    # print(f"curr_plan_seq: {curr_group_seq}")
    # print(f"model_states: {model_states}, uniq_model_states: {uniq_model_states}")
        

    print(f"finish step 3")


    # 3. get the possible plan groups for the current infer stage.
    plan_groups: List[MyExecPlanGroup] = get_one_stage_exec_plans_sorted(
        gen_execplans_baseline,
        check_gap, sort_input,
        curr_group_seq.get_last_stage_exec_plans(),
        cost_table, model_sys, gpu_name, tot_gpu_num, byte_per_gpu, top_k)
    # # ori_left_flops = [model.get_left_flops_per_token() for model in model_list]
    # # ori_left_seqlens = [model.get_remaining_seqlens() for model in model_list]
    # ori_inp_out_lens_list = [model.get_inp_out_seqlens() for model in model_list]
    # ori_remaining_decode_flops_list = [model.get_remaining_flops() for model in model_list]
    ori_inp_out_lens_list = model_sys.get_model_inp_out_lens()
    ori_remaining_decode_flops_list = model_sys.get_model_remaining_decode_flops()
    ori_inp_seq_ids_list = model_sys.get_model_inp_seq_ids()
    ori_inp_model_ids_list = model_sys.get_model_inp_model_ids()
    ori_model_sys = model_sys

    
    # 3.1 pruning rule: if using the highest throughput in plan_groups still cannot beat best_group_seq, skip all of them
    # if (get_model_left_decode_flops(model_list, cost_table) / plan_groups[0].get_comp_throughput_only() + curr_group_seq.get_tot_time()) \
    #     >= best_group_seq.get_tot_time():
    # TODO:这个地方的early prune逻辑应该改成什么？如果该system可以有唯一地划分成两部分的划分方式，那么就可以把各个plan group
    # 按照各个阶段来对比throughput。最general的写法应该是这样的，但是感觉好复杂啊。可以按照一个model距离起始点的所有路径长度是否一致
    # 来对原始的system进行阶段分割。但是这样好像也没用，还是没法做原来的early pruning。还是简单对进入到最后一个阶段的plan group seq 
    # 进行early pruning吧。。。。。搜索算法之后可以想办法再优化。
    tot_ori_remaining_decode_flops = sum([\
        sum(flops) if isinstance(flops, list) else flops \
         for flops in ori_remaining_decode_flops_list])
    if (model_sys.remaining_models_are_on_the_last_layer()) \
        and ((tot_ori_remaining_decode_flops / plan_groups[0].get_comp_throughput_only() \
              + curr_group_seq.get_tot_time()) >= best_group_seq.get_tot_time()):
        
        print(f"using the highest throughput in plan_groups still cannot beat best_group_seq")
        
        return


    print(f"finish step 4")


    # 4. try each candidate plan group and do depth-first search.
    for plan_group in plan_groups:
        print(f"trying adding plan_group: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans]}, models are finished? {[plan.model.is_finished() for plan in plan_group.exec_plans]}")
        if len(plan_group) == 0:
            continue

        # TODO: 这个地方在考虑了extra preparation cost之后可能要修改-->已经改成了不考虑extra cost的近似throughput，是一个会偏高的估计
        # if avg valid throughput of adding plan_group to current group seq is lower than the current_group_seq's throughput, skip it
        # if curr_group_seq.get_tmp_throughput_after_adding_a_plan_group(plan_group) \
        # TODO: 此处的early pruning道理等同于上面loop外部的pruning [加入remaining_models_are_on_the_last_layer的判断条件]
        if (model_sys.remaining_models_are_on_the_last_layer()) \
            and (curr_group_seq.get_tmp_only_comp_throughput_after_adding_a_plan_group(plan_group) \
                 < best_group_seq.get_valid_throughput()):
            continue

        # we first recover the ori model sys
        model_sys = ori_model_sys
        
        # update the remaining workload of all models after this stage
        # recover_model_state(model_list, ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        
        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        model_sys.recover_model_state(
            ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list,
            ori_inp_model_ids_list)
        
        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        # we need to update model_sys as we may introduce fused model nodes
        model_sys = model_sys.gen_new_model_sys_with_fused_models(
            fused_model_list=plan_group.get_involved_fused_models())

        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        print(f"plan_group model objects: {[(plan.model.model_id, plan.model) for plan in plan_group.exec_plans]}")

        # check inp out len accuracy
        model_sys.check_finish_states_accuracy()

        # comp_time = update_model_state(plan_group, gpu_name)
        plan_group.update_model_inp_out_lens(cost_table)

        # check inp out len accuracy
        model_sys.check_finish_states_accuracy()

        curr_group_seq.append_plan_group(plan_group)
        curr_group_seq.append_exec_time(plan_group.get_infer_stage_latency())
        _get_best_model_schedule(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table,
            model_sys, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu, top_k)
        curr_group_seq.pop_one_stage()
    # print(f"best_group_seq: {best_group_seq}")






# we compute the best model execution plan for the given model list.
# assumption: the output lengths of all the models are the same.
# we can directly use a table for the cost model.
def _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe(
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        cost_table: CostTable, 
        model_sys: MyModelSystem, 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: dict,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''

    # check the max seq-to-check number
    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    if _CHECKED_SEQ_NUM > _MAX_SEQ_NUM:
        return

    print(f"CURRENT PLAN GROUP SEQ: {str(curr_group_seq)}")
    print(f"CURRENT BEST PLAN GROUP SEQ: {str(best_group_seq)}")



    # 1. check stop condition
    # stop condition of the depth-first search: there are <= len(model_list) exec stages 
    # since in each stage >= 1 models will finish
    if len(curr_group_seq.plan_group_seq) > model_sys.get_model_num():
        # assert False, f'{[model.left_flops_per_token for model in model_list]},'\
        #             f'{[[str(_) for _ in group] for group in curr_group_seq]}'
        assert False, f'{[[str(_) for _ in group.exec_plans] for group in curr_group_seq]}'


    print(f"finish step 1:      curr depth is within limit")

    # 1. check stop condition
    # exit the loop when all model are finished
    # if models_are_finished(model_list):
    if model_sys.is_finished():
        
        print(f"all models finished")

        _CHECKED_SEQ_NUM += 1
        
        # update best plan seq info
        # print(f"curr_plan_seq return: {curr_plan_seq}")
        if curr_group_seq.get_tot_time() < best_group_seq.get_tot_time():
            best_group_seq.set_plan_group_and_time(curr_group_seq.plan_group_seq, curr_group_seq.time_seq)
        return 
    # 

    if curr_group_seq.get_tot_time() >= best_group_seq.get_tot_time():
        # the current group seq is definitely slower than the currently best group seq, no need to search deeper
        return


    print(f"finish step 2")

    # 2. get the model states before the current infer stage, check its redundancy
    # model_states = get_model_states(model_list)
    # model_states = model_sys.get_model_states()
    model_states = model_sys.get_base_model_states()


    if model_states in uniq_model_states:
        # do not need to check this plan group, as we have check an equivalent one
        # print(f"redundant plan group due to model states: {model_states, str(curr_plan_seq)}")
        # return

        # only when the latency of the current group seq is higher, we do not check it further
        if curr_group_seq.get_tot_time() > uniq_model_states[model_states]:

            print(f"the state has been checked and curr is not the best choice for it")

            return
        else:
            uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    else:
        # update uniq_model_states
        # uniq_model_states.add(model_states)
        uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    # print(f"curr_plan_seq: {curr_group_seq}")
    # print(f"model_states: {model_states}, uniq_model_states: {uniq_model_states}")
        

    print(f"finish step 3")


    # 3. get the possible plan groups for the current infer stage.
    plan_groups: List[MyExecPlanGroup] = get_one_stage_exec_plans_sorted_greedy_baseline_adapted_from_MuxServe(
        gen_execplans_baseline,
        check_gap, sort_input,
        curr_group_seq.get_last_stage_exec_plans(),
        cost_table, model_sys, gpu_name, tot_gpu_num, byte_per_gpu)
    # # ori_left_flops = [model.get_left_flops_per_token() for model in model_list]
    # # ori_left_seqlens = [model.get_remaining_seqlens() for model in model_list]
    # ori_inp_out_lens_list = [model.get_inp_out_seqlens() for model in model_list]
    # ori_remaining_decode_flops_list = [model.get_remaining_flops() for model in model_list]
    ori_inp_out_lens_list = model_sys.get_model_inp_out_lens()
    ori_remaining_decode_flops_list = model_sys.get_model_remaining_decode_flops()
    ori_inp_seq_ids_list = model_sys.get_model_inp_seq_ids()
    ori_inp_model_ids_list = model_sys.get_model_inp_model_ids()
    ori_model_sys = model_sys



    # 3.1 pruning rule: if using the highest throughput in plan_groups still cannot beat best_group_seq, skip all of them
    # if (get_model_left_decode_flops(model_list, cost_table) / plan_groups[0].get_comp_throughput_only() + curr_group_seq.get_tot_time()) \
    #     >= best_group_seq.get_tot_time():
        
    #     print(f"using the highest throughput in plan_groups still cannot beat best_group_seq")
        
    #     return


    print(f"finish step 4")


    # 4. try each candidate plan group and do depth-first search.
    for plan_group in plan_groups:
        # print(f"plan_group: {plan_group}")
        if len(plan_group) == 0:
            continue

        # TODO: 这个地方在考虑了extra preparation cost之后可能要修改-->已经改成了不考虑extra cost的近似throughput，是一个会偏高的估计
        # # if avg valid throughput of adding plan_group to current group seq is lower than the current_group_seq's throughput, skip it
        # # if curr_group_seq.get_tmp_throughput_after_adding_a_plan_group(plan_group) \
        # if curr_group_seq.get_tmp_only_comp_throughput_after_adding_a_plan_group(plan_group) \
        #     < best_group_seq.get_valid_throughput():
        #     continue

        # we first recover the ori model sys
        model_sys = ori_model_sys


        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        model_sys.recover_model_state(
            ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list,
            ori_inp_model_ids_list)
        
        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        # we need to update model_sys as we may introduce fused model nodes
        model_sys = model_sys.gen_new_model_sys_with_fused_models(
            fused_model_list=plan_group.get_involved_fused_models())

        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        print(f"plan_group model objects: {[(plan.model.model_id, plan.model) for plan in plan_group.exec_plans]}")

        # check inp out len accuracy
        model_sys.check_finish_states_accuracy()

        # comp_time = update_model_state(plan_group, gpu_name)
        plan_group.update_model_inp_out_lens(cost_table)

        # check inp out len accuracy
        model_sys.check_finish_states_accuracy()



        # update the remaining workload of all models after this stage
        # recover_model_state(model_list, ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        # model_sys.recover_model_state(
        #     ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        # # comp_time = update_model_state(plan_group, gpu_name)
        # plan_group.update_model_inp_out_lens(cost_table)
        
        
        curr_group_seq.append_plan_group(plan_group)
        curr_group_seq.append_exec_time(plan_group.get_infer_stage_latency())
        _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table,
            model_sys, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu)
        curr_group_seq.pop_one_stage()
    # print(f"best_group_seq: {best_group_seq}")




# we compute the best model execution plan for the given model list.
# assumption: the output lengths of all the models are the same.
# we can directly use a table for the cost model.
def _get_best_model_schedule_dispatcher(
        search_method_baseline: str,
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        cost_table: CostTable, 
        model_sys: MyModelSystem, 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: dict,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), top_k=float('inf'),):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''
    if search_method_baseline == 'ours':
        _get_best_model_schedule(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table, 
            model_sys, 
            curr_group_seq, 
            best_group_seq, 
            uniq_model_states,
            gpu_name, tot_gpu_num, byte_per_gpu, top_k)
    else:
        assert False, "ERROR: we currently do not support greedy search algorithm"
        _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table, 
            model_sys, 
            curr_group_seq, 
            best_group_seq, 
            uniq_model_states,
            gpu_name, tot_gpu_num, byte_per_gpu)        





def get_engin_args(model_path, tensor_parallel_size):
    backend = "ours"
    model = model_path
    tokenizer = None
    quantization = None
    # tensor_parallel_size = 1
    n = 1
    use_beam_search = False
    # num_prompts = 1000
    seed = 0
    hf_max_batch_size = None
    trust_remote_code = True
    max_model_len = None
    dtype = 'auto'
    enforce_eager = True
    kv_cache_dtype = "auto"
    device = "cuda"
    # weight_load_degree = wldeg
    gpu_use_ratio = 0.9
    # temperature = 1.0
   
    # copy the args check from vllm-----------------------------------------
    if tokenizer is None:
        tokenizer = model

    # get engine args
    from vllm import LLM
    (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config) = LLM.get_engine_configs_only(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        # <jingzhi>
        # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
        # max_num_seqs=2048,
        gpu_memory_utilization=gpu_use_ratio,
        max_num_seqs=512,
        max_paddings=512,
    )
    return (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config)






# TODO: 根据model config计算model weight parameter的数据。可以把这部分计算挪到exec plan里面。
# TODO: the functions only support llama2 models now.
# compute model parameter mem in bytes (about the weights not in the decode layers)
def get_extra_param_byte_manual_computation(model_info: MyModelInfor, tp_size: int):
    '''
        Input:
            tp_size: the number of tensor parallel workers.
    '''
    assert "Llama" in model_info.model_name, "we only support llama models now"

    hf_config = model_info.hf_config
    V: int = hf_config.vocab_size
    h: int = hf_config.hidden_size

    # rotary_emb: only store one copy of the cos_sin_cache
    max_position_embeddings = getattr(hf_config, "max_position_embeddings",
                                          8192)

    rotary_dim=hf_config.hidden_size // hf_config.num_attention_heads
    cos_sin_cache = max_position_embeddings*rotary_dim


    extra_param_byte = (2*V*h/tp_size+h+cos_sin_cache) * model_info.data_byte
    return extra_param_byte




# compute model parameter mem in bytes (about the weights of the decode layers)
def get_param_byte_per_layer_manual_computation(model_info: MyModelInfor, tp_size: int):
    '''
        Input:
            comp_worker_num: the number of tensor parallel workers.
    '''
    assert "Llama" in model_info.model_name, f"we only support llama models now, but the model name is {model_info.model_name}"

    hf_config = model_info.hf_config

    total_num_heads = hf_config.num_attention_heads
    head_dim = hf_config.hidden_size // total_num_heads
    total_num_kv_heads = hf_config.num_key_value_heads

    # QKV: 
    input_size = hf_config.hidden_size
    num_heads = total_num_heads// tp_size
    num_kv_heads = max(total_num_kv_heads//tp_size, 1)
    output_size = (num_heads+2*num_kv_heads)*tp_size*head_dim
    W = output_size/tp_size*input_size

    # o proj
    # W = [total_num_heads*head_dim/tp_size, hf_config.hidden_size]
    W = W + hf_config.hidden_size/tp_size*hf_config.hidden_size

    # rotary_emb: only store one copy of the cos_sin_cache
    # max_position_embeddings = getattr(hf_config, "max_position_embeddings",
    #                                       8192)
    # rotary_dim=head_dim
    # cos_sin_cache = max_position_embeddings*rotary_dim
    # W = W + cos_sin_cache

    # attn: no weight

    # gate_up_proj
    W = W + hf_config.hidden_size*hf_config.intermediate_size*2/tp_size


    # down_proj
    W = W + hf_config.intermediate_size/tp_size * hf_config.hidden_size

    # act_fn: no weight

    # input_layernorm
    W = W + hf_config.hidden_size
    # post_attention_layernorm
    W = W + hf_config.hidden_size

    # 所以大部分layer的计算每个worker的计算量和tp_size是有关的，除了RMSNorm这个layer。再print出layer信息来确认一下。

    return W * model_info.data_byte

    # below is the old version, which does not consider kv_head_num != head_num

    _, h,I, _ = model_info.model_config
    param_byte_per_layer =  (4*h*h/comp_worker_num+3*I*h/comp_worker_num+2*h) * model_info.data_byte
    return param_byte_per_layer




# compute model parameter mem in bytes (about the weights of the decode layers)
def get_per_layer_and_extra_param_and_buffer_byte(
        model_info: MyModelInfor, tp_size: int):
    '''
        Input:
            tp_size: the number of tensor parallel workers.

        NOTE: compute according to model.parameters() and model.buffers().
    '''
    per_layer, extra = model_sizes[(model_info.model_path, tp_size)]
    if per_layer == None:
        per_layer = 1e12
        extra = 1e12
    return per_layer, extra





def get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config):
    cache_block_size = CacheEngine.get_cache_block_size(
            cache_config.block_size, cache_config.cache_dtype, model_config, parallel_config)
    return cache_block_size




def get_model_info_objs(
        cost_table: CostTable,
        data_byte: int,
        inp_lens: List[int],
        model_paths: List[str], 
        inp_seq_ids_dict: Dict[int, int],
        outlen_generator,
        # edge_dict: Dict[int, List[int]],
        sample_config: Tuple[float, float, float, float], 
        trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Get the list of MyModelInfor objects for the given model paths.
        NOTE: 
            inp_seq_ids_dict: stores the ids of the inp seqs each model needs to answer. 
                Support the chain summary case where each LLM stage has different number of inp reqs.
    '''
    # out_lens_dict = {model_path: output_length_sampler.sample_out_len_for_given_model(
    #         model=model_path[model_path.find('/')+1:], inp_lens=inp_lens) for model_path in set(model_paths)}
    out_lens_dict = {model_path: outlen_generator(
            model_path[model_path.find('/')+1:], inp_lens) for model_path in set(model_paths)}

    # try to use the output lengths set by the SharedGPT dataset
    # out_lens_dict = {model_path: get_outlens()  for model_path in set(model_paths)}

    return [MyModelInfor(
                model_id,
                cost_table,
                model_path, # model_info
                outlen_generator,
                sample_config, trust_remote_code, revision,
                data_byte, # the number of bytes to store a value in model weights or intermediate tensors
                inp_lens,
                out_lens=out_lens_dict[model_path],
                inp_seq_ids=inp_seq_ids_dict[model_id]
                # input_model_ids=edge_dict[model_id],
            ) for model_id, model_path in enumerate(model_paths)]





def get_inplens(req_num: int):
    import json
    def get_lens(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'output_lens =' in line:
                    pos = len('output_lens =')
                    values = json.loads(line[pos:])
                    return values

    filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_7.log' # not ignore eos
    lens = get_lens(filename)
    inps = [i[0] for i in lens]
    return inps[:req_num]


def get_outlens():
    import json
    def get_lens(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'output_lens =' in line:
                    pos = len('output_lens =')
                    values = json.loads(line[pos:])
                    return values

    filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_7.log' # not ignore eos
    lens = get_lens(filename)
    outs = [i[2] for i in lens] # the out len set by the dataset
    return outs[:]



def get_best_model_schedule(
        search_method_baseline: str,
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        model_paths: List[str], 
        # 
        num_prompts, inp_seq_ids_dict, out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]], 
        inp_generator, inp_merger, outlen_generator,
        # 
        out_edge_dict: Dict[int, List[int]],
        sample_config: Tuple[float, float, float, float],
        trust_remote_code:bool=True, revision:Optional[str] = None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=float('inf'), top_k=float('inf'), similar_threshold: float=0.1):
    """
        NOTE: ``inp_generator``, ``inp_merger``, ``outlen_generator`` are 3 functions about model inp/out lens.
    """

    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM, _MODEL_ID
    _MAX_SEQ_NUM = max_group_seq_num
    _CHECKED_SEQ_NUM = 0

    import time
    time1 = time.perf_counter()

    # 1. first initialize cost_table
    cost_table = get_cost_table()

    # 2. get input lengths
    # inp_lens = get_inplens()
    inp_lens = inp_generator(num_prompts)
    print(f"len(inp_lens): {len(inp_lens)}")

    # 3.  initialize model info objects and the model system object
    model_list: List[MyModelInfor] = get_model_info_objs(
        cost_table,
        data_byte, inp_lens, model_paths, inp_seq_ids_dict, outlen_generator, sample_config, trust_remote_code, revision)
    
    model_sys = MyModelSystem(model_list=model_list, out_edge_dict=out_edge_dict, 
                              cost_table=cost_table, inp_merger=inp_merger, outlen_generator=outlen_generator,
                              need_correct_inp_out_lens=True, 
                              out_req_id_mapping=out_req_id_mapping)

    _MODEL_ID = len(model_list)
    # 1. correct the model ori remaining decoding flops
    # 2. set the inp_base_model_ids
    for model in model_sys.model_dict.values():
        model.ori_tot_remaining_decode_flops = model.remaining_decode_flops
        model.inp_base_model_ids = model.input_model_ids

    # 3. directly fuse some models vertically to reduce the total model number in the system for faster and better search
    # TODO: 这个地方对于search method的naive版本我们其实有两种变体，所以之后可能还需要用不同的str来控制。
    if search_method_baseline == 'naive' or gen_execplans_baseline == 'naive':
        similar_threshold = float('inf')
    model_sys = model_sys.fuse_similar_models_in_a_chain(
            tot_gpu_num, byte_per_gpu, cost_table,
            check_gap, sort_input,
            similar_threshold)

    total_flops = get_total_model_flops(model_list, cost_table)
    curr_group_seq = MyExecPlanGroupSeq(total_flops, [], [])
    best_group_seq = MyExecPlanGroupSeq(total_flops, [None], [float('inf')])    


    time_before_search = time.perf_counter()

    _get_best_model_schedule_dispatcher(
        search_method_baseline,
        gen_execplans_baseline, 
        check_gap, sort_input,
        cost_table, model_sys, curr_group_seq, best_group_seq, dict(), gpu_name, tot_gpu_num, byte_per_gpu, top_k)
    
    time2 = time.perf_counter()
    print(f"Total search time: {time2 - time1}")
    print(f"Total time for preparation before search: {time_before_search - time1}")
    print(f"Best group seq: {str(best_group_seq)}")

    return best_group_seq






def test_data_parallel_improvement(
        search_method_baseline: str,
        gen_execplans_baseline: str,
        model_paths: List[str], 
        sample_config: Tuple[float, float, float, float],
        trust_remote_code:bool=True, revision:Optional[str] = None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=float('inf')):
    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    _MAX_SEQ_NUM = max_group_seq_num
    _CHECKED_SEQ_NUM = 0

    import time
    time1 = time.perf_counter()

    # 1. first initialize cost_table
    cost_table = get_cost_table()

    # 2. get input lengths
    inp_lens = get_inplens()

    out_lens_list = []
    inp_lens = sorted(inp_lens, reverse=True)

    for inp_num in [250, 500, 1000]:
        _CHECKED_SEQ_NUM = 0

        # 3.  initialize model info objects
        model_list: List[MyModelInfor] = get_model_info_objs(
            cost_table,
            data_byte, inp_lens, model_paths, sample_config, trust_remote_code, revision)

        # change the inp request info
        if out_lens_list == []:
            assert len(model_list[0].out_lens) == 1000
            for model in model_list:
                out_lens_list.append(model.out_lens)
        for model, out_lens in zip(model_list, out_lens_list):
            model.inp_lens = tuple([inp_lens[(1000//inp_num)*i] for i in range(inp_num)])
            model.out_lens = tuple([out_lens[(1000//inp_num)*i] for i in range(inp_num)])

        print(f"len(inp_lens): {len(model.inp_lens)}, len(inp_lens): {len(model.out_lens)}")

        total_flops = get_total_model_flops(model_list, cost_table)
        curr_group_seq = MyExecPlanGroupSeq(total_flops, [], [])
        best_group_seq = MyExecPlanGroupSeq(total_flops, [None], [float('inf')])    

        _get_best_model_schedule_dispatcher(
            search_method_baseline,
            gen_execplans_baseline, 
            cost_table, model_list, curr_group_seq, best_group_seq, dict(), gpu_name, tot_gpu_num, byte_per_gpu)
        
        time2 = time.perf_counter()
        print(f"Total search time: {time2 - time1}")
        print(f"Best group seq: {str(best_group_seq)}")

    return best_group_seq










def get_dependent_exec_plans_for_each_plan(
        plan_group_seq: MyExecPlanGroupSeq
        )->Tuple[List[MyExecPlan], Dict[int, List[int]]]:
    '''
        Return the dependent exec plans for each exec plan in the given plan_group_seq.
        Definition of exec plan dependency:
            (1) exec plan in stage ``i+1'' depend on exec plans in stage ``i'';
            (2) exec plans in the same stage do not depend on each other;
            (3) if a model's exec plans are the same in two consecutive stages, they are regarded as the same one, and it 
            and any other exec plan in its alive stages do not depend on each other.
    '''
    exec_plan_serial_id: Dict[MyExecPlan, int] = dict()
    depend_on: Dict[int, List[int]] = dict()
    uniq_exec_plan_seq: List[MyExecPlan] = list()
    exec_plan_id = 0
    for stage_i in range(len(plan_group_seq.plan_group_seq)):
        plan_group = plan_group_seq.plan_group_seq[stage_i]
        if stage_i == 0:
            # stage 0; depend on no exec plans
            for exec_plan in plan_group.exec_plans:
                depend_on[exec_plan_id] = list()
                exec_plan_serial_id[exec_plan] = exec_plan_id
                exec_plan_id += 1
                uniq_exec_plan_seq.append(exec_plan)
        else:
            # depend on the last stage exec plans
            last_stage_exec_plans = plan_group_seq.plan_group_seq[stage_i-1].exec_plans
            last_stage_exec_plan_id_dict = {
                (exec_plan.model, exec_plan.get_key()):exec_plan_serial_id[exec_plan]
                for exec_plan in last_stage_exec_plans
                }
            
            # the exec plan ids of the plans which are alive in stage_i-1 and stage_i
            continue_exec_plan_ids: List[int] = list()
            new_exec_plan_ids: List[int] = list()
            # check whether there are models not changing exec plans
            for exec_plan in plan_group.exec_plans:
                if (exec_plan.model, exec_plan.get_key()) in last_stage_exec_plan_id_dict:
                    # this exec plan is not a new one
                    same_exec_plan_id = last_stage_exec_plan_id_dict[(exec_plan.model, exec_plan.get_key())]
                    exec_plan_serial_id[exec_plan] = same_exec_plan_id
                    continue_exec_plan_ids.append(same_exec_plan_id)
                else:
                    # this exec plan is a new one
                    exec_plan_serial_id[exec_plan] = exec_plan_id
                    new_exec_plan_ids.append(exec_plan_id)
                    exec_plan_id += 1
                    uniq_exec_plan_seq.append(exec_plan)
            
            # update depend_on
            depend_on_last_stage_exec_plan_ids = \
                list(np.intersect1d(list(last_stage_exec_plan_id_dict.values()), continue_exec_plan_ids))
            for this_plan_id in new_exec_plan_ids:
                depend_on[this_plan_id] = depend_on_last_stage_exec_plan_ids
    

    return uniq_exec_plan_seq, depend_on






def get_arxiv_data_set_chunks(file_name: str, chunk_size: int):
    """
        Just used to check the dataset used in Parrot's experiments.
    """
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from transformers import AutoTokenizer
    # 
    loader = TextLoader(f"../workloads/arxiv-march-2023/arxiv-sampled/{file_name}.txt")
    docs = loader.load()
    # 
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    # 
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator=" ",
    )
    split_docs = text_splitter.split_documents(docs)
    # 
    return len(split_docs) # [doc.page_content for doc in split_docs]



'''
res = dict()
for i in range(10):
    res[i] = list()
    for bs in [512, 1024, 1536, 2048]:
        chunk_num = get_arxiv_data_set_chunks(f"article_{i}", bs)
        for ol in [25, 50, 75, 100]:
            res[i].append(chunk_num*ol)


            
'''




        
'''
我们的整体计算逻辑如下：
1. 在一开始假定所有model的output length都一样，找到下一个stage的最佳exec plan组合。
2. 当一个stage结束之后，根据已有的model的运行信息，对不同model的output length进行重新评估。然后再重复step 1，计算当前的最佳exec plan组合。
试试效果吧，should be OK.
'''

# 吃完饭test一下。
'''
V, h, I, L = model_param_configs['llama_7b']['V'], model_param_configs['llama_7b']['h'], model_param_configs['llama_7b']['I'], \
    model_param_configs['llama_7b']['L']
model_list = [MyModelInfor('llama_7b', 7, 32, 
                           cal_each_layer_param_bytes(V, h, I, L, v_byte=2), 
                           cal_extra_param_bytes(V, h, v_byte=2)) for _ in range(2)]

V, h, I, L = model_param_configs['llama_70b']['V'], model_param_configs['llama_70b']['h'], model_param_configs['llama_70b']['I'], \
    model_param_configs['llama_70b']['L']
model_list = model_list + [MyModelInfor('llama_70b', 70, 80, 
                           cal_each_layer_param_bytes(V, h, I, L, v_byte=2), 
                           cal_extra_param_bytes(V, h, v_byte=2))]



best_group_seq = get_best_model_schedule(model_list, gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3))
'''



if __name__ == "__main__":
    # from search_exec_plans import *
    print("start")
    model_paths = ['NousResearch/Llama-2-7b-hf', 
                       'NousResearch/Llama-2-13b-hf',
                       'NousResearch/Llama-2-70b-hf']
    model_paths = [ 'NousResearch/Llama-2-7b-hf'] *5 + \
                   ['NousResearch/Llama-2-70b-hf']
    model_paths = [
                    'NousResearch/Llama-2-7b-hf', 
                   'NousResearch/Llama-2-7b-chat-hf',
                #    'NousResearch/Llama-2-13b-hf',
                #    'NousResearch/Llama-2-70b-hf',
                #    'THUDM/chatglm3-6b',
                #    'EleutherAI/gpt-j-6b', 
                #    'EleutherAI/gpt-neox-20b',
                #    'baichuan-inc/Baichuan2-13B-Chat',
                #    'baichuan-inc/Baichuan-7B',
                #    'mistralai/Mixtral-8x7B-v0.1'
                ]
    out_edge_dict = {}

    # test models from LLM-blender
    model_paths = [
                    'THUDM/chatglm3-6b',
                    'lmsys/vicuna-13b-v1.5',
                    'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
                    'chavinlo/alpaca-13b',
                    'project-baize/baize-v2-13b',
                    'TheBloke/koala-13B-HF',
                    'databricks/dolly-v2-12b',
                    'mosaicml/mpt-7b-chat',
                ]


    # inp/out len generator functions for the general setting
    # req_num = 1000
    # inp_generator = get_inplens
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
    # outlen_generator = output_length_sampler.sample_out_len_for_given_model
    # inp_seq_ids_dict = defaultdict(list)
    # inp_seq_ids_dict.update({i:list(range(req_num)) for i in range(len(model_paths))})


    # inp/out len generator functions for the map-reduce or chain summary scenario
    # map-reduce
    # 如果input sequence的长度差别很大的话，可能会导致chunk的数量差别很大，可能会有很多个LLM instance，但是每个instance的inp workload数量不一样，这样会有影响吗？
    # 试试再说吧
    # TODO: 还有一个问题，我们现在判断redundancy不会把相同的model但是不同instance看成是相同的，这样可能会导致大量redundancy。
    # req_num = 10
    # chunk_size = 512
    # model_paths = ['NousResearch/Llama-2-13b-hf'] * (20000 // chunk_size)
    # model_paths = model_paths + ['NousResearch/Llama-2-13b-hf']
    # out_edge_dict = {i:[len(model_paths)-1] for i in range(len(model_paths)-1)}
    # # 
    # inp_generator = lambda req_num: [chunk_size]*req_num
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
    # outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
    # inp_seq_ids_dict = defaultdict(list)
    # inp_lens = np.asarray(inp_generator(req_num))
    # inp_seq_ids_dict.update({i:list(range(sum(inp_lens>chunk_size*(i-1)))) for i in range(len(model_paths)-1)})
    # inp_seq_ids_dict.update({len(model_paths)-1:list(range(req_num))})



    # # chain summary
    req_num = 100
    chunk_size = 512
    max_length = chunk_size*5 # 20000
    model_paths = ['NousResearch/Llama-2-13b-hf'] * (max_length // chunk_size)
    print(f"model_paths: {model_paths}")
    # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
    out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
    inp_generator = lambda req_num: [chunk_size]*req_num
    inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
    outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
    inp_seq_ids_dict = defaultdict(list)
    # inp_lens = np.asarray(inp_generator(req_num))
    inp_lens = np.asarray([chunk_size]*int(0.2*req_num)+[2*chunk_size]*int(0.2*req_num)\
                          +[3*chunk_size]*int(0.2*req_num)+[4*chunk_size]*int(0.2*req_num)\
                            +[5*chunk_size]*int(0.2*req_num))
    inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*i)))) for i in range(len(model_paths))})
    print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")
    # add another model after the chain summary
    model_paths.append('NousResearch/Llama-2-7b-hf')
    out_edge_dict[3].append(5)
    out_edge_dict[4] = [5]
    inp_seq_ids_dict[5] = sorted(set(inp_seq_ids_dict[3] + inp_seq_ids_dict[4]))

    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreal out_edge_dict: {out_edge_dict}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")




    # # chain summary
    req_num = 100
    chunk_size = 512
    max_length = chunk_size*50 # 20000
    model_paths = ['NousResearch/Llama-2-13b-hf'] * (max_length // chunk_size)
    print(f"model_paths: {model_paths}")
    # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
    out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
    inp_generator = lambda req_num: [chunk_size]*req_num
    inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
    outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
    inp_seq_ids_dict = defaultdict(list)
    # inp_lens = np.asarray(inp_generator(req_num))
    inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))
    inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*i)))) for i in range(len(model_paths))})
    print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")

    # add another chain in parallel with the first chain
    first_chain_len = len(model_paths)
    model_paths.extend(['NousResearch/Llama-2-7b-hf'] * (max_length // chunk_size))
    print(f"model_paths: {model_paths}")
    # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
    out_edge_dict.update({i:[i+1] for i in range(first_chain_len, len(model_paths)-1)})
    # inp_generator = lambda req_num: [chunk_size]*req_num
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
    # outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
    # inp_seq_ids_dict = defaultdict(list)
    # inp_lens = np.asarray(inp_generator(req_num))
    inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))
    inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*(i-first_chain_len))))) for i in range(first_chain_len,len(model_paths))})
    print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")

    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreal out_edge_dict: {out_edge_dict}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")

    out_req_id_mapping = dict()









    # # # 一组简单的测试
    # req_num = 20
    # chunk_size = 512
    # # max_length = chunk_size*5 # 20000
    # model_paths = ['NousResearch/Llama-2-13b-hf']
    # print(f"model_paths: {model_paths}")
    # # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
    # # out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
    # out_edge_dict = {}
    # inp_generator = lambda req_num: [chunk_size*5]*req_num
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
    # outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
    # inp_seq_ids_dict = defaultdict(list)
    # # inp_lens = np.asarray(inp_generator(req_num))
    # inp_lens = np.asarray([2*chunk_size]*req_num)
    # inp_seq_ids_dict.update({i:list(range(req_num)) for i in range(len(model_paths))})
    # print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")


    # print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")

    # print(f"\nreal model_paths: {model_paths}")
    # print(f"\nreal out_edge_dict: {out_edge_dict}")
    # print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")










    # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # search_method_baseline = 'ours' # 'naive'  'ours'
    gen_execplans_baseline = 'ours' # 'naive'  'ours'
    search_method_baseline = 'ours' # 'naive'  'ours'
    check_gap = 16
    sort_input = True
    best_group_seq = get_best_model_schedule(
        search_method_baseline,
        gen_execplans_baseline,
        check_gap,
        sort_input,
        model_paths, 
        num_prompts=req_num, inp_seq_ids_dict=inp_seq_ids_dict, 
        out_req_id_mapping=out_req_id_mapping,
        inp_generator=inp_generator, inp_merger=inp_merger, outlen_generator=outlen_generator,
        out_edge_dict=out_edge_dict,
        sample_config=(1, 1, -1, 0),
        trust_remote_code=True, revision=None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=100, # float('inf'),
        top_k=100, 
        similar_threshold=0.1
    )


    # best_group_seq = test_data_parallel_improvement(
    #     search_method_baseline,
    #     gen_execplans_baseline,
    #     model_paths, 
    #     sample_config=(1, 1, -1, 0),
    #     trust_remote_code=True, revision=None,
    #     gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
    #     data_byte=2,
    #     max_group_seq_num=float('inf'),
    # )



'''
from search_exec_plans import *
model_paths = [ 'NousResearch/Llama-2-7b-hf', 
                   'NousResearch/Llama-2-13b-hf',
                   # 'NousResearch/Llama-2-70b-hf'
                   ]
model_paths = [ 'NousResearch/Llama-2-7b-hf'] *2 + \
                   ['NousResearch/Llama-2-70b-hf']
model_paths = [
                'NousResearch/Llama-2-7b-hf', 
                'NousResearch/Llama-2-7b-chat-hf',
            #    'NousResearch/Llama-2-13b-hf',
            #    'NousResearch/Llama-2-70b-hf',
                'THUDM/chatglm3-6b',
                'EleutherAI/gpt-j-6b', 
            #    'EleutherAI/gpt-neox-20b',
            #    'baichuan-inc/Baichuan2-13B-Chat',
                'baichuan-inc/Baichuan-7B',
            #    'mistralai/Mixtral-8x7B-v0.1'
            ]
                   
# gen_execplans_baseline = 'ours' # 'naive'  'ours'
# search_method_baseline = 'ours' # 'naive'  'ours'
gen_execplans_baseline = 'ours' # 'naive'  'ours'
search_method_baseline = 'naive' # 'naive'  'ours'
best_group_seq = get_best_model_schedule(
    search_method_baseline,
    gen_execplans_baseline,
    model_paths, 
    sample_config=(1, 1, -1, 0),
    trust_remote_code=True, revision=None,
    gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
    data_byte=2
)


'''