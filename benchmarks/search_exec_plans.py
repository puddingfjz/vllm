"""
This file contains the search method to find the best exec plans
for the given set of models and the given set of requests.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import itertools
import fake_scheduling
from my_per_iter_latency_estimator import CostTable, get_cost_table
import output_length_sampler

from vllm.transformers_utils.config import get_config
from vllm.worker.cache_engine import CacheEngine
from collections import defaultdict

from model_size_database import model_sizes


_ENGINE_ARGS_LIST = dict()
_FAKE_SCHEDULING_RES = dict() # key: (model name, exec_plan, inp_lens, out_lens) value: the output of fake_scheduling
_MAX_SEQ_NUM = 0
_CHECKED_SEQ_NUM = 0

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

        # model-level pipeline
        self.ori_tot_inp_num: int = len(inp_lens) # this value will not change during the search
        self.inp_seq_ids = np.asarray(range(self.ori_tot_inp_num))

    # def set_input_model_ids(self, input_model_ids: List[int]):
    #     self.input_model_ids = input_model_ids

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
    


    def get_finish_times_merged(self, seq_ids: List[int]):

        # print(f"in get_finish_times_merged: exec_plan: {str(self)}")
        # print(f"self.dp_inp_seq_ids_list: {self.dp_inp_seq_ids_list}")

        if len(self.finish_times_merged) == 0:
            self.finish_times_merged = np.asarray([-1-self.extra_cost]*self.model.ori_tot_inp_num) 
            for dp_inp_seq_ids, finish_times in zip(self.dp_inp_seq_ids_list, self.finish_times_list):
                self.finish_times_merged[dp_inp_seq_ids] = finish_times
        return self.finish_times_merged[seq_ids]



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

        inp_lens, out_lens = self.model.get_inp_out_seqlens()
        
        # model-level pipeline
        # SORT the input seqs by their arrival times and their seq ids
        inp_seq_ids = self.model.get_inp_seq_ids()
        # consider the extra cost in arrive times
        arrive_times = np.asarray(arrive_times) - self.extra_cost
        to_sort = list(zip(arrive_times, inp_seq_ids))
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
        

        # print(f"inp lens: {inp_lens}")
        # print(f"out lens: {out_lens}")

        # 2. do fake scheduling
        # support model-level pipeline: we add arrive_times to the key
        # key = (self.model.model_name, self.get_key(), self.model.get_inp_out_seqlens(), tuple(arrive_times))
        # NOTE: input ``arrive_times`` may not be sorted, so we sort it
        key = (self.model.model_name, self.get_key(), (tuple(inp_lens), tuple(out_lens)), tuple(arrive_times))
        if key in _FAKE_SCHEDULING_RES:
            (self.cumsum_latencys_list, self.cum_rng_nums_list, self.rng_starts_list, self.rng_ends_list,
                self.is_prefill_steps_list, self.finish_times_list) = _FAKE_SCHEDULING_RES[key]
            
            self.total_latency_list = [cumsum_latencys[-1] if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]

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

        # print(f"exec_plan: {str(self)}")
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
            self.estimate_exec_time(cost_table, 
                check_gap=1, sort_input=sort_input, arrive_times=list())

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

            # 0. check whether we have run this function for the corresponding stop iter.
            stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')
            if stop_iter_i in cache_stop_time_info:

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
            
            # dp_seq_ids = self.dp_seq_ids_list[dp_id]
            # finish_times_merged[dp_seq_ids[alive_old_indices]] = finish_times_of_alive_seqs
            finish_times_list[dp_id] = finish_times_of_alive_seqs
            assert (new_inp_out_lens_list[dp_id][2] == alive_old_indices).all()


        new_key = (self.model.model_name, self.get_key(), (tuple(new_inp_lens_merged), tuple(new_out_lens_merged)), \
                   tuple())
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
        self.tmp_inp_out_lens_list: List[Tuple[List[int], List[int]]] = list()
        self.tmp_remaining_decode_flops_after_infer_stage: List[float] = list()
        self.tmp_stop_iter_i_list: List[int] = list()
        self.compute_infer_stage_data(cost_table=cost_table, 
                                      last_stage_exec_plans=last_stage_exec_plans, 
                                      check_gap=check_gap, sort_input=sort_input)





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
        model_exec_plan_mapping: Dict[int, MyExecPlan] = \
            {exec_plan.model.model_id:exec_plan for exec_plan in self.exec_plans}
        
        all_model_ids = set(model_exec_plan_mapping.keys())

        inp_model_ids_dict: Dict[MyExecPlan, List[int]] = defaultdict(list)

        # get self.inp_exec_plan_dict
        for exec_plan in self.exec_plans:
            inp_model_ids_this_stage = set.intersection(set(exec_plan.model.input_model_ids), all_model_ids)
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




    def _get_arrive_times(self, exec_plan: MyExecPlan):
        """
            Compute the input arrive times for this exec plan.
            NOTE: the finish_times are in the order of model.inp_seq_ids. [感觉这个影响不会太大？]
        """
        inp_seq_ids: List[int] = exec_plan.model.get_inp_seq_ids()
        inp_exec_plans = self.inp_exec_plan_dict[exec_plan]

        # print(f"inp_exec_plans: {[str(exec_plan) for exec_plan in inp_exec_plans]}")

        if len(inp_exec_plans) > 0:
            finish_times = np.asarray([inp_plan.get_finish_times_merged(inp_seq_ids) + inp_plan.extra_cost \
                                    for inp_plan in inp_exec_plans])
            finish_times = np.max(finish_times, axis=0)
        else:
            # no input exec plan, i.e., all inputs are available at the beginning
            finish_times = np.asarray([-1]*len(inp_seq_ids))
        return finish_times




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

        print(f"INIT plan group: {[str(exec_plan) for exec_plan in self.exec_plans]}")

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
        return tuple(sorted([(
            exec_plan.model.model_name, exec_plan.model.model_id, 
            tuple(np.concatenate([inps for inps, outs, indices in inp_out_lens])), 
            tuple(np.concatenate([outs for inps, outs, indices in inp_out_lens])) 
            ) for exec_plan, inp_out_lens \
                in zip(self.exec_plans, self.tmp_inp_out_lens_list)]))
    
    def get_model_states_before_infer_stage(self):
        return tuple(sorted([(exec_plan.model.get_state()) for exec_plan in self.exec_plans]))
    

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
        for exec_plan, inp_out_lens, stop_iter_i in zip(self.exec_plans, self.tmp_inp_out_lens_list, self.tmp_stop_iter_i_list):
            # old_inp_lens, _ = exec_plan.model.get_inp_out_seqlens()
            old_inp_lens = exec_plan.dp_inp_lens_list

            # we sort the inps by their seq ids
            # merged_inp_out_lens = merge_inp_out_lens_of_data_parallel_workers(inp_out_lens)
            merged_inp_out_lens = exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)
            
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
        cost_table: CostTable, inp_merger, outlen_generator
    ) -> None:
        # here we have model_list[i].model_id = i
        self.model_list = model_list
        # model-level pipeline
        self.all_level_model_ids: List[List[int]] = list()
        # here we use edge_dict[i] to store the output edges of model i
        self.out_edge_dict = defaultdict(list, out_edge_dict)
        # # generate inp_edge_dict for use
        # self.inp_edge_dict = defaultdict(list)
        # for inp in out_edge_dict:
        #     for out in out_edge_dict[inp]:
        #         self.inp_edge_dict[out].append(inp)

        
        # set the inp_edges information for each LLM
        for inp in out_edge_dict:
            for out in out_edge_dict[inp]:
                self.model_list[out].input_model_ids.append(inp)

        print(f"out_edge_dict: {out_edge_dict}")
        for model in self.model_list:
            print(f"inp model ids of model {model.model_id}: {model.input_model_ids}")
        print("original inp/out lens of each model: ")
        for model in self.model_list:
            print(f"{model.model_id}")
            print(f"{model.get_inp_out_seqlens()[0]}")
            print(f"{model.get_inp_out_seqlens()[1]}")

        # correct inp/out lens of models considering LLM dependency
        # TODO: 这个地方可以优化一下，一步到位生成所有模型的正确的inp/out lens
        self._get_inp_out_lens_considering_LLM_dependency(
            cost_table=cost_table, inp_merger=inp_merger, outlen_generator=outlen_generator)
        
        print("correct inp/out lens of each model: ")
        for model in self.model_list:
            print(f"{model.model_id}")
            print(f"{model.get_inp_out_seqlens()[0]}")
            print(f"{model.get_inp_out_seqlens()[1]}")



    def _get_inp_out_lens_considering_LLM_dependency(
            self, 
            cost_table: CostTable,
            inp_merger, outlen_generator):
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
        # sort the models by the topological order first
        self.get_all_level_models()
        for model_ids in self.all_level_model_ids:
            for model_id in model_ids:
                model = self.model_list[model_id]
                if len(model.input_model_ids) == 0:
                    # we do not need to change the inp seq lengths of this model
                    continue
                ori_inp_lens = model.get_inp_out_seqlens()[0]
                new_inp_lens = inp_merger(
                    [ori_inp_lens] + \
                        [self.model_list[inp_model_id].get_inp_out_seqlens()[1] for inp_model_id in model.input_model_ids]
                        )
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
        for model in self.model_list:
            if is_finished_or_running(model, running_model_ids):
                continue

            # inps = self.inp_edge_dict[model.model_id]
            inps = model.input_model_ids
            inps_status = [is_finished_or_running(self.model_list[inp], running_model_ids) for inp in inps]
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
        return [self.model_list[model_id] for model_id in self.all_level_model_ids[level_num]]



    def get_runnable_plans_from_cand_plans(
            self, 
            running_plan_group: List[MyExecPlan], 
            cand_models: List[MyModelInfor], cand_exec_plans: List[List[MyExecPlan]]):
        
        def is_finished_or_running(model: MyModelInfor, running_model_ids: List[int]):
            return (model.is_finished()) or \
                (model.model_id in running_model_ids)

        runnable_exec_plans: List[List[MyExecPlan]] = list()
        running_model_ids: List[int] = [exec_plan.model.model_id for exec_plan in running_plan_group]
        for model, exec_plans in zip(cand_models, cand_exec_plans):
            inps = model.input_model_ids
            inps_status = [is_finished_or_running(self.model_list[inp], running_model_ids) for inp in inps]
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
        visited: List[bool] = np.asarray([False] * len(self.model_list))

        while sum(visited) < len(self.model_list):
            # get the model ids on this level
            newly_visited: List[bool] = np.asarray([False] * len(self.model_list))
            new_model_ids: List[int] = list()
            for model in self.model_list:
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
            visited = visited + newly_visited
            if len(new_model_ids) > 0:
                self.all_level_model_ids.append(new_model_ids)





    def get_model_num(self)->int:
        return len(self.model_list)




    def get_not_finished_model_num(self) -> int:
        not_finished_model_num = sum([not model.is_finished() for model in self.model_list])
        return not_finished_model_num


    def is_finished(self) -> bool:
        """
            Return True if all the models in the system is finished.
        """
        # return False not in [model.is_finished() for model in self.model_list]
        return self.get_not_finished_model_num() == 0



    def get_model_states(self):
        '''
            Get the current inference progress of the given list of models.
            NOTE: the returned progress should be able to be added to a set.
        '''
        return tuple(sorted([model.get_state() for model in self.model_list]))
    
    
    def get_model_inp_out_lens(self):
        ori_inp_out_lens_list = [model.get_inp_out_seqlens() for model in self.model_list]
        return ori_inp_out_lens_list
    
    def get_model_remaining_decode_flops(self):
        ori_remaining_decode_flops_list = [model.get_remaining_flops() for model in self.model_list]
        return ori_remaining_decode_flops_list

    def get_model_inp_seq_ids(self):
        ori_inp_seq_ids_list = [model.get_inp_seq_ids() for model in self.model_list]
        return ori_inp_seq_ids_list


    def recover_model_state(
            self,
            inp_seq_ids_list: List[List[int]],
            inp_out_lens_list: List[Tuple[List[int], List[int]]], 
            cost_table: CostTable, remaining_decode_flops_list: List[float]):
        # for model, inp_out_lens in zip(model_list, inp_out_lens_list):
        # for i in range(len(self.model_list)):
        #     model = self.model_list[i]
        #     inp_out_lens = inp_out_lens_list[i]
        #     remaining_decode_flops = remaining_decode_flops_list[i]
        #     model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)
        for model, inp_seq_ids, inp_out_lens, remaining_decode_flops in \
            zip(self.model_list,inp_seq_ids_list,inp_out_lens_list,remaining_decode_flops_list):
            model.update_inp_out_seqlens(*inp_out_lens, inp_seq_ids, cost_table, remaining_decode_flops)


    def print_model_list(self):
        print(f"model_list: {[str(model) for model in self.model_list]}")




    def get_candidate_plan_groups(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3))->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
        """
        # running_model_ids are models running in this exec stage
        tot_plan_groups = [[]]
        new_plan_groups = [[]]

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
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline)
                exec_plans_list.append(exec_plans)
                # print(f"can exec_plans: {[str(plan) for plan in exec_plans]}")

            # print(f"New Round get_candidate_plan_groups ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            for cand_plan_group in new_plan_groups:
                # running_model_ids = [exec_plan.model.model_id for exec_plan in cand_plan_group]
                # cand_models = self.get_next_level_models(last_level_running_model_ids=, all_running_model_ids=running_model_ids)

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)

                
                # 2. second combine exec plans for different models into a group
                plan_groups = [cand_plan_group]
                _append_exec_plan(plan_groups, runnable_exec_plans_list, 0, tot_gpu_num, byte_per_gpu)
                # plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans) \
                #             for plan_group in plan_groups if len(plan_group) > 0]
                
                if len(plan_groups) == 1:
                    # no new model is added to cand_plan_group
                    tot_plan_groups.extend(plan_groups)
                else:
                    tmp_new_plan_groups.extend(plan_groups[1:])

                # print(f"tot_plan_groups: {[[str(plan) for plan in plan_group] for plan_group in tot_plan_groups]}")
                # print(f"tmp_new_plan_groups: {[[str(plan) for plan in plan_group] for plan_group in tmp_new_plan_groups]}")
                # print(f"cand_plan_group: {[str(plan) for plan in cand_plan_group]}")
                # print(f"tot_plan_groups: ----------------------")
                # for plan_group in tot_plan_groups:
                #     print(f"{len(plan_group)}, {[str(plan) for plan in plan_group]}")
                # print(f"tmp_new_plan_groups: ------------------")
                # for plan_group in tmp_new_plan_groups:
                #     print(f"{len(plan_group)}, {[str(plan) for plan in plan_group]}")


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


        print(f"in get_candidate_plan_groups: the plan groups we generated: ")
        for plan_group in tot_plan_groups:
            print(f"{len(plan_group)}, {[str(plan) for plan in plan_group]}")



        # convert plan_groups to MyExecPlanGroup objects
        plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                        check_gap=check_gap, sort_input=sort_input,) \
                    for plan_group in tot_plan_groups if len(plan_group) > 0]
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
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline)
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
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline)
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
        remaining_model_ids = [model.model_id for model in self.model_list if not model.is_finished()]
        output_model_num = sum([len(self.out_edge_dict[model_id]) for model_id in remaining_model_ids])
        return (output_model_num == 0)






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
                            
                            # do not need to call is_valid_exec_plan for dp_size
                            exec_plans.append(exec_plan)

    return exec_plans






# NOTE: we set dp_size to 1 here, but maybe we should select the best exec plan for the model?
# ==> such function is implemented in "_get_possible_exec_plans_naive_baseline_2()"
def _get_possible_exec_plans_naive_baseline(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable):
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
    
    print(f"gen an exec plan: {str(exec_plan)}")

    # check whether exec_plan is valid
    if is_valid_exec_plan(exec_plan, cost_table):
        exec_plans.append(exec_plan)


    return exec_plans





def _get_possible_exec_plans_naive_baseline_2(
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
        for wld_degree in get_factors(model.layer_num, 2, model.layer_num):
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
                    
                    # print(f"gen an exec plan: {str(exec_plan)}")

                    # check whether exec_plan is valid
                    if is_valid_exec_plan(exec_plan, cost_table):
                        exec_plans.append(exec_plan)

                        # print(f"valid")

                        # support data parallel
                        for dp_size in range(1, tot_gpu_num // num_worker + 1):
                            if dp_size * num_worker + cache_gpu_num > tot_gpu_num:
                                # each dp worker occupies num_worker GPUs for computation
                                # all dp workers can share cache_gpu_num GPUs for cache (but it is not necessary)
                                continue

                            # add exec_plan with dp_size
                            exec_plan = MyExecPlan(model,
                                num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                                param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                                gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                            
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
        baseline: str):
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
        return _get_possible_exec_plans_naive_baseline(model, tot_gpu_num, byte_per_gpu, cost_table)




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





def _append_exec_plan(plan_groups, exec_plans_list, depth_i, tot_gpu_num, byte_per_gpu):
    '''
    Get all the possible exec plans with depth-first search.
    The initial plan_groups is [[]], i.e., containing a group with no exec plan.
    All plan groups are valid if they are put into plan_groups and returned.
    '''
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
            # print(f"plan group: {[str(_) for _ in tmp_plan_group]}")
            if is_valid_exec_plan_combination(tmp_plan_group, tot_gpu_num, byte_per_gpu):
                new_plan_groups.append(tmp_plan_group)

                # print(f"valid")

    plan_groups.extend(new_plan_groups)
    _append_exec_plan(plan_groups, exec_plans_list, depth_i+1, tot_gpu_num, byte_per_gpu)





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
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
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
        gen_execplans_baseline, check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num, byte_per_gpu)
    

    print(f"in get_one_stage_exec_plans_sorted: the plan groups we generated: ")
    for plan_group in plan_groups:
        print(f"{len(plan_group)}, {str(plan_group)}")




    # 2. delete plan_groups which do not occupy all comp gpus when there are models not executed
    # also for cases where there are idle comp resources, check whether increase comp resources can improve throughput, if yes, delete it.
    # TODO (jingzhi) it seems not_finished_model_num is not used
    # not_finished_model_num = sum([not model.is_finished() for model in model_list])
    not_finished_model_num = model_sys.get_not_finished_model_num()
    useful_plan_groups = list()
    idle_comp_plan_groups = dict() # {models executed: (best_throughput, best_plan_group)}
    # comp_throughput_given_plan_groups(plan_groups, gpu_name)

    # 2.1 first delete inefficient plan groups for each possible set of involved models
    for plan_group in plan_groups:
        
        # print(f"check redundancy 1 of: {str(plan_group)}")

        key = plan_group.get_model_states_before_infer_stage()
        if key not in idle_comp_plan_groups:
            # print(f"new key (before infer stage): {key}")
            idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
        else:
            # print(f"old key (before infer stage): {key}")
            if plan_group.get_throughput() > idle_comp_plan_groups[key][0]:
                idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)

    plan_groups: List[MyExecPlanGroup] = [plan_group for _, plan_group in idle_comp_plan_groups.values()]
    idle_comp_plan_groups = dict() # {models executed: (best_throughput, best_plan_group)}

    # 2.2 then delete inefficient plan groups for each possible set of model states after this infer stage
    for plan_group in plan_groups:

        # print(f"check redundancy 2 of: {str(plan_group)}")


        # 1. if there are models available but the comp gpus are not fully utilized
        if (not_finished_model_num > len(plan_group)) and (get_tot_worker_num(plan_group.exec_plans)<tot_gpu_num):
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
        print(str(plan_group))


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
    # NOTE: after introducing data parallel, in each stage maybe only a part of a model will finish
    # 但是感觉这样的话，整个scheduling的过程会变成砍一刀砍一刀，被砍得很细，效率很低，其实，虽然会自动重新均匀分配dp workload
    # solution 1：就这么做；2：一个model的运行时间按照时间最长的dp worker来算，在此基础上估计plan group的运行时间。
    # 先试试solution1 ==> solution1 搜索空间太大了，换成solution2。
    # stop condition of the depth-first search: there are <= len(model_list) exec stages 
    # since in each stage >= 1 models will finish
    if len(curr_group_seq.plan_group_seq) > model_sys.get_model_num():
        # assert False, f'{[model.left_flops_per_token for model in model_list]},'\
        #             f'{[[str(_) for _ in group] for group in curr_group_seq]}'
        assert False, f'{[[str(_) for _ in group] for group in curr_group_seq]}'


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
    model_states = model_sys.get_model_states()


    if model_states in uniq_model_states:
        # do not need to check this plan group, as we have check an equivalent one
        # print(f"redundant plan group due to model states: {model_states, str(curr_plan_seq)}")
        # return

        # only when the latency of the current group seq is higher, we do not check it further
        if curr_group_seq.get_tot_time() >= uniq_model_states[model_states]:

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
    plan_groups: List[MyExecPlanGroup] = get_one_stage_exec_plans_sorted(
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

    
    # 3.1 pruning rule: if using the highest throughput in plan_groups still cannot beat best_group_seq, skip all of them
    # if (get_model_left_decode_flops(model_list, cost_table) / plan_groups[0].get_comp_throughput_only() + curr_group_seq.get_tot_time()) \
    #     >= best_group_seq.get_tot_time():
    # TODO:这个地方的early prune逻辑应该改成什么？如果该system可以有唯一地划分成两部分的划分方式，那么就可以把各个plan group
    # 按照各个阶段来对比throughput。最general的写法应该是这样的，但是感觉好复杂啊。可以按照一个model距离起始点的所有路径长度是否一致
    # 来对原始的system进行阶段分割。但是这样好像也没用，还是没法做原来的early pruning。还是简单对进入到最后一个阶段的plan group seq 
    # 进行early pruning吧。。。。。搜索算法之后可以想办法再优化。
    if (model_sys.remaining_models_are_on_the_last_layer()) \
        and ((sum(ori_remaining_decode_flops_list) / plan_groups[0].get_comp_throughput_only() \
              + curr_group_seq.get_tot_time()) >= best_group_seq.get_tot_time()):
        
        print(f"using the highest throughput in plan_groups still cannot beat best_group_seq")
        
        return


    print(f"finish step 4")


    # 4. try each candidate plan group and do depth-first search.
    for plan_group in plan_groups:
        # print(f"plan_group: {plan_group}")
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

        # update the remaining workload of all models after this stage
        # recover_model_state(model_list, ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        model_sys.recover_model_state(
            ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        # comp_time = update_model_state(plan_group, gpu_name)
        plan_group.update_model_inp_out_lens(cost_table)
        curr_group_seq.append_plan_group(plan_group)
        curr_group_seq.append_exec_time(plan_group.get_infer_stage_latency())
        _get_best_model_schedule(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table,
            model_sys, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu)
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
        assert False, f'{[[str(_) for _ in group] for group in curr_group_seq]}'


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
    model_states = model_sys.get_model_states()


    if model_states in uniq_model_states:
        # do not need to check this plan group, as we have check an equivalent one
        # print(f"redundant plan group due to model states: {model_states, str(curr_plan_seq)}")
        # return

        # only when the latency of the current group seq is higher, we do not check it further
        if curr_group_seq.get_tot_time() >= uniq_model_states[model_states]:

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

        # update the remaining workload of all models after this stage
        # recover_model_state(model_list, ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        model_sys.recover_model_state(
            ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list)
        # comp_time = update_model_state(plan_group, gpu_name)
        plan_group.update_model_inp_out_lens(cost_table)
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
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
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
            gpu_name, tot_gpu_num, byte_per_gpu)
    else:
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
        outlen_generator,
        # edge_dict: Dict[int, List[int]],
        sample_config: Tuple[float, float, float, float], 
        trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Get the list of MyModelInfor objects for the given model paths.
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
                # input_model_ids=edge_dict[model_id],
            ) for model_id, model_path in enumerate(model_paths)]





def get_inplens():
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
    return inps[:]


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
        inp_generator, inp_merger, outlen_generator,
        # 
        out_edge_dict: Dict[int, List[int]],
        sample_config: Tuple[float, float, float, float],
        trust_remote_code:bool=True, revision:Optional[str] = None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=float('inf')):
    """
        NOTE: ``inp_generator``, ``inp_merger``, ``outlen_generator`` are 3 functions about model inp/out lens.
    """

    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    _MAX_SEQ_NUM = max_group_seq_num
    _CHECKED_SEQ_NUM = 0

    import time
    time1 = time.perf_counter()

    # 1. first initialize cost_table
    cost_table = get_cost_table()

    # 2. get input lengths
    # inp_lens = get_inplens()
    inp_lens = inp_generator()
    print(f"len(inp_lens): {len(inp_lens)}")

    # 3.  initialize model info objects and the model system object
    model_list: List[MyModelInfor] = get_model_info_objs(
        cost_table,
        data_byte, inp_lens, model_paths, outlen_generator, sample_config, trust_remote_code, revision)
    
    model_sys = MyModelSystem(model_list=model_list, out_edge_dict=out_edge_dict, 
                              cost_table=cost_table, inp_merger=inp_merger, outlen_generator=outlen_generator)

    total_flops = get_total_model_flops(model_list, cost_table)
    curr_group_seq = MyExecPlanGroupSeq(total_flops, [], [])
    best_group_seq = MyExecPlanGroupSeq(total_flops, [None], [float('inf')])    

    _get_best_model_schedule_dispatcher(
        search_method_baseline,
        gen_execplans_baseline, 
        check_gap, sort_input,
        cost_table, model_sys, curr_group_seq, best_group_seq, dict(), gpu_name, tot_gpu_num, byte_per_gpu)
    
    time2 = time.perf_counter()
    print(f"Total search time: {time2 - time1}")
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
    # inp_generator = get_inplens
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
    # outlen_generator = output_length_sampler.sample_out_len_for_given_model

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
    # inp_generator = lambda : [512]*req_num
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
    # outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))


    # # chain summary
    req_num = 1000
    chunk_size = 512
    model_paths = ['NousResearch/Llama-2-13b-hf'] * (20000 // chunk_size)
    out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
    # out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
    inp_generator = lambda : [512]*req_num
    inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
    outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))



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
        inp_generator=inp_generator, inp_merger=inp_merger, outlen_generator=outlen_generator,
        out_edge_dict=out_edge_dict,
        sample_config=(1, 1, -1, 0),
        trust_remote_code=True, revision=None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=100, # float('inf'),
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