"""
This file contains functions to schedule multiple models together on a given set of GPUs on the same machine. 
The original methods are from ``my_bench_multimodel_throughput.py''.
"""


'''
Basic idea: we use asyncio with process pool + shared variable among process.

When there is a model finished, the main process here will determine the new best execution plan for the remaining
models and notify them through the shared variables.

In each model inference process, it will check the shared varibles to see whether it need to change its execution plan.


from my_bench_multimodel_throughput import *


/ssddata/jingzhi/Nsight_Systems_2023_2_1/target-linux-x64/nsys profile -w true -t cuda,nvtx,osrt -s cpu  --cudabacktrace=true -x true -o ./nsys_profile/my_profile1 python3 my_bench_multimodel_throughput.py > ours_multimodel_0313_13b70b_100req_DEBUG_fast_soft_1.log

python3 my_bench_multimodel_throughput.py > ours_multimodel_0313_13b70b_100req_DEBUG_fast_soft_1.log
'''




from concurrent.futures import ProcessPoolExecutor
import asyncio
from multiprocessing import Array, Event


from vllm.core.multimodel_scheduler import SHARED_CONTECT, LLM_COMMUNICATOR, MyManager
import benchmark_throughput

import time
import numpy as np
from typing import List, Optional, Tuple, Dict
import itertools

from search_exec_plans import MyExecPlan, MyExecPlanGroupSeq, get_best_model_schedule, get_dependent_exec_plans_for_each_plan, get_inplens
import output_length_sampler

from collections import defaultdict

# shared_counter: Array # = Array('d', [-1, -1])

import traceback


class MyExecPlanState:
    """Record the state of an exec plan"""
    def __init__(self, 
        exec_plan: MyExecPlan, 
        launched: bool,
        stage_i: int,
        last_exec_plan_for_the_model: bool,
        need_prepare_infer_env: bool,
    ) -> None:
        self.exec_plan = exec_plan
        self.stage_i = stage_i
        self.last_exec_plan_for_the_model = last_exec_plan_for_the_model
        self.need_prepare_infer_env = need_prepare_infer_env
        
        # parameters below will be set once
        self.comp_gpus: List[int] = list()

        # parameters below will be changed once from False to True
        self.launched = launched
    
    def set_comp_gpus(self, comp_gpus: List[int]):
        self.comp_gpus = comp_gpus
    def get_comp_gpus(self):
        return self.comp_gpus

    def __str__(self) ->str:
        return f'{str(self.exec_plan)}, launched:{self.launched}, stage_i:{self.stage_i}'





# define the args we need
class InferenceArgs:
    """Arguments for vLLM single model inference."""
    def __init__(self, 
        model:str="huggyllama/llama-7b", 
        num_prompts: int = 1000
        # tensor_parallel_size:int=1
    ) -> None:
        self.backend: str = "vllm"
        self.dataset: str = "ShareGPT_V3_unfiltered_cleaned_split.json"
        self.input_len: int = None
        self.output_len: int = None
        self.model: str = model
        self.tokenizer: str = None
        self.quantization = None
        self.tensor_parallel_size: int = 1
        self.n: int = 1
        self.use_beam_search: bool = False
        # TODO: setting num_prompts
        self.num_prompts: int = num_prompts # 1000
        self.seed: int = 0
        self.hf_max_batch_size: int = None
        self.trust_remote_code: bool = True
        self.max_model_len: int = None
        self.dtype: str = 'auto'
        self.enforce_eager: bool = True
        self.kv_cache_dtype: str = "auto"
        self.device: str = "cuda"

        # added parameters
        self.weight_load_degree: str = '16'
        self.gpu_use_ratio: float = 0.9
        # TODO: setting temperature
        self.temperature: float = 1.0
        # TODO: setting ignore_eos
        self.ignore_eos: bool = False

        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.dataset is None:
            assert self.input_len is not None
            assert self.output_len is not None
        else:
            assert self.input_len is None

        if self.backend == "vllm":
            if self.hf_max_batch_size is not None:
                raise ValueError("HF max batch size is only for HF backend.")
        elif self.backend == "hf":
            if self.hf_max_batch_size is None:
                raise ValueError("HF max batch size is required for HF backend.")
            if self.quantization is not None:
                raise ValueError("Quantization is only for vLLM backend.")
        elif self.backend == "mii":
            if self.dtype != "auto":
                raise ValueError("dtype must be auto for MII backend.")
            if self.n != 1:
                raise ValueError("n must be 1 for MII backend.")
            if self.use_beam_search:
                raise ValueError("Beam search is not supported for MII backend.")
            if self.quantization is not None:
                raise ValueError("Quantization is only for vLLM backend.")
            if self.hf_max_batch_size is not None:
                raise ValueError("HF max batch size is only for HF backend.")
            if self.tokenizer != self.model:
                raise ValueError("Tokenizer must be the same as the model for MII "
                                "backend.")







# start a model for inference
def start_a_model_inference_child_process(
        communicator: LLM_COMMUNICATOR, use_vllm: bool, gpus: str, model_id: int, model: str = "huggyllama/llama-7b", 
        return_str=True, req_num=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    # TODO: some models do not support dynamic model weight loading now
    os.environ['USE_VLLM']='False'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
    if use_vllm:
        os.environ['USE_VLLM']='True'
        os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False'
      
    # os.environ['RUN_MULTI_MODEL'] = 'True'
    args = InferenceArgs(model, req_num)

    # set os.environ['CUDA_VISIBLE_DEVICES'] before importing benchmark_throughput
    # benchmark_throughput.SHARED_CONTECT.shared_setting = SHARED_CONTECT.shared_setting
    # set shared id for each model
    SHARED_CONTECT.shared_id = model_id
    SHARED_CONTECT.communicator = communicator
    SHARED_CONTECT.return_str = return_str
    SHARED_CONTECT.tot_req_num_remained = req_num
    # benchmark_throughput.main(args)
    try:
        benchmark_throughput.main(args)
    except Exception as e:
        print(f"Exception in running benchmark_throughput.main(): {e}")
        print(traceback.format_exc())





# start a model for inference
def start_a_model_inference(
        communicator: LLM_COMMUNICATOR, use_vllm: bool, gpus: str, model_id: int, model: str = "huggyllama/llama-7b", 
        return_str=True, req_num=None):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(start_a_model_inference_child_process, communicator, use_vllm, gpus, model_id, model, return_str, req_num)



# support data parallel
def get_exec_settings_from_exec_plans(
        exec_plan: MyExecPlan, available_gpus: List[int], tot_gpu_num: int, gpu_order_we_set: List[int]):
    """
        Get the exec setting to store in the SHARED_CONTECT later, based on the given exec_plan.
    """

    # the key of a exec plan is (tp, gpu_ratio, wldeg, cache_gpu_num) 
    # ==> (tp, gpu_ratio, wldeg, cache_gpu_num, dp_size) 
    # tp, gpu_ratio, wldeg, cache_gpu_num = exec_plan.get_key()
    tp, gpu_ratio, wldeg, cache_gpu_num, dp_size = exec_plan.get_key()
    gpu_list = available_gpus + [i for i in range(tot_gpu_num) if i not in available_gpus]

    # reorder gpu list according to the gpu order we set
    gpu_list = [gpu_order_we_set[i] for i in gpu_list]
    print(f"gpu_list to set: {gpu_list}", flush=True)


    # tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    # new_setting = [tp, int(gpu_ratio*10), wldeg] + gpu_list
    new_setting = [tp, int(gpu_ratio*10), wldeg, dp_size] + gpu_list
    return new_setting




# TODO: setting model_paths
def get_model_path_list() -> List[str]:
    model_paths = [
                #     'NousResearch/Llama-2-7b-hf', 
                #    'NousResearch/Llama-2-7b-chat-hf',
                # #    'NousResearch/Llama-2-13b-hf',
                # #    'NousResearch/Llama-2-70b-hf',
                   'THUDM/chatglm3-6b',
                #    'EleutherAI/gpt-j-6b', 
                # #    'EleutherAI/gpt-neox-20b',
                # #    'baichuan-inc/Baichuan2-13B-Chat',
                #    'baichuan-inc/Baichuan-7B',
                # #    'mistralai/Mixtral-8x7B-v0.1'
                # 
                # NEWROUND models
                'lmsys/vicuna-13b-v1.5',
                'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
                'chavinlo/alpaca-13b',
                'project-baize/baize-v2-13b',
                'TheBloke/koala-13B-HF',
                'databricks/dolly-v2-12b',
                'mosaicml/mpt-7b-chat',
                ]
    
    # for test
    model_paths = ['NousResearch/Llama-2-7b-hf']
    return model_paths



def query_use_vllm(model_path: str) -> bool:
    setting_dict = {
        'NousResearch/Llama-2-7b-hf': False, 
        'NousResearch/Llama-2-7b-chat-hf': False,
        'NousResearch/Llama-2-13b-hf': False,
        'NousResearch/Llama-2-70b-hf': False,
        'THUDM/chatglm3-6b': True,
        'EleutherAI/gpt-j-6b': True, 
        'EleutherAI/gpt-neox-20b': True,
        'baichuan-inc/Baichuan2-13B-Chat': True,
        'baichuan-inc/Baichuan-7B': True,
        'mistralai/Mixtral-8x7B-v0.1': True,
        # 
        # NEWROUND models
        'lmsys/vicuna-13b-v1.5': True,
        'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5': True,
        'chavinlo/alpaca-13b': True,
        'project-baize/baize-v2-13b': True,
        'TheBloke/koala-13B-HF': True,
        'databricks/dolly-v2-12b': True,
        'mosaicml/mpt-7b-chat': True,        
    }
    return setting_dict[model_path]






def prepare_exec_plan_states(
        plan_group_seq: MyExecPlanGroupSeq
        )->List[List[MyExecPlanState]]:
    '''
        Prepare the corresponding MyExecPlanState object for each exec plan.
        Output:
            List of exec plan state objects for each stage.
    '''
    plan_group_list = plan_group_seq.plan_group_seq
    plan_state_group_list: List[List[MyExecPlanState]] = [[] for i in range(len(plan_group_list))]
    checked_model_ids: List[int] = list()

    for stage_i in range(len(plan_group_list)-1, -1, -1):
        plan_group = plan_group_list[stage_i]
        plan_state_group: List[MyExecPlanState] = plan_state_group_list[stage_i]
        if stage_i == 0:
            # stage 0; depend on no exec plans
            for exec_plan in plan_group.exec_plans:
                last_exec_plan_for_the_model = False
                if exec_plan.model.model_id not in checked_model_ids:
                    last_exec_plan_for_the_model = True
                    checked_model_ids.append(exec_plan.model.model_id)

                plan_state_group.append(MyExecPlanState(
                    exec_plan, launched=False, stage_i=stage_i, 
                    last_exec_plan_for_the_model=last_exec_plan_for_the_model, 
                    need_prepare_infer_env=False))

        else:
            # check whether there is the same exec plan in the last stage
            last_stage_exec_plans = plan_group_seq.plan_group_seq[stage_i-1].exec_plans
            last_stage_exec_plans_info = [
                (exec_plan.model.model_id, exec_plan.get_key()) for exec_plan in last_stage_exec_plans
            ]
            
            for exec_plan in plan_group.exec_plans:

                # TODO: when considering data parallel, we may need to redefine ``need_prepare_infer_env``
                # if we allow some dp workers' infer env to be reused

                need_prepare_infer_env = True
                if (exec_plan.model.model_id, exec_plan.get_key()) in last_stage_exec_plans_info:
                    # this exec plan is not a new one
                    need_prepare_infer_env = False
                
                last_exec_plan_for_the_model = False
                if exec_plan.model.model_id not in checked_model_ids:
                    last_exec_plan_for_the_model = True
                    checked_model_ids.append(exec_plan.model.model_id)
                
                plan_state_group.append(MyExecPlanState(
                    exec_plan, launched=False, stage_i=stage_i, 
                    last_exec_plan_for_the_model=last_exec_plan_for_the_model, 
                    need_prepare_infer_env=need_prepare_infer_env))   

    return plan_state_group_list







def search_best_scheduling(
        gen_execplans_baseline:str,
        search_method_baseline:str,
        model_paths: List[str], 
        # 
        out_edge_dict: Dict[int, List[int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, inp_generator, inp_merger, outlen_generator,
        # 
        tot_gpu_num: int = 4,
        max_group_seq_num: int = 100,
    )->List[List[MyExecPlanState]]:
    
    # 1. first search the best scheduling
    
    # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # search_method_baseline = 'ours' # 'naive'  'ours'
    # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # search_method_baseline = 'ours' # 'naive'  'ours'
    best_group_seq = get_best_model_schedule(
        search_method_baseline,
        gen_execplans_baseline,
        check_gap,
        sort_input,
        model_paths, 
        num_prompts,
        inp_generator,
        inp_merger,
        outlen_generator,
        out_edge_dict,
        sample_config=(1, 1, -1, 0),
        trust_remote_code=True, revision=None,
        gpu_name='A100-80G', tot_gpu_num = tot_gpu_num, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=max_group_seq_num,
    )


    # 2. convert best_group_seq to exec plan state list
    plan_state_group_list = prepare_exec_plan_states(best_group_seq)
    
    return plan_state_group_list





def initialize_SHARED_CONTECT(
        tot_gpu_num: int,
        model_paths: List[str], 
        check_gap: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        gpu_order_we_set: List[int],
    ) -> Tuple[List[MyExecPlanState], int, List[MyExecPlanState]]:
    '''
        Update: (1) SHARED_CONTECT events, shared_finish_status, shared_setting
                (2) call SHARED_CONTECT.start_specific_models()
        Output: (1) launched_exec_plan_states; (2) new target stage i; (3) candidate_exec_plan_states
    '''

    import ctypes
    
    SHARED_CONTECT.set_execution_plan_size(tot_gpu_num)
    counter = Array('i', [0 for i in range(len(model_paths)*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+len(model_paths))]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.started_status = [Event() for _ in range(len(model_paths))]
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(len(model_paths))])
    # add check_out_gaps
    check_out_gaps = Array('i', [int(1e9)]*len(model_paths)) # 'd' is for double
    SHARED_CONTECT.check_out_gaps = check_out_gaps
    SHARED_CONTECT.check_in_gap = check_gap

    
    # set the initial execution plan
    available_gpus: List[int] = list(range(tot_gpu_num))
    launched_exec_plan_states: List[MyExecPlanState] = plan_state_group_list[0]
    for exec_plan_state in launched_exec_plan_states:
        
        exec_plan = exec_plan_state.exec_plan
        # exec_plan_state.set_comp_gpus(available_gpus[:exec_plan.num_worker])
        # data parallel
        # NOTE: the gpu assignment policy for data parallel
        exec_plan_state.set_comp_gpus(available_gpus[:exec_plan.num_worker*exec_plan.dp_size])
        
        setting = get_exec_settings_from_exec_plans(
            exec_plan=exec_plan, available_gpus=available_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
        SHARED_CONTECT.set_execution_plan(setting, model_ids=[exec_plan.model.model_id])
        # TODO: does not consider that we can reuse some dp workers' infer envs 
        # so that more than 1 dp driver worker's gpu infor should be record
        if exec_plan.model.model_id not in model_driver_worker_gpu_i:
            model_driver_worker_gpu_i[exec_plan.model.model_id] = available_gpus[0]

        # update the available gpus
        # available_gpus = available_gpus[exec_plan.num_worker:]
        # support data parallel
        available_gpus = available_gpus[exec_plan.num_worker*exec_plan.dp_size:]

        exec_plan_state.launched = True
        # NOTE: wait until all models finish the preparation before init their LLM objects to start them
        # SHARED_CONTECT.start_specific_models([exec_plan.model.model_id])

    new_target_stage_i: int = 1
    candidate_exec_plan_states: List[MyExecPlanState] = []
    if len(plan_state_group_list)>1:
        candidate_exec_plan_states = plan_state_group_list[1]

    return launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states




    



def get_the_next_round_exec_plan_schedule_deprecated(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], List[int], List[MyExecPlanState], int]:
    '''
        Output: 
            (1) the updated launched_exec_plan_states (i.e., running exec plan states);
            (2) the updated candidate_exec_plan_states;
            (3) the models to stop;
            (4) the new exec plans to launch;
            (5) the new target stage i;
    '''

    to_launch: List[MyExecPlanState] = list()
    to_launch_model_ids: List[int] = list()
    
    # 0. get the exec plan that must keep running as it is the last plan for that model
    # AND
    # 1. get the exec plans before the target_stage_i
    cand_to_launch_list: List[List[MyExecPlanState]] = [list(), list()]
    for plan_state in launched_exec_plan_states:
        # do not consider exec plans of finished models
        if SHARED_CONTECT.query_finish_status(plan_state.exec_plan.model.model_id):
            continue

        if plan_state.stage_i < target_stage_i:
            if plan_state.last_exec_plan_for_the_model:
                to_launch.append(plan_state)
                to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
                print(f"to_launch add 0: {str(plan_state)}")
            else:
                # consider to stop it
                cand_to_launch_list[0].append(plan_state)
        else:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            print(f"to_launch add 1: {str(plan_state)}")
    
    # 2. deal with the models which does not change their exec plan
    for plan_state in candidate_exec_plan_states:
        # do not consider exec plans of finished models
        if SHARED_CONTECT.query_finish_status(plan_state.exec_plan.model.model_id):
            plan_state.launched = True
            continue

        if not plan_state.need_prepare_infer_env:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            print(f"to_launch add 2: {str(plan_state)}")
        else:
            # may not be able to launch these plans
            cand_to_launch_list[1].append(plan_state)
    
    print(f"to_launch 1: {[str(i) for i in to_launch]}")
    print(f"cand_to_launch_list 1: {[[str(i) for i in cand_to_launch] for cand_to_launch in cand_to_launch_list]}")


    # 3. determine the exec plans running in the last round from cand_to_launch
    occupied_gpus: List[int] = list()
    for plan_state in to_launch:
        occupied_gpus.extend(SHARED_CONTECT.get_comp_gpus(plan_state.exec_plan.model.model_id))
    available_gpus = [i for i in range(tot_gpu_num) if i not in occupied_gpus]

    # sort cand exec plans from newer stage to older stage, from large tp_size to small tp_size
    cand_to_launch = sorted(cand_to_launch, key=lambda i: (i.stage_i, i.exec_plan.num_worker), reverse=True)
    # NOTE: first check the target stage
    # cand_to_launch = sorted(cand_to_launch_list[1], key=lambda i: i.exec_plan.num_worker, reverse=True)
    new_launch: List[MyExecPlanState] = list()
    model_ids_to_stop: List[int] = list()
    new_candidate_exec_plan_states: List[MyExecPlanState] = list()
    for plan_state in cand_to_launch:
        if plan_state.exec_plan.model.model_id in to_launch_model_ids:
            # we already select a plan for this model ==> this model will not change exec plan
            # this exec plan has been replaced by a new same one
            continue

        tp_size = plan_state.exec_plan.num_worker
        if tp_size <= len(available_gpus):
            # can run
            to_launch.append(plan_state)
            print(f"to_launch add 3: {str(plan_state)}")
            if plan_state.launched:
                # this plan is running and we do not stop it
                comp_gpus = SHARED_CONTECT.get_comp_gpus(plan_state.exec_plan.model.model_id)
                available_gpus = [i for i in available_gpus if i not in comp_gpus]
            else:
                plan_state.launched = True
                if plan_state.exec_plan.model.model_id not in model_driver_worker_gpu_i:
                    # this model is started for the first time
                    plan_state.set_comp_gpus(available_gpus[:tp_size])
                    model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id] = available_gpus[0]
                    available_gpus = available_gpus[tp_size:]
                else:
                    # this model has been started
                    driver_gpu_i = model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id]
                    assert driver_gpu_i in available_gpus, f"The driver gpu is not available: {driver_gpu_i, available_gpus}"
                    
                    available_gpus = [i for i in available_gpus if i != driver_gpu_i]
                    comp_gpus = [driver_gpu_i]+available_gpus[:tp_size-1]
                    plan_state.set_comp_gpus(comp_gpus)
                    available_gpus = available_gpus[tp_size-1:]
                # available_gpus = available_gpus[tp_size:]
                new_launch.append(plan_state)
        else:
            # cannot run
            print(f"cannot run")
            if plan_state.launched:
                print(f"plan launced: {str(plan_state)}")
                model_ids_to_stop.append(plan_state.exec_plan.model.model_id)
            else:
                print(f"add to new candidate: {str(plan_state)}")
                new_candidate_exec_plan_states.append(plan_state)


    new_target_stage_i = target_stage_i
    if len(new_candidate_exec_plan_states) == 0:
        new_target_stage_i = target_stage_i + 1
        if new_target_stage_i < len(plan_state_group_list):
            new_candidate_exec_plan_states = plan_state_group_list[new_target_stage_i]
    
    return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i








def _get_the_first_plan_state_when_sorted_by_gpu_num_and_topology(
        plan_state_list: List[MyExecPlanState]
    )->List[MyExecPlanState]:
    """
        Sort the given list of plan states by the topology order and their gpu numbers.
        Policy:
            if a plan state A depends on plan state B, then A must be checked later than B.
            但是这种定义的话可能会导致有的gpu num小的model反倒被提前check。和我们之前想的先check大一点gpu需求的模型的想法相违背。
        但是其实更本质的做法是我们直接去找可能得模型组合，满足依赖关系，同时尽量把空闲的GPU占满 ==> 还是先greedy地来吧。
        Output:
            1. the first plan state to consider
            2. the remaining plan states to be considered
    """
    # get the root plan states first
    model_ids = [plan_state.exec_plan.model.model_id for plan_state in plan_state_list]
    roots = [plan_state for plan_state in plan_state_list if len(set(plan_state.exec_plan.model.input_model_ids).intersection(model_ids)) == 0]
    roots = sorted(roots, key=lambda plan_state: (plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size) , reverse=True)
    remaining_plan_states = [plan_state for plan_state in plan_state_list if plan_state != roots[0]]
    return roots[0], remaining_plan_states









def _try_to_load_exec_plans(
        cands_to_launch: List[MyExecPlanState],
        to_launch: List[MyExecPlanState],
        # new_launch: List[MyExecPlanState],
        # model_ids_to_stop: List[int],
        new_candidate_exec_plan_states: List[MyExecPlanState],
        model_driver_worker_gpu_i: Dict[int,int],
        available_gpus: List[int],
        to_launch_model_ids: List[int]
        ) -> List[int]:
    '''
        Get the exec plans to launch from the given candidates.
        Update: to_launch, to_launch_model_ids, model_driver_worker_gpu_i;
                new_launch,
                model_ids_to_stop, new_candidate_exec_plan_states;
                set the comp gpus of the to-launch plans;
        Output: available_gpus
        NOTE:
            since we support model-level pipeline: we do not simply sort the plan states by their gpu number, but also consider their dependency,
            therefore, we call ``_get_the_first_plan_state_when_sorted_by_gpu_num_and_topology`` to get the next plan state to consider every time.
    '''
    # for plan_state in cands_to_launch:
    # support model-level parallel
    while len(cands_to_launch) > 0:
        plan_state, cands_to_launch = _get_the_first_plan_state_when_sorted_by_gpu_num_and_topology(cands_to_launch)


        model_id = plan_state.exec_plan.model.model_id
        if model_id in to_launch_model_ids:
            # we already select a plan for this model ==> this model will not change exec plan
            # this exec plan has been replaced by a new same one
            continue

        # tp_size = plan_state.exec_plan.num_worker
        # if tp_size <= len(available_gpus):
        # NOTE: dp parallel
        gpu_num = plan_state.exec_plan.num_worker * plan_state.exec_plan.dp_size
        if gpu_num <= len(available_gpus):
            # can run
            to_launch.append(plan_state)
            to_launch_model_ids.append(model_id)
            print(f"to_launch add 3: {str(plan_state)}")
            if plan_state.launched:
                # this plan is running and we do not stop it
                # comp_gpus = SHARED_CONTECT.get_comp_gpus(plan_state.exec_plan.model.model_id)
                comp_gpus = plan_state.get_comp_gpus()
                available_gpus = [i for i in available_gpus if i not in comp_gpus]
            else:
                plan_state.launched = True
                # we fix a bug and do not need to care about model_driver_worker_gpu_i now.
                # plan_state.set_comp_gpus(available_gpus[:tp_size])
                # model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id] = available_gpus[0]
                # available_gpus = available_gpus[tp_size:]      

                plan_state.set_comp_gpus(available_gpus[:gpu_num])
                model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id] = available_gpus[0]
                available_gpus = available_gpus[gpu_num:]   

                
                # if plan_state.exec_plan.model.model_id not in model_driver_worker_gpu_i:
                #     # this model is started for the first time
                #     plan_state.set_comp_gpus(available_gpus[:tp_size])
                #     model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id] = available_gpus[0]
                #     available_gpus = available_gpus[tp_size:]
                # else:
                #     # this model has been started
                #     driver_gpu_i = model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id]
                #     assert driver_gpu_i in available_gpus, f"{plan_state.exec_plan.model.model_name}: The driver gpu is not available: {driver_gpu_i, available_gpus}"
                    
                #     available_gpus = [i for i in available_gpus if i != driver_gpu_i]
                #     comp_gpus = [driver_gpu_i]+available_gpus[:tp_size-1]
                #     plan_state.set_comp_gpus(comp_gpus)
                #     available_gpus = available_gpus[tp_size-1:]
                # # available_gpus = available_gpus[tp_size:]
                # new_launch.append(plan_state) 
        else:
            # cannot run
            print(f"cannot run")
            if plan_state.launched:
                print(f"plan launced: {str(plan_state)}")
                # model_ids_to_stop.append(plan_state.exec_plan.model.model_id)
            else:
                print(f"add to new candidate: {str(plan_state)}")
                new_candidate_exec_plan_states.append(plan_state)

    return available_gpus










def _has_model_finished(
        plan_state_group_list:List[List[MyExecPlanState]],
        stage_i: int,
    )-> bool:
    finished = [i for i in plan_state_group_list[stage_i] if SHARED_CONTECT.query_finish_status(i.exec_plan.model.model_id)]
    return len(finished) > 0

def _get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
    # )->Tuple[List[MyExecPlanState], List[MyExecPlanState], List[int], List[MyExecPlanState], int]:
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], int]:
    '''
        Output: 
            (1) the updated launched_exec_plan_states (i.e., running exec plan states);
            (2) the updated candidate_exec_plan_states;
            (3) the models to stop;
            (4) the new exec plans to launch;
            (5) the new target stage i;
    '''

    to_launch: List[MyExecPlanState] = list()
    to_launch_model_ids: List[int] = list()

    launched_plan_gpus = {(i.exec_plan.model.model_id, tuple(i.exec_plan.get_key())):i.get_comp_gpus() for i in launched_exec_plan_states}
    
    # 0. get the exec plan that must keep running as it is the last plan for that model
    # AND
    # 1. get the exec plans before the target_stage_i
    cand_to_launch_list: List[List[MyExecPlanState]] = [list(), list()]
    for plan_state in launched_exec_plan_states:
        # do not consider exec plans of finished models
        if SHARED_CONTECT.query_finish_status(plan_state.exec_plan.model.model_id):
            continue

        if plan_state.stage_i < target_stage_i:
            if plan_state.last_exec_plan_for_the_model:
                to_launch.append(plan_state)
                to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
                print(f"to_launch add 0: {str(plan_state)}")
            else:
                # consider to stop it
                cand_to_launch_list[0].append(plan_state)
        else:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            print(f"to_launch add 1: {str(plan_state)}")
    
    # 2. deal with the models which does not change their exec plan
    for plan_state in candidate_exec_plan_states:
        # do not consider exec plans of finished models
        if SHARED_CONTECT.query_finish_status(plan_state.exec_plan.model.model_id):
            plan_state.launched = True
            continue

        if not plan_state.need_prepare_infer_env:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            plan_state.set_comp_gpus(launched_plan_gpus[(plan_state.exec_plan.model.model_id, tuple(plan_state.exec_plan.get_key()))])

            print(f"to_launch add 2: {str(plan_state)}")
        else:
            # may not be able to launch these plans
            cand_to_launch_list[1].append(plan_state)
    
    print(f"to_launch 1: {[str(i) for i in to_launch]}")
    print(f"cand_to_launch_list 1: {[[str(i) for i in cand_to_launch] for cand_to_launch in cand_to_launch_list]}")


    # 3. determine the exec plans running in the last round from cand_to_launch
    occupied_gpus: List[int] = list()
    for plan_state in to_launch:
        # occupied_gpus.extend(SHARED_CONTECT.get_comp_gpus(plan_state.exec_plan.model.model_id))
        occupied_gpus.extend(plan_state.get_comp_gpus())
    available_gpus = [i for i in range(tot_gpu_num) if i not in occupied_gpus]

    new_launch: List[MyExecPlanState] = list()
    # can be obtained from the currently launched exec plans
    model_ids_to_stop: List[int] = list()
    # can only be from the target stage
    new_candidate_exec_plan_states: List[MyExecPlanState] = list()


    # sort cand exec plans from newer stage to older stage, from large tp_size to small tp_size
    # cand_to_launch = sorted(cand_to_launch, key=lambda i: (i.stage_i, i.exec_plan.num_worker), reverse=True)
    # NOTE: first check the target stage

    # TODO: 从这里开始写，但是排序要保证拓扑在里面，还是说，我们要保证有依赖关系的model组合一定要一起被load上去。感觉好像
    # 没有必要非得把有依赖关系的绑在一起。

    cand_to_launch = sorted(cand_to_launch_list[1], key=lambda i: i.exec_plan.num_worker, reverse=True)
    available_gpus = _try_to_load_exec_plans(cand_to_launch, to_launch, # new_launch, model_ids_to_stop,
        new_candidate_exec_plan_states, model_driver_worker_gpu_i, available_gpus, to_launch_model_ids)

    new_target_stage_i = target_stage_i
    if len(new_candidate_exec_plan_states)>0:
        # consider the old stages
        cand_to_launch = sorted(cand_to_launch_list[0], key=lambda i: i.exec_plan.num_worker, reverse=True)
        available_gpus = _try_to_load_exec_plans(cand_to_launch, to_launch, # new_launch, model_ids_to_stop,
            new_candidate_exec_plan_states, model_driver_worker_gpu_i, available_gpus, to_launch_model_ids)        
    else:
        # consider the next stage
        new_target_stage_i = target_stage_i + 1
        if new_target_stage_i < len(plan_state_group_list):
            # there are stages left
            new_candidate_exec_plan_states = plan_state_group_list[new_target_stage_i]
            if _has_model_finished(plan_state_group_list, target_stage_i):
                # to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i = \
                to_launch, new_candidate_exec_plan_states, new_target_stage_i = \
                    _get_the_next_round_exec_plan_schedule(
                    to_launch, new_candidate_exec_plan_states,
                    new_target_stage_i,
                    tot_gpu_num,
                    plan_state_group_list,
                    model_driver_worker_gpu_i,
                )
    
    # return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i
    return to_launch, new_candidate_exec_plan_states, new_target_stage_i





def get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], List[int], List[MyExecPlanState], int]:

    # 1. get the new launch plan
    to_launch, new_candidate_exec_plan_states, new_target_stage_i = \
        _get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states, candidate_exec_plan_states,
        target_stage_i,
        tot_gpu_num,
        plan_state_group_list,
        model_driver_worker_gpu_i,
    )

    # 2. get model_ids_to_stop and new_launch
    launched_exec_plans = [(i.exec_plan.model.model_id,i.exec_plan.get_key()) for i in launched_exec_plan_states]
    to_launch_exec_plans = [(i.exec_plan.model.model_id,i.exec_plan.get_key()) for i in to_launch]
    
    new_launch = [i for i in to_launch if (i.exec_plan.model.model_id,i.exec_plan.get_key()) not in launched_exec_plans]
    model_ids_to_stop = [i.exec_plan.model.model_id for i in launched_exec_plan_states 
                         if (i.exec_plan.model.model_id,i.exec_plan.get_key()) not in to_launch_exec_plans]

    return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i




def start_exec_plans(new_launch: List[MyExecPlanState], tot_gpu_num: int, gpu_order_we_set: List[int]):
    for exec_plan_state in new_launch:
        
        exec_plan = exec_plan_state.exec_plan
        assert len(exec_plan_state.comp_gpus) == (exec_plan.num_worker * exec_plan.dp_size)
        
        setting = get_exec_settings_from_exec_plans(
            exec_plan=exec_plan, available_gpus=exec_plan_state.comp_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
        SHARED_CONTECT.set_execution_plan(setting, model_ids=[exec_plan.model.model_id])

        exec_plan_state.launched = True
        SHARED_CONTECT.start_specific_models([exec_plan.model.model_id])






def get_out_edge_dict_from_in_edge_dict_with_inp_nodes(
        in_edge_dict:Dict[int, List[int]]):
    """
        NOTE: we use negative model_ids to represent dummy inp nodes.
    """
    out_edge_dict = defaultdict(list)
    for tgt, srcs in in_edge_dict.items():
        # tgt cannot be dummy inp nodes, as they do not have inputs
        for src in srcs:
            if src >= 0:
                out_edge_dict[src].append(tgt)
    return out_edge_dict






def init_prompts_for_the_model_system(
        communicator: LLM_COMMUNICATOR,
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]], 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]], 
        num_prompts: int):
    """
        Sample input dataset for the model system.
        INPUT:
            node_dataset_partition_mapping: {model_id: (dataset_name, chunk_id, chunk_size)}
        NOTE: we also set the total number of requests each model needs to do inference for.
        OUTPUT:
            req_num_dict: the number of req to answer for each model.
    """

    # simply use the tokenizer of llama2 7b to check the lengths of the prompts
    args = InferenceArgs(model='NousResearch/Llama-2-7b-hf', num_prompts=num_prompts)

    from transformers import AutoTokenizer
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    datasets = set([v[0] for v in node_dataset_chunk_mapping.values()])
    dataset_dict = dict()
    for dataset in datasets:
        requests = benchmark_throughput.sample_requests(
            dataset, args.num_prompts, tokenizer,args.output_len)
        inp_prompts = [(i, req[0]) for i, req in enumerate(requests)]
        dataset_dict[dataset] = inp_prompts

    req_num_dict = defaultdict(int)
    prompts_dict = dict()
    for model_id, (dataset, chunk_id, chunk_size) in node_dataset_chunk_mapping.items():
        inp_prompts = dataset_dict[dataset]
        to_add = inp_prompts
        if chunk_size > 0:
            to_add = [(i, req[chunk_id*chunk_size:(chunk_id+1)*chunk_size]) for i, req in inp_prompts if (len(req)>chunk_id*chunk_size)]
        
        # communicator.add_seqs(model_id, to_add)
        prompts_dict[model_id] = to_add
        req_num_dict[model_id] = len(to_add)
    

    # set the req number for each non-dummy model node
    tot_node_num = len(req_num_dict) + len(in_edge_dict_with_dummy_inp_nodes)
    visited = list(req_num_dict.keys())
    while len(visited) < tot_node_num:
        for tgt, srcs in in_edge_dict_with_dummy_inp_nodes.items():
            if set(srcs).issubset(visited):
                req_num_dict[tgt] = min([req_num_dict[src] for src in srcs])
                visited.append(tgt)

    
    ungened_out_req_nums = req_num_dict.copy()

    # send the req_num_dict to the communicator
    # set the unavailable_req_num (i.e., unavailable inp req num) for the dummy inp nodes to 0
    for model_id in node_dataset_chunk_mapping:
        req_num_dict[model_id] = 0
    communicator.init_unavailable_req_nums_and_ungened_out_req_nums(req_num_dict, ungened_out_req_nums)

    # send the prompts of the dummy inp nodes (i.e., dummy inp nodes' outputs) to the communicator 
    for model_id, to_add in prompts_dict.items():
        communicator.add_seqs(model_id, to_add)

    return req_num_dict
        
    




def get_return_str(out_edge_dict: Dict[int, List[int]], model_id: int, model_paths: List[str])->bool:
    outs = out_edge_dict[model_id]
    tgt = model_paths[model_id]
    return (True in [tgt != model_paths[out] for out in outs])




def set_check_in_out_gap(
        curr_stage_plan_states: List[MyExecPlanState], 
        check_gap: int, out_edge_dict: Dict[int, List[int]]):
    """
        If a model has no input model in this stage, check_in_gap is 1e9;
        If a model has no output model in this stage, check_out_gap is 1e9.
        OUTPUT:
            SHARED_CONTECT.check_in (deleted), SHARED_CONTECT.check_in_gap, SHARED_CONTECT.check_out_gap
        NOTE:
            all the models share the same fixed SHARED_CONTECT.check_in_gap;
            different models have different SHARED_CONTECT.check_out_gap which may change over stages
    """

    model_ids = [plan_state.exec_plan.model.model_id for plan_state in curr_stage_plan_states]
    for plan_state in curr_stage_plan_states:
        model_id = plan_state.exec_plan.model.model_id
        out_model_ids = out_edge_dict[model_id]
        if len(set(out_model_ids).intersection(model_ids)) > 0:
            SHARED_CONTECT.check_out_gaps[model_id] = check_gap
        else:
            SHARED_CONTECT.check_out_gaps[model_id] = int(1e9)


    # model_ids = [plan_state.exec_plan.model.model_id for plan_state in curr_stage_plan_states]
    # for plan_state in curr_stage_plan_states:
    #     model_id = plan_state.exec_plan.model.model_id
    #     inp_model_ids = plan_state.exec_plan.model.input_model_ids
    #     if len(set(inp_model_ids).intersection(model_ids)) > 0:
    #         SHARED_CONTECT.check_in_gap = check_gap
    #         SHARED_CONTECT.check_in = True
    #     else:
    #         SHARED_CONTECT.check_in_gap = int(1e9)
    #         SHARED_CONTECT.check_in = False
    #     out_model_ids = out_edge_dict[model_id]
    #     if len(set(out_model_ids).intersection(model_ids)) > 0:
    #         SHARED_CONTECT.check_out_gap = check_gap
    #     else:
    #         SHARED_CONTECT.check_out_gap = int(1e9)
        







def test_search(
        gen_execplans_baseline:str,
        search_method_baseline:str,
        # 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]],
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, inp_generator, inp_merger, outlen_generator,
        # 
        tot_gpu_num:int = 4,
        max_group_seq_num: float = float('inf'),
):
    import os
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'False'
    os.environ['COLLECT_TIME_LOG'] = 'False' 
    os.environ['MY_SORT_INPS'] = 'True' if sort_input else 'False'
    os.environ['GET_INP_FROM_COMMUNICATOR'] = 'True' # whether to get the input from the communicator

    print(f"os.environ['CUDA_VISIBLE_DEVICES'] in main_with_preemption: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    gpu_order_we_set = None
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpu_order_we_set = list(range(tot_gpu_num))
    else:
        gpu_order_we_set = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(gpu_order_we_set) == tot_gpu_num, f"gpu_order_we_set: {gpu_order_we_set} should contain {tot_gpu_num} cards"


    model_driver_worker_gpu_i: Dict[int, int] = dict()

    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4

    # loop = asyncio.get_running_loop()
    # tasks = []

    # search the best model scheduling
    model_paths = get_model_path_list()
    # convert the in_edge_dict to out_edge_dict
    out_edge_dict = get_out_edge_dict_from_in_edge_dict_with_inp_nodes(in_edge_dict_with_dummy_inp_nodes)
    plan_state_group_list:List[List[MyExecPlanState]] = search_best_scheduling(
        gen_execplans_baseline,
        search_method_baseline,
        model_paths, 
        # 
        out_edge_dict,
        check_gap, sort_input,
        num_prompts, inp_generator, inp_merger, outlen_generator,
        # 
        tot_gpu_num = tot_gpu_num, 
        max_group_seq_num = max_group_seq_num)

    print("\n\n\n\n\nfinish searching!\n\n\n\n\n", flush=True)
    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4







async def main_with_preemption(
        gen_execplans_baseline:str,
        search_method_baseline:str,
        # 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]],
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, inp_generator, inp_merger, outlen_generator,
        # 
        tot_gpu_num:int = 4,
        max_group_seq_num: float = float('inf'),
):
    import os
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'False'
    os.environ['COLLECT_TIME_LOG'] = 'False' 
    os.environ['MY_SORT_INPS'] = 'True' if sort_input else 'False'
    os.environ['GET_INP_FROM_COMMUNICATOR'] = 'True' # whether to get the input from the communicator

    print(f"os.environ['CUDA_VISIBLE_DEVICES'] in main_with_preemption: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    gpu_order_we_set = None
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpu_order_we_set = list(range(tot_gpu_num))
    else:
        gpu_order_we_set = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(gpu_order_we_set) == tot_gpu_num, f"gpu_order_we_set: {gpu_order_we_set} should contain {tot_gpu_num} cards"


    model_driver_worker_gpu_i: Dict[int, int] = dict()

    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4

    loop = asyncio.get_running_loop()
    tasks = []

    # search the best model scheduling
    model_paths = get_model_path_list()
    # convert the in_edge_dict to out_edge_dict
    out_edge_dict = get_out_edge_dict_from_in_edge_dict_with_inp_nodes(in_edge_dict_with_dummy_inp_nodes)
    plan_state_group_list:List[List[MyExecPlanState]] = search_best_scheduling(
        gen_execplans_baseline,
        search_method_baseline,
        model_paths, 
        # 
        out_edge_dict,
        check_gap, sort_input,
        num_prompts, inp_generator, inp_merger, outlen_generator,
        # 
        tot_gpu_num = tot_gpu_num, 
        max_group_seq_num = max_group_seq_num)

    print("\n\n\n\n\nfinish searching!\n\n\n\n\n", flush=True)
    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4
    

    launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states = initialize_SHARED_CONTECT(
        tot_gpu_num=tot_gpu_num, model_paths=model_paths, check_gap=check_gap,
        plan_state_group_list=plan_state_group_list,
        model_driver_worker_gpu_i=model_driver_worker_gpu_i, 
        gpu_order_we_set=gpu_order_we_set)
    first_stage_model_ids = [exec_plan_state.exec_plan.model.model_id for exec_plan_state in launched_exec_plan_states]




    print(f"\nTIMESTAMP 1: {time.perf_counter()}\n")

    with MyManager() as manager:

        print(f"\nTIMESTAMP 2: {time.perf_counter()}\n")

        communicator: LLM_COMMUNICATOR = manager.Communicator(len(model_paths), in_edge_dict_with_dummy_inp_nodes)

        # set inputs for dummy inp nodes in the system
        req_num_dict = init_prompts_for_the_model_system(communicator, node_dataset_chunk_mapping, in_edge_dict_with_dummy_inp_nodes,
                                                         num_prompts)


        print(f"\nTIMESTAMP 3: {time.perf_counter()}\n")

        # launch the exec_plans in order
        with ProcessPoolExecutor(max_workers=len(model_paths)) as executor:

            print(f"\nTIMESTAMP 4: {time.perf_counter()}\n")

            # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
            
            # start a process for each model, no matter it is in launched_exec_plan_states or not
            # NOTE: we will use os.environ['TOT_ORDERED_GPUS'] to control the actual gpu order in each model to support reschedule
            for model_id, model_path in enumerate(model_paths):
                tasks.append(
                    loop.run_in_executor(
                        executor, start_a_model_inference, 
                        communicator, query_use_vllm(model_path), ','.join([str(i) for i in gpu_order_we_set]), model_id, model_path, 
                        get_return_str(out_edge_dict=out_edge_dict, model_id=model_id, model_paths=model_paths),
                        req_num_dict[model_id],
                    )        
                )



            print(f"\nTIMESTAMP 5: {time.perf_counter()}\n")


            # wait for all processes finishing the preparation before initializing their LLM objects
            SHARED_CONTECT.wait_all_models_to_finish_preparation_before_init_LLM(model_ids=range(len(model_paths)))
            # start the first stage models
            SHARED_CONTECT.start_specific_models(first_stage_model_ids)


            start = time.perf_counter()
            print(f"Outer iter start time ---abs: {start}")

            pending_list = tasks
            model_schedule_iter = 0
            while len(pending_list) > 0:
                # repeat this loop until all models are finished
                print(f"a new iteration==================", flush=True)
                print(f"MAIN PROCESS: {[str(plan_state) for plan_state in launched_exec_plan_states]}", flush=True)

                done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)

                # <jingzhi> For Profiling
                start_waiting = time.perf_counter()
                print(f"MAIN PROCESS: total time to launch processes (just the value of iter 0 is useful) {model_schedule_iter}: {start_waiting-start}s ---abs: {start_waiting}", flush=True)

                # 1. get models that need to be stopped
                launched_exec_plan_states, candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i = \
                    get_the_next_round_exec_plan_schedule(
                        launched_exec_plan_states, candidate_exec_plan_states,
                        new_target_stage_i,
                        tot_gpu_num, plan_state_group_list,
                        model_driver_worker_gpu_i,
                    )
                

                # 2. stop the models
                SHARED_CONTECT.stop_specific_models(model_ids_to_stop)

                # TODO (jingzhi) try to make the finished processes release their resources
                print(len(done_list), len(pending_list))
                print(f"MAIN PROCESS: next iter plans: {[str(plan_state) for plan_state in launched_exec_plan_states]}")
                print(f"MAIN PROCESS: model_ids_to_stop: {model_ids_to_stop}")
                print(f"MAIN PROCESS: new_launch: {[str(plan_state) for plan_state in new_launch]}")
                print(f"MAIN PROCESS: candidate_exec_plan_states: {[str(plan_state) for plan_state in candidate_exec_plan_states]}")
                
                for task in done_list:
                    await task

                # 3. wait for model finish preparing for rescheduling            
                SHARED_CONTECT.wait_all_models_to_finish_prepare_for_reschedule(model_ids_to_stop)
                
                # 4. start newly launched models
                set_check_in_out_gap(curr_stage_plan_states=launched_exec_plan_states, check_gap=check_gap, out_edge_dict=out_edge_dict)
                start_exec_plans(new_launch, tot_gpu_num, gpu_order_we_set=gpu_order_we_set)


                # <jingzhi> For Profiling
                end_waiting = time.perf_counter()
                print(f"MAIN PROCESS: total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
                model_schedule_iter += 1

            end = time.perf_counter()
            print(f"total running time: {end-start}s ---abs: {end}")












async def main_with_preemption_debug():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'False'
    os.environ['COLLECT_TIME_LOG'] = 'False' 

    model_driver_worker_gpu_i: Dict[int, int] = dict()
    
    # TODO: setting tot_gpu_num
    tot_gpu_num = 2

    loop = asyncio.get_running_loop()
    tasks = []

    # search the best model scheduling
    model_paths = get_model_path_list()[-2:-1]
    plan_state_group_list:List[List[MyExecPlanState]] = search_best_scheduling(model_paths, tot_gpu_num = tot_gpu_num)

    print("\n\n\n\n\nfinish searching!\n\n\n\n\n", flush=True)
    # TODO: setting tot_gpu_num
    tot_gpu_num = 4
    
    launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states = initialize_SHARED_CONTECT(
        tot_gpu_num=tot_gpu_num, model_paths=model_paths, 
        plan_state_group_list=plan_state_group_list,
        model_driver_worker_gpu_i=model_driver_worker_gpu_i)
    first_stage_model_ids = [exec_plan_state.exec_plan.model.model_id for exec_plan_state in launched_exec_plan_states]


    # launch the exec_plans in order
    with ProcessPoolExecutor(max_workers=len(model_paths)) as executor:
        # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
        
        # start a process for each model, no matter it is in launched_exec_plan_states or not
        # NOTE: we will use os.environ['TOT_ORDERED_GPUS'] to control the actual gpu order in each model to support reschedule
        for model_id, model_path in enumerate(model_paths):
            tasks.append(
                loop.run_in_executor(
                    executor, start_a_model_inference_child_process, query_use_vllm(model_path), '0,1,2,3', model_id, model_path
                )        
            )


        # wait for all processes finishing the preparation before initializing their LLM objects
        SHARED_CONTECT.wait_all_models_to_finish_preparation_before_init_LLM(model_ids=range(len(model_paths)))
        # start the first stage models
        SHARED_CONTECT.start_specific_models(first_stage_model_ids)


        start = time.perf_counter()
        print(f"Outer iter start time ---abs: {start}")
        SHARED_CONTECT.stop_specific_models([0])
        SHARED_CONTECT.wait_all_models_to_finish_prepare_for_reschedule([0])

        SHARED_CONTECT.set_execution_plan([2,9,2,1,2,3,0], model_ids=[0])
        SHARED_CONTECT.start_specific_models([0])





# Test individual execution plan
def main_test(
        # gen_execplans_baseline:str,
        # search_method_baseline:str,
        tot_gpu_num:int = 4,
        # max_group_seq_num: float = float('inf'),
):
    import os
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'False'
    os.environ['COLLECT_TIME_LOG'] = 'False' 

    print(f"os.environ['CUDA_VISIBLE_DEVICES'] in main_with_preemption: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    gpu_order_we_set = None
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpu_order_we_set = list(range(tot_gpu_num))
    else:
        gpu_order_we_set = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(gpu_order_we_set) == tot_gpu_num, f"gpu_order_we_set: {gpu_order_we_set} should contain {tot_gpu_num} cards"


    model_driver_worker_gpu_i: Dict[int, int] = dict()

    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4

    # loop = asyncio.get_running_loop()
    tasks = []

    # search the best model scheduling
    model_paths = [
                    'NousResearch/Llama-2-7b-hf', 
                #    'NousResearch/Llama-2-7b-chat-hf',
                # #    'NousResearch/Llama-2-13b-hf',
                # #    'NousResearch/Llama-2-70b-hf',
                #    'THUDM/chatglm3-6b',
                   ]
    # plan_state_group_list:List[List[MyExecPlanState]] = search_best_scheduling(
    #     gen_execplans_baseline,
    #     search_method_baseline,
    #     model_paths, tot_gpu_num = tot_gpu_num, 
    #     max_group_seq_num = max_group_seq_num)

    print("\n\n\n\n\nfinish searching!\n\n\n\n\n", flush=True)
    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4
    
    # launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states = initialize_SHARED_CONTECT(
    #     tot_gpu_num=tot_gpu_num, model_paths=model_paths, 
    #     plan_state_group_list=plan_state_group_list,
    #     model_driver_worker_gpu_i=model_driver_worker_gpu_i, 
    #     gpu_order_we_set=gpu_order_we_set)
    
    import ctypes
    
    SHARED_CONTECT.set_execution_plan_size(tot_gpu_num)
    counter = Array('i', [0 for i in range(len(model_paths)*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+len(model_paths))]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.started_status = [Event() for _ in range(len(model_paths))]
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(len(model_paths))])
    SHARED_CONTECT.set_execution_plan([2, 9, 2, 2, 0, 1, 2, 3], model_ids=[0])


    first_stage_model_ids = [0]

    SHARED_CONTECT.start_specific_models([0])
    start_a_model_inference_child_process(query_use_vllm(model_paths[0]), ','.join([str(i) for i in gpu_order_we_set]), 0, model_paths[0])
    return

    # # launch the exec_plans in order
    # with ProcessPoolExecutor(max_workers=len(model_paths)) as executor:
    #     # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
        
    #     # start a process for each model, no matter it is in launched_exec_plan_states or not
    #     # NOTE: we will use os.environ['TOT_ORDERED_GPUS'] to control the actual gpu order in each model to support reschedule
    #     for model_id, model_path in enumerate(model_paths):
    #         tasks.append(
    #             loop.run_in_executor(
    #                 executor, start_a_model_inference, 
    #                 query_use_vllm(model_path), ','.join([str(i) for i in gpu_order_we_set]), model_id, model_path
    #             )        
    #         )


    #     # wait for all processes finishing the preparation before initializing their LLM objects
    #     SHARED_CONTECT.wait_all_models_to_finish_preparation_before_init_LLM(model_ids=range(len(model_paths)))
    #     # start the first stage models
    #     SHARED_CONTECT.start_specific_models(first_stage_model_ids)


    #     start = time.perf_counter()
    #     print(f"Outer iter start time ---abs: {start}")

    #     pending_list = tasks
    #     model_schedule_iter = 0
    #     while len(pending_list) > 0:
    #         # repeat this loop until all models are finished
    #         print(f"a new iteration==================", flush=True)
    #         # print(f"MAIN PROCESS: {[str(plan_state) for plan_state in launched_exec_plan_states]}", flush=True)

    #         done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)

    #         # <jingzhi> For Profiling
    #         start_waiting = time.perf_counter()
    #         print(f"MAIN PROCESS: total time to launch processes (just the value of iter 0 is useful) {model_schedule_iter}: {start_waiting-start}s ---abs: {start_waiting}", flush=True)

    #         # TODO (jingzhi) try to make the finished processes release their resources
    #         print("len(done_list): ", len(done_list), "len(pending_list): ", len(pending_list))
    #         # print(f"MAIN PROCESS: next iter plans: {[str(plan_state) for plan_state in launched_exec_plan_states]}")
    #         # print(f"MAIN PROCESS: model_ids_to_stop: {model_ids_to_stop}")
    #         # print(f"MAIN PROCESS: new_launch: {[str(plan_state) for plan_state in new_launch]}")
    #         # print(f"MAIN PROCESS: candidate_exec_plan_states: {[str(plan_state) for plan_state in candidate_exec_plan_states]}")
            
    #         for task in done_list:
    #             await task


    #         # <jingzhi> For Profiling
    #         end_waiting = time.perf_counter()
    #         print(f"MAIN PROCESS: total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
    #         model_schedule_iter += 1

    #     end = time.perf_counter()
    #     print(f"total running time: {end-start}s ---abs: {end}")



















def get_tot_latency_from_log(filename: str):
    tot = 0
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # if 'exec time: ' in line:
            #     v = float(line[len('exec time: '):])
            #     tot += v
            # elif 'sample time: ' in line:
            #     v = float(line[len('sample time: '):])
            #     tot += v
            # elif 'prepInp time: ' in line:
            #     v = float(line[len('prepInp time: '):])
            #     tot += v
            # elif 'time_for_schedule: ' in line:
            #     pos = line.find('time_for_schedule: ')
            #     v = float(line[pos+len('time_for_schedule: '):])
            #     tot += v 
            
            # if 'time_per_iter:' in line:
            #     pos = line.find('time_per_iter:')
            #     pos1 = line.find(',')
            #     v = float(line[pos+len('time_per_iter:'):pos1])
            #     tot += v 
            if 'proc_output_time:' in line:
                pos = line.find('proc_output_time:')
                v = float(line[pos+len('proc_output_time:'):])
                tot += v 
    return tot






def get_schedule_setting(test_case:str):

    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    if test_case == 'general':
        # inp/out len generator functions for the general setting
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(get_model_path_list()))}
        req_num = 1000
        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
        outlen_generator = output_length_sampler.sample_out_len_for_given_model
        node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", 0, -1) \
                                      for i in range(len(get_model_path_list()))}

    elif test_case == 'map-reduce':
        # inp/out len generator functions for the map-reduce or chain summary scenario
        # map-reduce
        # 如果input sequence的长度差别很大的话，可能会导致chunk的数量差别很大，可能会有很多个LLM instance，但是每个instance的inp workload数量不一样，这样会有影响吗？
        # 试试再说吧
        # TODO: 还有一个问题，我们现在判断redundancy不会把相同的model但是不同instance看成是相同的，这样可能会导致大量redundancy。
        req_num = 10
        chunk_size = 512
        model_paths = ['NousResearch/Llama-2-13b-hf'] * (20000 // chunk_size)
        model_paths = model_paths + ['NousResearch/Llama-2-13b-hf']
        # out_edge_dict = {i:[len(model_paths)-1] for i in range(len(model_paths)-1)}
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths)-1)}
        in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = list(range(len(model_paths)-1))
        # 
        inp_generator = lambda : [512]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", i, chunk_size) \
                                      for i in range(len(model_paths)-1)}

    elif test_case == 'chain-summary':
        # # chain summary
        req_num = 1000
        chunk_size = 512
        model_paths = ['NousResearch/Llama-2-13b-hf'] * (20000 // chunk_size)
        # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
        # # out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] + list(range(i)) for i in range(len(model_paths))}
        
        inp_generator = lambda : [512]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", i, chunk_size)\
                                      for i in range(len(model_paths))}



    # # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # # search_method_baseline = 'ours' # 'naive'  'ours'
    # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # search_method_baseline = 'ours' # 'naive'  'ours'
    check_gap = 16
    sort_input = True

    return check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, req_num, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping




if __name__ == "__main__":
    print("start")
    # --------------------------------------------------------------------
    # # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # # search_method_baseline = 'ours' # 'naive'  'ours'
    gen_execplans_baseline = 'ours' # 'naive'  'ours'
    search_method_baseline = 'ours' # 'naive'  'ours'
    
    test_case = 'general' # 'general' 'map-reduce' 'chain-summary'
    check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, num_prompts, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        get_schedule_setting(test_case=test_case)
    
    asyncio.run(main_with_preemption(
        gen_execplans_baseline=gen_execplans_baseline,
        search_method_baseline=search_method_baseline,
        # 
        # support model-level pipeline parallel
        in_edge_dict_with_dummy_inp_nodes=in_edge_dict_with_dummy_inp_nodes,
        node_dataset_chunk_mapping=node_dataset_chunk_mapping,
        check_gap=check_gap, sort_input=sort_input,
        num_prompts=num_prompts, inp_generator=inp_generator, inp_merger=inp_merger, outlen_generator=outlen_generator,
        # 
        tot_gpu_num=2,
        max_group_seq_num=100))
    # # asyncio.run(main_with_preemption_debug())


    # test_search(
    #     gen_execplans_baseline=gen_execplans_baseline,
    #     search_method_baseline=search_method_baseline,
    #     # 
    #     # support model-level pipeline parallel
    #     in_edge_dict_with_dummy_inp_nodes=in_edge_dict_with_dummy_inp_nodes,
    #     node_dataset_chunk_mapping=node_dataset_chunk_mapping,
    #     check_gap=check_gap, sort_input=sort_input,
    #     num_prompts=num_prompts, inp_generator=inp_generator, inp_merger=inp_merger, outlen_generator=outlen_generator,
    #     # 
    #     tot_gpu_num=2,
    #     max_group_seq_num=100)


    # --------------------------------------------------------------------

    # main_test(
    #     tot_gpu_num=4,
    # )


'''
from schedule_multi_model import *
asyncio.run(main_with_preemption())

from schedule_multi_model import *
model_paths = get_model_path_list()
tot_gpu_num=4
gen_execplans_baseline = 'ours' # 'naive'  'ours'
search_method_baseline = 'ours' # 'naive'  'ours'
res = list()
for i in range(10):
    best_group_seq = get_best_model_schedule(
        search_method_baseline,
        gen_execplans_baseline,
        model_paths, 
        sample_config=(1, 1, -1, 0),
        trust_remote_code=True, revision=None,
        gpu_name='A100-80G', tot_gpu_num = tot_gpu_num, byte_per_gpu=80*(1024**3), 
        data_byte=2
    )
    res.append(best_group_seq.get_tot_time())


    
get_tot_latency_from_log('./Cost_Model_per_iter/baseline_tp4_Llama-2-7b-chat-hf_1.log')
get_tot_latency_from_log('./Cost_Model_per_iter/baseline_tp2_chatglm3-6b_1.log')
get_tot_latency_from_log('./test_end2end_schedule/test_2.log')




'''