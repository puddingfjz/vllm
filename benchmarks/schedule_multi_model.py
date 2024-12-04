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




from concurrent.futures import ProcessPoolExecutor, wait
import asyncio
from multiprocessing import Array, Event


from vllm.core.multimodel_scheduler import SHARED_CONTECT, LLM_COMMUNICATOR, MyManager
from vllm.sampling_params import SamplingParams
import benchmark_throughput

import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import itertools

from search_exec_plans import MyExecPlan, MyExecPlanGroupSeq, MyModelInfor, get_best_model_schedule, get_dependent_exec_plans_for_each_plan #, get_inplens
import output_length_sampler

from collections import defaultdict

# shared_counter: Array # = Array('d', [-1, -1])

import traceback
import argparse


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
        self.comp_gpus = list(comp_gpus)
    def get_comp_gpus(self):
        return self.comp_gpus

    def __str__(self) ->str:
        return f'{str(self.exec_plan)}, launched:{self.launched}, stage_i:{self.stage_i}, model_id: {self.exec_plan.model.model_id}, comp_gpus: {self.comp_gpus}'





# define the args we need
class InferenceArgs:
    """Arguments for vLLM single model inference."""
    def __init__(self, 
        model:str="huggyllama/llama-7b", 
        num_prompts: int = 1000,
        dataset: str = "ShareGPT_V3_unfiltered_cleaned_split.json",
        ignore_eos: bool = False, 
        fixed_output_len: int = None,
        # tensor_parallel_size:int=1
    ) -> None:
        self.backend: str = "vllm"
        self.dataset: str = dataset
        self.input_len: int = None
        self.output_len: int = fixed_output_len
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
        self.ignore_eos: bool = ignore_eos

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
        communicator: LLM_COMMUNICATOR, use_vllm: bool, gpus: str, shared_id: int, model: str = "huggyllama/llama-7b", 
        return_str=True, req_num=None):
    try:
        print(f"in running start_a_model_inference_child_process")
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        # TODO: some models do not support dynamic model weight loading now
        os.environ['USE_VLLM']='False'
        os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
        if use_vllm:
            os.environ['USE_VLLM']='True'
            os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False'
        
        # os.environ['RUN_MULTI_MODEL'] = 'True'
        # NOTE: the dataset, ignore_eos, and fixed_output_len does not matter here
        args = InferenceArgs(model, req_num)

        # set os.environ['CUDA_VISIBLE_DEVICES'] before importing benchmark_throughput
        # benchmark_throughput.SHARED_CONTECT.shared_setting = SHARED_CONTECT.shared_setting
        # set shared id for each model
        SHARED_CONTECT.shared_id = shared_id
        SHARED_CONTECT.communicator = communicator
        SHARED_CONTECT.return_str = return_str
        SHARED_CONTECT.tot_req_num_remained = req_num
        print(f"SHARED_CONTECT.shared_id: {SHARED_CONTECT.shared_id}")
        print(f"SHARED_CONTECT.tot_req_num_remained: {SHARED_CONTECT.tot_req_num_remained}")
        # benchmark_throughput.main(args)
        benchmark_throughput.main(args)
        print(f"MODEL PROCESS ENDS: shared_id: {SHARED_CONTECT.shared_id}", flush=True)
    except Exception as e:
        print(f"Exception in running benchmark_throughput.main(): {e}")
        print(traceback.format_exc())





# start a model for inference
def start_a_model_inference(
        communicator: LLM_COMMUNICATOR, use_vllm: bool, gpus: str, model_id: int, model: str = "huggyllama/llama-7b", 
        return_str=True, req_num=None):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    print(f"in running start_a_model_inference")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            print(f"in running start_a_model_inference 1")
            executor.submit(start_a_model_inference_child_process, communicator, use_vllm, gpus, model_id, 
                            model, return_str, req_num)
        except Exception as e:
            print(f"Exception in running start_a_model_inference: {e}")
            print(traceback.format_exc())



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

    # <jingzhi> FOR DEBUG
    if max(gpu_list) > tot_gpu_num-1:
        print(f"available_gpus:{available_gpus}, tot_gpu_num: {tot_gpu_num}, [i for i in range(tot_gpu_num) if i not in available_gpus]: {[i for i in range(tot_gpu_num) if i not in available_gpus]}")
        assert False

    # reorder gpu list according to the gpu order we set
    print(f"gpu_order_we_set: {gpu_order_we_set}, gpu_list: {gpu_list}")
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
                #    'THUDM/chatglm3-6b',
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
    # model_paths = ['NousResearch/Llama-2-7b-hf', 'NousResearch/Llama-2-7b-chat-hf']
    return model_paths



def query_use_vllm(model_path: str) -> bool:
    return True
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
        # new models for routerbench
        'meta-llama/Llama-2-70b-chat-hf': True,   
        'mistralai/Mixtral-8x7B-Instruct-v0.1': True,   
        'WizardLMTeam/WizardLM-13B-V1.2': True,   
        'meta-llama/CodeLlama-34b-Instruct-hf': True,   
        'mistralai/Mistral-7B-Instruct-v0.2': True,   
    }
    return setting_dict[model_path]






def prepare_exec_plan_states(
        plan_group_seq: MyExecPlanGroupSeq
        )->List[List[MyExecPlanState]]:
    '''
        Prepare the corresponding MyExecPlanState object for each exec plan.
        Set the ``last_exec_plan_for_the_model`` and ``need_prepare_infer_env`` attributes for each exec plan.
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




def _get_model_sys_structure_from_selected_plan_group_seq(
        plan_state_group_list: List[List[MyExecPlanState]], 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]], 
        out_edge_dict: Dict[int, List[int]],
) -> Tuple[Dict[int, int], Dict[int, MyModelInfor], Dict[int, List[int]], Dict[int, List[int]]]:
    """
        Input:
            1. in_edge_dict_with_dummy_inp_nodes: the in edge dict of the initial model system with dummy inp nodes.
            2. out_edge_dict: the out edge dict of the initial model system without dummy inp nodes.
    """


    print(f"in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"out_edge_dict: {out_edge_dict}")


    model_id_shared_id_mapping: Dict[int, int] = dict()
    shared_id: int = 0

    # 1. get the involved models (base model or fused model)
    model_dict: Dict[int, MyModelInfor] = dict()
    for plan_state_group in plan_state_group_list:
        for plan_state in plan_state_group:
            model_id = plan_state.exec_plan.model.model_id
            model_dict[model_id] = plan_state.exec_plan.model
            if model_id not in model_id_shared_id_mapping:
                model_id_shared_id_mapping[model_id] = shared_id
                shared_id += 1

    # 2. get the node mapping between the original model system and the new model system which may contain some fused models
    node_mapping: Dict[int, int] = dict()
    for model_id, model in model_dict.items():
        for ori in model.get_base_model_ids():
            node_mapping[ori] = model_id
    dummy_model_ids = np.concatenate(list(in_edge_dict_with_dummy_inp_nodes.values()))
    dummy_model_ids = dummy_model_ids[dummy_model_ids<0]
    for model_id in dummy_model_ids:
        node_mapping[model_id] = model_id

    # 3. get the new dummy in edge dict
    new_in_edge_dict_with_dummy_inp_nodes = defaultdict(list)
    for k, vs in in_edge_dict_with_dummy_inp_nodes.items():
        new_in_edge_dict_with_dummy_inp_nodes[node_mapping[k]].extend([node_mapping[v] for v in vs])

    # 4. get the new out edge dict
    new_out_edge_dict = defaultdict(list)
    for k, vs in out_edge_dict.items():
        new_out_edge_dict[node_mapping[k]].extend([node_mapping[v] for v in vs])
    

    return model_id_shared_id_mapping, model_dict, new_in_edge_dict_with_dummy_inp_nodes, new_out_edge_dict




def search_best_scheduling(
        test_case: str,
        gen_execplans_baseline:str,
        search_method_baseline:str,
        model_paths: List[str], 
        # 
        out_edge_dict: Dict[int, List[int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, 
        inp_seq_ids_dict, 
        out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]],
        inp_req_ids: Dict[int, Dict[int, List[int]]], 
        independent_srcs: Dict[int, bool],
        # inp_generator, inp_merger, outlen_generator,
        # 
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num: int = 4,
        max_group_seq_num: int = 100,
        top_k: int=100,
        similar_threshold: float=0.1,
        fully_connected_gpu_unit: int=4,
        machine_name:str='lccpu',
    )->List[List[MyExecPlanState]]:
    
    # 1. first search the best scheduling
    
    inp_generator, inp_merger, outlen_generator = _get_req_len_funcs(test_case=test_case)

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
        inp_seq_ids_dict,
        out_req_id_mapping,
        inp_req_ids, 
        independent_srcs,
        # 
        inp_generator,
        inp_merger,
        outlen_generator,
        out_edge_dict,
        sample_config=(1, 1, -1, 0),
        trust_remote_code=True, revision=None,
        gpu_name=gpu_name, tot_gpu_num = tot_gpu_num, byte_per_gpu=byte_per_gpu, 
        data_byte=2,
        max_group_seq_num=max_group_seq_num,
        top_k=top_k,
        similar_threshold=similar_threshold,
        fully_connected_gpu_unit=fully_connected_gpu_unit,
        machine_name=machine_name,
    )


    # 2. convert best_group_seq to exec plan state list
    plan_state_group_list = prepare_exec_plan_states(best_group_seq)
    
    return plan_state_group_list





def initialize_SHARED_CONTECT_not_support_fused_models(
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
        NOTE:
            1. this version does not support the case where there are fused models in the model system.
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





def initialize_SHARED_CONTECT(
        tot_gpu_num: int,
        # model_paths: List[str], 
        check_gap: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        gpu_order_we_set: List[int],
        model_id_shared_id_mapping: Dict[int, int],
        new_out_edge_dict: Dict[int, List[int]],
        sampling_args_dict: Dict[int, Tuple[bool, int, int]]
    ) -> Tuple[List[MyExecPlanState], int, List[MyExecPlanState]]:
    '''
        Update: (1) SHARED_CONTECT events, shared_finish_status, shared_setting
                (2) call SHARED_CONTECT.start_specific_models()
        Output: (1) launched_exec_plan_states; (2) new target stage i; (3) candidate_exec_plan_states
    '''

    import ctypes

    new_model_num = len(model_id_shared_id_mapping)
    
    SHARED_CONTECT.set_execution_plan_size(tot_gpu_num)
    counter = Array('i', [0 for i in range(new_model_num*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+new_model_num)]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.started_status = [Event() for _ in range(new_model_num)]
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(new_model_num)])
    # add check_out_gaps
    check_out_gaps = Array('i', [int(1e9)]*new_model_num) # 'd' is for double
    SHARED_CONTECT.check_out_gaps = check_out_gaps
    SHARED_CONTECT.check_in_gap = check_gap
    SHARED_CONTECT.sampling_args_dict = sampling_args_dict

    
    # set the initial execution plan
    available_gpus: List[int] = list(range(tot_gpu_num))
    launched_exec_plan_states: List[MyExecPlanState] = plan_state_group_list[0]
    # sort launched_exec_plan_states by tp size:
    launched_exec_plan_states = sorted(launched_exec_plan_states, key=lambda plan_state: plan_state.exec_plan.num_worker)
    for exec_plan_state in launched_exec_plan_states:
        
        exec_plan = exec_plan_state.exec_plan
        # exec_plan_state.set_comp_gpus(available_gpus[:exec_plan.num_worker])
        # data parallel
        # NOTE: the gpu assignment policy for data parallel
        exec_plan_state.set_comp_gpus(available_gpus[:exec_plan.num_worker*exec_plan.dp_size])
        
        setting = get_exec_settings_from_exec_plans(
            exec_plan=exec_plan, available_gpus=available_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
        # SHARED_CONTECT.set_execution_plan(setting, model_ids=[exec_plan.model.model_id])
        # NOTE: here we set setting for shared id, instead of model id
        SHARED_CONTECT.set_execution_plan(setting, shared_ids=[model_id_shared_id_mapping[exec_plan.model.model_id]])
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



    # need to set check out gap 
    set_check_in_out_gap(
        curr_stage_plan_states=launched_exec_plan_states, check_gap=check_gap, new_out_edge_dict=new_out_edge_dict,
        model_id_shared_id_mapping=model_id_shared_id_mapping)

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
        model_id_shared_id_mapping: Dict[int, int],
    )-> bool:
    finished = [i for i in plan_state_group_list[stage_i] \
                if SHARED_CONTECT.query_finish_status(model_id_shared_id_mapping[i.exec_plan.model.model_id])]
    return len(finished) > 0

def _get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        model_id_shared_id_mapping: Dict[int, int],
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
        if SHARED_CONTECT.query_finish_status(model_id_shared_id_mapping[plan_state.exec_plan.model.model_id]):
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
        if SHARED_CONTECT.query_finish_status(model_id_shared_id_mapping[plan_state.exec_plan.model.model_id]):
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
            if _has_model_finished(plan_state_group_list, target_stage_i, model_id_shared_id_mapping):
                # to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i = \
                to_launch, new_candidate_exec_plan_states, new_target_stage_i = \
                    _get_the_next_round_exec_plan_schedule(
                    to_launch, new_candidate_exec_plan_states,
                    new_target_stage_i,
                    tot_gpu_num,
                    plan_state_group_list,
                    model_driver_worker_gpu_i,
                    model_id_shared_id_mapping,
                )
    
    # return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i
    return to_launch, new_candidate_exec_plan_states, new_target_stage_i




def _adjust_comp_gpus_for_current_launched_exec_plans(
        launched_exec_plan_states: List[MyExecPlanState],
        new_launch: List[MyExecPlanState],
        tot_gpu_num: int,
        fully_connected_gpu_unit: int)->List[MyExecPlanState]:
    """
        INPUT:
            fully_connected_gpu_unit: the number of gpus that are fully connected, 
                e.g., 2 if 1 gpu is only connected with 1 other gpu with NV-links;
                      4 if 1 gpu is connected by 3 other gpus with NV-links.
        OUTPUT: 
            the plan_states that will need reload model weights.
    """
    launched_exec_plan_states = sorted(launched_exec_plan_states, key=lambda plan_state: plan_state.exec_plan.num_worker)
    cand_gpu_groups = np.arange(tot_gpu_num).reshape((-1, fully_connected_gpu_unit))
    # stores the cost to reload models assigned to each gpu group
    cost_to_clean_models = np.asarray([0]*cand_gpu_groups.shape[0])
    plan_state_to_reassign_gpus: List[MyExecPlanState] = list()
    # 
    # 1. get the gpu groups that should be kept and the groups that will be considered in reassignment;
    #    get the plan_states that need to ressign gpus to (i.e., plan_state_to_reassign_gpus)
    for plan_state in launched_exec_plan_states:
        if plan_state not in new_launch:
            gpus = plan_state.get_comp_gpus()
            gpus = np.asarray(gpus)[:plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size]
            print(f"model_id: {plan_state.exec_plan.model.model_id}, gpus: {gpus}", flush=True)
            gpus, counts = np.unique(gpus // fully_connected_gpu_unit, return_counts=True)
            if plan_state.exec_plan.num_worker >= fully_connected_gpu_unit:
                # NOTE: we assume the gpus assigned to the plan_state must have been the best choice for it    
                # check the gpu groups that this model fully occupies
                assert (counts == fully_connected_gpu_unit).all()
                cost_to_clean_models[gpus] = 1e9
                print(f"keep gpu assignment: model_id: {plan_state.exec_plan.model.model_id}", flush=True)
            else:
                cost_to_clean_models[gpus] = cost_to_clean_models[gpus] + counts
                plan_state_to_reassign_gpus.append(plan_state)
        else:
            plan_state_to_reassign_gpus.append(plan_state)
    # 
    # 2. reassign gpus for plan_states in plan_state_to_reassign_gpus
    sorted_gpu_group_ids = np.argsort(cost_to_clean_models)
    sorted_gpu_group_ids = sorted_gpu_group_ids[ cost_to_clean_models[sorted_gpu_group_ids]<1e9 ]
    cand_gpus = np.concatenate(cand_gpu_groups[sorted_gpu_group_ids])
    extra_new_launch = list()

    # sort the plan states so that those not in new launch can be checked before others
    plan_state_to_reassign_gpus = sorted(
        plan_state_to_reassign_gpus, 
        key=lambda plan_state: (plan_state.exec_plan.num_worker, plan_state not in new_launch), reverse=True)

    for plan_state in plan_state_to_reassign_gpus:
        comp_gpu_num = plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size
        if plan_state.exec_plan.num_worker >= fully_connected_gpu_unit:
            gpus = cand_gpus[:comp_gpu_num]
            cand_gpus = cand_gpus[comp_gpu_num:]
            plan_state.set_comp_gpus(gpus)
            if plan_state not in new_launch:
                extra_new_launch.append(plan_state)
            print(f"model_id: {plan_state.exec_plan.model.model_id}, reassign gpus: {gpus}, cand_gpus: {cand_gpus}", flush=True)
        else:
            if plan_state not in new_launch:
                # check whether we do not need move this model
                gpus = plan_state.get_comp_gpus()[:comp_gpu_num]
                if set(gpus).issubset(cand_gpus):
                    # if (min(gpus) // fully_connected_gpu_unit) == (max(gpus) // fully_connected_gpu_unit):
                    num_worker = plan_state.exec_plan.num_worker
                    dp_size = plan_state.exec_plan.dp_size
                    gpu_for_dps = [gpus[dp_i*num_worker:(dp_i+1)*num_worker] for dp_i in range(dp_size)]
                    if False not in [(min(i) // fully_connected_gpu_unit) == (max(i) // fully_connected_gpu_unit) for i in gpu_for_dps]:
                        # we do not need to change its assigned gpus
                        cand_gpus = [_ for _ in cand_gpus if _ not in gpus]
                        print(f"model_id: {plan_state.exec_plan.model.model_id}, keep gpus: {gpus}, cand_gpus: {cand_gpus}", flush=True)
                        continue
                extra_new_launch.append(plan_state)
                # 
            gpus = cand_gpus[:comp_gpu_num]
            cand_gpus = cand_gpus[comp_gpu_num:]
            plan_state.set_comp_gpus(gpus)
            print(f"model_id: {plan_state.exec_plan.model.model_id}, reassign gpus: {gpus}, cand_gpus: {cand_gpus}", flush=True)
    #   
    # 
    return extra_new_launch


        


def get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        model_id_shared_id_mapping: Dict[int, int],
        fully_connected_gpu_unit: int,
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], List[int], List[MyExecPlanState], int]:

    # 1. get the new launch plan
    to_launch, new_candidate_exec_plan_states, new_target_stage_i = \
        _get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states, candidate_exec_plan_states,
        target_stage_i,
        tot_gpu_num,
        plan_state_group_list,
        model_driver_worker_gpu_i,
        model_id_shared_id_mapping,
    )

    # 2. get model_ids_to_stop and new_launch
    launched_exec_plans = [(i.exec_plan.model.model_id,i.exec_plan.get_key()) for i in launched_exec_plan_states]
    to_launch_exec_plans = [(i.exec_plan.model.model_id,i.exec_plan.get_key()) for i in to_launch]
    
    new_launch = [i for i in to_launch if (i.exec_plan.model.model_id,i.exec_plan.get_key()) not in launched_exec_plans]
    model_ids_to_stop = [i.exec_plan.model.model_id for i in launched_exec_plan_states 
                         if (i.exec_plan.model.model_id,i.exec_plan.get_key()) not in to_launch_exec_plans]


    # 3. change the comp gpu setting if the GPUs are not fully connected with NV-link.
    extra_new_launch = _adjust_comp_gpus_for_current_launched_exec_plans(
        to_launch, new_launch, tot_gpu_num, fully_connected_gpu_unit)
    new_launch = new_launch + extra_new_launch
    model_ids_to_stop = model_ids_to_stop + [i.exec_plan.model.model_id for i in extra_new_launch]

    print(f"ORI launched_exec_plan_states: {[str(plan_state) for plan_state in launched_exec_plan_states]}")
    print(f"extra_new_launch: {[str(plan_state) for plan_state in extra_new_launch]}")
    print(f"model_ids_to_stop: {model_ids_to_stop}")

    return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i




def start_exec_plans(
        new_launch: List[MyExecPlanState], tot_gpu_num: int, gpu_order_we_set: List[int],
        model_id_shared_id_mapping: Dict[int, int]):
    try:
        for exec_plan_state in new_launch:
            
            exec_plan = exec_plan_state.exec_plan
            assert len(exec_plan_state.comp_gpus) == (exec_plan.num_worker * exec_plan.dp_size)
            
            print(f"before call get_exec_settings_from_exec_plans: available_gpus: {exec_plan_state.comp_gpus}, tot_gpu_num: {tot_gpu_num}, gpu_order_we_set: {gpu_order_we_set}")

            setting = get_exec_settings_from_exec_plans(
                exec_plan=exec_plan, available_gpus=exec_plan_state.comp_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
            shared_id = model_id_shared_id_mapping[exec_plan.model.model_id]
            SHARED_CONTECT.set_execution_plan(setting, shared_ids=[shared_id])

            exec_plan_state.launched = True
            SHARED_CONTECT.start_specific_models([shared_id])
    except Exception as e:
        print(f"Exception in start_exec_plans: {e}")
        print(f"exec plan comp gpus: {[exec_plan_state.comp_gpus for exec_plan_state in new_launch]}, tot_gpu_num: {tot_gpu_num}, gpu_order_we_set: {gpu_order_we_set}")
        print(f"tot_gpu_num: {tot_gpu_num}, gpu_order_we_set: {gpu_order_we_set}")
        print(traceback.format_exc())
        assert False






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




def _get_dummy_requests():
    import json
    with open(f"./my_dummy_requests/my_dummy_requests.json", 'r') as f:
        dataset = json.load(f)
    return dataset


# def _init_dummy_requests(inp_lens: List[int]):
#     import json
#     with open(f"./my_dummy_requests/my_dummy_requests.json", 'w') as f:
#         requests = ["hi" * (input_len - 1) for input_len in inp_lens]
#         json.dump(requests, f)


def _init_dummy_requests(
        inp_lens: List[int],
        sampled_inps: Union[List[List[int]], List[str]]=None, 
        model_path:str=None, ):
    import json
    if model_path == None:
        with open(f"./my_dummy_requests/my_dummy_requests.json", 'w') as f:
            requests = ["hi" * (input_len - 1) for input_len in inp_lens]
            json.dump(requests, f)    
    else:
        with open(f"./my_dummy_requests/my_dummy_requests.json", 'w') as f:
            # convert the token ids back to str
            # NOTE: TODO: 貌似把token ids转回str非常复杂，所以干脆直接在文件里写入token ids了。
            assert sampled_inps != None
            json.dump(sampled_inps, f)


def init_prompts_for_the_model_system(
        communicator: LLM_COMMUNICATOR,
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]], 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]], 
        num_prompts: int, 
        inp_seq_ids_dict: Dict[int, List[int]],
        model_path:str):
    """
        Sample input dataset for the model system.
        INPUT:
            node_dataset_chunk_mapping: {model_id: (dataset_name, chunk_id, chunk_size)}
            independent_srcs: stores whether each model's different input sources are independent, or need to be concatenated to be an input.
        NOTE: we also set the total number of requests each model needs to do inference for.
        OUTPUT:
            req_num_dict: the number of req to answer for each model.
        Modify:
            call communicator.add_seqs to add the input requests to the corresponding dummy models.
    """

    # simply use the tokenizer of llama2 7b to check the lengths of the prompts
    args = InferenceArgs(model=model_path, num_prompts=num_prompts)

    from transformers import AutoTokenizer
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    datasets = set([v[0] for v in node_dataset_chunk_mapping.values()])
    dataset_dict = dict()
    print(f"datasets: {datasets}")
    for dataset in datasets:
        if dataset == None:
            requests = _get_dummy_requests()
            inp_prompts = [(i, req) for i, req in enumerate(requests)]
            dataset_dict[dataset] = inp_prompts
            
            print(f"prompt_lens: {[len(req) for _, req in dataset_dict[None]]}")
            
            continue
        requests = benchmark_throughput.sample_requests(
            dataset, args.num_prompts, tokenizer,args.output_len, random_seed=args.seed)
        inp_prompts = [(i, req[0]) for i, req in enumerate(requests)]
        dataset_dict[dataset] = inp_prompts

    req_num_dict = defaultdict(int)
    prompts_dict = dict()
    for model_id, (dataset, chunk_id, chunk_size) in node_dataset_chunk_mapping.items():
        inp_prompts = dataset_dict[dataset]
        to_add = inp_prompts
        # print(f"model_id, (dataset, chunk_id, chunk_size): {model_id, (dataset, chunk_id, chunk_size)}, to_add[0]: {to_add[0]}")

        print(f"model_id, (dataset, chunk_id, chunk_size): {model_id, (dataset, chunk_id, chunk_size)} prompt_lens: {[len(req) for _, req in dataset_dict[dataset]]}")


        if dataset != None:
            if chunk_size > 0:
                to_add = [(i, req[chunk_id*chunk_size:(chunk_id+1)*chunk_size]) for i, req in inp_prompts if (len(req)>chunk_id*chunk_size)]
        else:
            if chunk_size > 0:
                # to_add = [(i, req[max(0,(chunk_id*chunk_size-1)*2):((chunk_id+1)*chunk_size-1)*2]) for i, req in inp_prompts if (len(req)>(chunk_id*chunk_size-1)*2)]
                to_add = [(i, req[chunk_id*chunk_size:(chunk_id+1)*chunk_size]) for i, req in inp_prompts if (len(req)>chunk_id*chunk_size)]

        # communicator.add_seqs(model_id, to_add)
        if model_id not in inp_seq_ids_dict:
            prompts_dict[model_id] = to_add
            req_num_dict[model_id] = len(to_add)
        else:
            prompts_dict[model_id] = [to_add[i] for i in inp_seq_ids_dict[model_id]] # to_add
            req_num_dict[model_id] = len(inp_seq_ids_dict[model_id]) # len(to_add)
    

    print(f"req_num_dict: {req_num_dict}", flush=True)
    # print(f"to_add: {to_add}")

    # set the req number for each non-dummy model node
    tot_node_num = len(req_num_dict) + len(in_edge_dict_with_dummy_inp_nodes)
    visited = list(req_num_dict.keys())
    
    print(f"tot_node_num: {tot_node_num}, visited: {visited}", flush=True)

    while len(visited) < tot_node_num:
        for tgt, srcs in in_edge_dict_with_dummy_inp_nodes.items():
            print(f"tgt, srcs: {tgt, srcs}", flush=True)
            if set(srcs).issubset(visited):
                # if independent_srcs[tgt]:
                #     req_num_dict[tgt] = sum([req_num_dict[src] for src in srcs])
                # else:
                #     req_num_dict[tgt] = min([req_num_dict[src] for src in srcs])
                req_num_dict[tgt] = len(inp_seq_ids_dict[tgt])
                visited.append(tgt)
                print(f"visited: {visited}", flush=True)

    
    ungened_out_req_nums = req_num_dict.copy()

    # send the req_num_dict to the communicator
    # set the unavailable_req_num (i.e., unavailable inp req num) for the dummy inp nodes to 0
    for model_id in node_dataset_chunk_mapping:
        req_num_dict[model_id] = 0
    communicator.init_unavailable_req_nums_and_ungened_out_req_nums(req_num_dict, ungened_out_req_nums)

    # send the prompts of the dummy inp nodes (i.e., dummy inp nodes' outputs) to the communicator 
    for model_id, to_add in prompts_dict.items():

        # convert token ids to strs
        # TODO: 暂时先这么写，但是tokenizer.decode还有参数需要完善
        print(f"model_id: {model_id}")
        # print(to_add[0])
        if not isinstance(to_add[0][1], str):
            to_add = [(req_i, tokenizer.decode(token_ids)) for req_i, token_ids in to_add]

        communicator.add_seqs(model_id, to_add)

    return req_num_dict
        
    




def get_return_str_list_version(out_edge_dict: Dict[int, List[int]], model_id: int, model_paths: List[str])->bool:
    outs = out_edge_dict[model_id]
    tgt = model_paths[model_id]
    return (True in [tgt != model_paths[out] for out in outs])


def get_return_str(
        new_out_edge_dict: Dict[int, List[int]], model_id: int, model_path_dict: Dict[int, str],
        # new_in_edge_dict_with_dummy_nodes: Dict[int, List[int]], 
        # independent_srcs: Dict[int,bool]
        )->bool:
    """
        NOTE: 1. we also need to consider the other inputs of the output nodes of model_id, because different input sources 
            may need to be concatenated. 
            Currently, we return str if two input sources need to be concat, even if they use the same tokenizer.
            # TODO: 这个地方还要改
            2. each base model has its own ``return_str``. [暂时还没有实现这一点]
    """
    return True
    outs = new_out_edge_dict[model_id]
    tgt = model_path_dict[model_id]
    return (True in [tgt != model_path_dict[out] for out in outs])


def set_check_in_out_gap_not_support_fused_model(
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
        





def set_check_in_out_gap(
        curr_stage_plan_states: List[MyExecPlanState], 
        check_gap: int, new_out_edge_dict: Dict[int, List[int]],
        model_id_shared_id_mapping: Dict[int, int]):
    """
        If a model has no input model in this stage, check_in_gap is 1e9;
        If a model has no output model in this stage, check_out_gap is 1e9.
        OUTPUT:
            SHARED_CONTECT.check_in (deleted), SHARED_CONTECT.check_in_gap, SHARED_CONTECT.check_out_gap
        NOTE:
            all the models share the same fixed SHARED_CONTECT.check_in_gap;
            different models have different SHARED_CONTECT.check_out_gap which may change over stages
            1. support fused models in the model system.
    """

    model_ids = [plan_state.exec_plan.model.model_id for plan_state in curr_stage_plan_states]
    for plan_state in curr_stage_plan_states:
        model_id = plan_state.exec_plan.model.model_id
        out_model_ids = new_out_edge_dict[model_id]
        shared_id = model_id_shared_id_mapping[model_id]
        if len(set(out_model_ids).intersection(model_ids)) > 0:
            SHARED_CONTECT.check_out_gaps[shared_id] = check_gap
        else:
            SHARED_CONTECT.check_out_gaps[shared_id] = int(1e9)




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



def _search_best_scheduling_with_another_process(
        test_case:str,
        gen_execplans_baseline,
        search_method_baseline,
        model_paths, 
        # 
        out_edge_dict,
        check_gap, sort_input,
        num_prompts, 
        inp_seq_ids_dict, 
        out_req_id_mapping, inp_req_ids, independent_srcs,
        # 
        gpu_name,
        byte_per_gpu,
        tot_gpu_num, 
        max_group_seq_num,
        top_k,
        similar_threshold,
        fully_connected_gpu_unit,
        machine_name,
):
    print(f"in running _search_best_scheduling_with_another_process")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(
                search_best_scheduling, 
                    test_case,
                    gen_execplans_baseline,
                    search_method_baseline,
                    model_paths, 
                    # 
                    out_edge_dict,
                    check_gap, sort_input,
                    num_prompts, 
                    inp_seq_ids_dict, 
                    out_req_id_mapping, inp_req_ids, independent_srcs,
                    # inp_generator, inp_merger, outlen_generator,
                    # 
                    gpu_name,
                    byte_per_gpu,
                    tot_gpu_num, 
                    max_group_seq_num,
                    top_k,
                    similar_threshold,
                    fully_connected_gpu_unit,
                    machine_name)
            done, not_done = wait([future])
            plan_state_group_list = list(done)[0].result()
            return plan_state_group_list

        except Exception as e:
            print(f"Exception in running start_a_model_inference: {e}")
            print(traceback.format_exc())



async def main_with_preemption(
        test_case:str,
        model_paths:List[str],
        gen_execplans_baseline:str,
        search_method_baseline:str,
        # 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]],
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, 
        sampling_args_dict: Dict[int, SamplingParams],
        # 
        inp_seq_ids_dict, 
        inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs,
        # 
        inp_generator, inp_merger, outlen_generator,
        # 
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num:int = 4,
        max_group_seq_num: float = float('inf'),
        top_k: float = float('inf'),
        similar_threshold: float=0.1,
        fully_connected_gpu_unit: int = 4,
        machine_name: str='lccpu',
):
    
    print(f"fully_connected_gpu_unit: {fully_connected_gpu_unit}")

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
    # NOTE: we obtain the model paths from the input
    # model_paths = get_model_path_list()
    # convert the in_edge_dict to out_edge_dict
    out_edge_dict = get_out_edge_dict_from_in_edge_dict_with_inp_nodes(in_edge_dict_with_dummy_inp_nodes)
    

    # plan_state_group_list:List[List[MyExecPlanState]] = search_best_scheduling(
    plan_state_group_list:List[List[MyExecPlanState]] = _search_best_scheduling_with_another_process(
        test_case,
        gen_execplans_baseline,
        search_method_baseline,
        model_paths, 
        # 
        out_edge_dict,
        check_gap, sort_input,
        num_prompts, 
        inp_seq_ids_dict, 
        out_req_id_mapping, inp_req_ids, independent_srcs,
        # inp_generator, inp_merger, outlen_generator,
        # 
        gpu_name,
        byte_per_gpu,
        tot_gpu_num = tot_gpu_num, 
        max_group_seq_num = max_group_seq_num,
        top_k = top_k,
        similar_threshold=similar_threshold,
        fully_connected_gpu_unit=fully_connected_gpu_unit,
        machine_name=machine_name)
    

    
    # # TODO: <jingzhi> FOR DEBUG
    # return

    # get the NEW model system STRUCTURE from the ``plan_state_group_list``
    model_id_shared_id_mapping, model_dict, new_in_edge_dict_with_dummy_inp_nodes, new_out_edge_dict = \
        _get_model_sys_structure_from_selected_plan_group_seq(
            plan_state_group_list, in_edge_dict_with_dummy_inp_nodes, out_edge_dict,)

    # the number of models in the new model system topo (may contain fused models)
    new_model_num = len(model_id_shared_id_mapping)
    new_model_path_dict = {model_id: model.model_path for model_id, model in model_dict.items()}

    print(f"\nnew_in_edge_dict_with_dummy_inp_nodes: {new_in_edge_dict_with_dummy_inp_nodes}")
    print(f"new_out_edge_dict: {new_out_edge_dict}")
    print(f"model_id_shared_id_mapping: {model_id_shared_id_mapping}")
    print(f"new_model_path_dict: {new_model_path_dict}")

    print("\n\n\n\n\nfinish searching!\n\n\n\n\n", flush=True)
    # TODO: setting tot_gpu_num
    # tot_gpu_num = 4
    

    launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states = initialize_SHARED_CONTECT(
        tot_gpu_num=tot_gpu_num, # model_paths=model_paths, 
        check_gap=check_gap,
        plan_state_group_list=plan_state_group_list,
        model_driver_worker_gpu_i=model_driver_worker_gpu_i, 
        gpu_order_we_set=gpu_order_we_set,
        model_id_shared_id_mapping=model_id_shared_id_mapping,
        new_out_edge_dict=new_out_edge_dict,
        sampling_args_dict=sampling_args_dict,)
    first_stage_model_ids = [exec_plan_state.exec_plan.model.model_id for exec_plan_state in launched_exec_plan_states]




    print(f"\nTIMESTAMP 1: {time.perf_counter()}\n")
    time_lists: List[float] = list()

    with MyManager() as manager:

        print(f"\nTIMESTAMP 2: {time.perf_counter()}\n")

        shared_id_2_base_model_ids_dict = {model_id_shared_id_mapping[model_id]:model.get_base_model_ids() for model_id, model in model_dict.items()}
        communicator: LLM_COMMUNICATOR = manager.Communicator(
            new_model_num, # len(model_paths), 
            in_edge_dict_with_dummy_inp_nodes,
            shared_id_2_base_model_ids_dict,
            inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs,
            )

        # set inputs for dummy inp nodes in the system
        # NOTE: stores the req num of base models (for fused models, store req num for the base models inside)
        # TODO: 这里的model_paths[0]也还需要修改
        base_req_num_dict = init_prompts_for_the_model_system(communicator, node_dataset_chunk_mapping, in_edge_dict_with_dummy_inp_nodes,
                                                         num_prompts, inp_seq_ids_dict, model_path=model_paths[0])

        print(f"base_req_num_dict: {base_req_num_dict}")


        print(f"\nTIMESTAMP 3: {time.perf_counter()}\n")

        # launch the exec_plans in order
        # with ProcessPoolExecutor(max_workers=len(model_paths)) as executor:
        with ProcessPoolExecutor(max_workers=new_model_num) as executor:

            print(f"\nTIMESTAMP 4: {time.perf_counter()}\n")

            # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
            
            # start a process for each model, no matter it is in launched_exec_plan_states or not
            # NOTE: we will use os.environ['TOT_ORDERED_GPUS'] to control the actual gpu order in each model to support reschedule
            # for model_id, model_path in enumerate(model_paths):
            #     tasks.append(
            #         loop.run_in_executor(
            #             executor, start_a_model_inference, 
            #             communicator, query_use_vllm(model_path), ','.join([str(i) for i in gpu_order_we_set]), model_id, model_path, 
            #             get_return_str(out_edge_dict=out_edge_dict, model_id=model_id, model_paths=model_paths),
            #             req_num_dict[model_id],
            #         )        
            #     )

            # for model_id, model_path in enumerate(model_paths):
            for model_id, model_path in new_model_path_dict.items():
                print(f"init process for {model_id, model_path}")
                shared_id = model_id_shared_id_mapping[model_id]
                tot_req_num = sum([base_req_num_dict[base_model_id] for base_model_id in model_dict[model_id].get_base_model_ids()])
                tasks.append(
                    loop.run_in_executor(
                        executor, start_a_model_inference, 
                        communicator, query_use_vllm(model_path), ','.join([str(i) for i in gpu_order_we_set]), shared_id, model_path, 
                        get_return_str(new_out_edge_dict=new_out_edge_dict, model_id=model_id, model_path_dict=new_model_path_dict),
                        tot_req_num,
                    )        
                )



            print(f"\nTIMESTAMP 5: {time.perf_counter()}\n")


            # wait for all processes finishing the preparation before initializing their LLM objects
            # SHARED_CONTECT.wait_all_models_to_finish_preparation_before_init_LLM(model_ids=range(len(model_paths)))
            SHARED_CONTECT.wait_all_models_to_finish_preparation_before_init_LLM(shared_ids=range(new_model_num))
            # start the first stage models
            # SHARED_CONTECT.start_specific_models(first_stage_model_ids)
            print(f"[model_id_shared_id_mapping[_] for _ in first_stage_model_ids]: {[model_id_shared_id_mapping[_] for _ in first_stage_model_ids]}")
            SHARED_CONTECT.start_specific_models([model_id_shared_id_mapping[_] for _ in first_stage_model_ids])


            start = time.perf_counter()
            print(f"Outer iter start time ---abs: {start}")
            time_lists.append(start)

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
                time_lists.append(start_waiting)


                # 1. get models that need to be stopped
                try:
                    launched_exec_plan_states, candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i = \
                        get_the_next_round_exec_plan_schedule(
                            launched_exec_plan_states, candidate_exec_plan_states,
                            new_target_stage_i,
                            tot_gpu_num, plan_state_group_list,
                            model_driver_worker_gpu_i,
                            model_id_shared_id_mapping,
                            fully_connected_gpu_unit,
                        )
                except Exception as e:
                    print(f"Exception in running benchmark_throughput.main(): {e}")
                    print(traceback.format_exc())
                
                print(f"new_launch: {new_launch}")
                print(f"model_ids_to_stop: {model_ids_to_stop}")
                
                # for plan_state in launched_exec_plan_states:
                #     print(f"running plan_state info: {plan_state.exec_plan.model.model_name, plan_state.exec_plan.get_key(), plan_state.comp_gpus}")


                # 2. stop the models
                # SHARED_CONTECT.stop_specific_models(model_ids_to_stop)
                SHARED_CONTECT.stop_specific_models([model_id_shared_id_mapping[_] for _ in model_ids_to_stop])

                # TODO (jingzhi) try to make the finished processes release their resources
                print(len(done_list), len(pending_list))
                print(f"MAIN PROCESS: next iter plans: {[str(plan_state) for plan_state in launched_exec_plan_states]}")
                print(f"MAIN PROCESS: model_ids_to_stop: {model_ids_to_stop}")
                print(f"MAIN PROCESS: new_launch: {[str(plan_state) for plan_state in new_launch]}")
                print(f"MAIN PROCESS: candidate_exec_plan_states: {[str(plan_state) for plan_state in candidate_exec_plan_states]}")
                
                for task in done_list:
                    await task

                # 3. wait for model finish preparing for rescheduling            
                # SHARED_CONTECT.wait_all_models_to_finish_prepare_for_reschedule(model_ids_to_stop)
                SHARED_CONTECT.wait_all_models_to_finish_prepare_for_reschedule([model_id_shared_id_mapping[_] for _ in model_ids_to_stop])
                
                # 4. start newly launched models
                set_check_in_out_gap(
                    curr_stage_plan_states=launched_exec_plan_states, check_gap=check_gap, new_out_edge_dict=new_out_edge_dict,
                    model_id_shared_id_mapping=model_id_shared_id_mapping)
                start_exec_plans(new_launch, tot_gpu_num, gpu_order_we_set=gpu_order_we_set,
                                 model_id_shared_id_mapping=model_id_shared_id_mapping)


                # <jingzhi> For Profiling
                end_waiting = time.perf_counter()
                print(f"MAIN PROCESS: total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
                model_schedule_iter += 1

            end = time.perf_counter()
            print(f"total running time: {end-start}s ---abs: {end}")

            time_lists.append(end)
            print(f"all stage times: timestamps: {time_lists}")
            print(f"all stage times: time lengths: {np.diff(time_lists).tolist()}")












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





def _get_document_prompts(
        dataset_path: str, model_path: str, num_requests: int
        ) -> List[List[int]]:
    """
        NOTE: 
            1. Currently, we do not sort the input chunks in this case.
            2. The dataset in this function only has prompts, i.e., no output texts.
        Output:
            1. list of prompt token ids
    """
    from benchmark_throughput import get_dataset
    import random
    dataset = get_dataset(dataset_path=dataset_path)

    # simply use the tokenizer of llama2 7b to check the lengths of the prompts
    args = InferenceArgs(model=model_path, num_prompts=1)

    random.seed(args.seed)

    from transformers import AutoTokenizer
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    # Tokenize the prompts and completions.
    prompts = dataset
    prompt_token_ids = tokenizer(prompts).input_ids

    tokenized_dataset = prompt_token_ids

    # Filter out too long sequences.
    filtered_dataset: List[List[int]] = tokenized_dataset

    # Sample the requests.
    # <jingzhi> make sample size be ``min(num_requests, len(filtered_dataset))''
    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))

    # if os.environ['SORT_REQS'] == 'True':
    #     sampled_requests = sorted(sampled_requests, key=lambda x: x[1], reverse=True)


    print(f"tot_tokens: {sum([len(x) for x in sampled_requests])}, tot_context_lens: {sum([(len(x)-1)*len(x)/2 for x in sampled_requests])}")


    return sampled_requests




# directly get req lengths 
# get seqs with specific seq ids
def get_inplens(req_num: int, model_path: str, inp_seq_ids: List[int]):
    import json
    inp_lens = list()
    with open("./my_dummy_requests/my_dummy_requests.json", 'r') as file:
        prompts = json.load(file)
        args = InferenceArgs(model=model_path, num_prompts=req_num)
        from transformers import AutoTokenizer
        # Sample the requests.
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        prompt_token_ids = tokenizer(prompts).input_ids
        inp_lens = [len(prompt) for prompt in prompt_token_ids]
    
    assert len(inp_lens) == req_num
    print(f"inp_seq_ids:{inp_seq_ids}")
    return list(np.asarray(inp_lens)[inp_seq_ids])




def get_inplens_router_bench(req_num: int, model_path: str, inp_seq_ids: List[int]):
    import json
    inp_lens = list()
    with open('/ssddata/jingzhi/vLLM/vllm/benchmarks/router_bench_not_multiple_choice_question_dataset.json', 'r') as f:
    # with open('/ssddata/jingzhi/vLLM/vllm/benchmarks/router_bench_multiple_choice_question_dataset.json', 'r') as f:
        prompt_dict = json.load(f)
        prompts = prompt_dict[model_path]
        # 
        args = InferenceArgs(model=model_path, num_prompts=req_num)
        from transformers import AutoTokenizer
        # Sample the requests.
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        prompt_token_ids = tokenizer(prompts).input_ids
        inp_lens = [len(prompt) for prompt in prompt_token_ids]
    

    print(f"len(inp_lens):{len(inp_lens)}")
    return inp_lens


def get_inplens_router_bench_MCQ(req_num: int, model_path: str, inp_seq_ids: List[int]):
    import json
    inp_lens = list()
    # with open('/ssddata/jingzhi/vLLM/vllm/benchmarks/router_bench_not_multiple_choice_question_dataset.json', 'r') as f:
    with open('/ssddata/jingzhi/vLLM/vllm/benchmarks/router_bench_multiple_choice_question_dataset.json', 'r') as f:
        prompt_dict = json.load(f)
        prompts = prompt_dict[model_path]
        # 
        args = InferenceArgs(model=model_path, num_prompts=req_num)
        from transformers import AutoTokenizer
        # Sample the requests.
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        prompt_token_ids = tokenizer(prompts).input_ids
        inp_lens = [len(prompt) for prompt in prompt_token_ids]
    

    print(f"len(inp_lens):{len(inp_lens)}")
    return inp_lens




def _get_req_len_funcs(test_case:str, version:str=None):
    inp_generator, inp_merger, outlen_generator = None, None, None
    if test_case == 'router':
        inp_generator = get_inplens_router_bench        
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
        outlen_generator = lambda model_name, inp_lens: np.minimum(8192, output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens))
        if version == 'multiple_choice_question':
            inp_generator = get_inplens_router_bench_MCQ
            outlen_generator = lambda model_name, inp_lens: np.asarray([1]*len(inp_lens))
    elif test_case == 'general':
        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
        outlen_generator = output_length_sampler.sample_out_len_for_given_model
    elif test_case == 'map-reduce':
        chunk_size = 512
        fixed_output_size = 50
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))
    elif test_case == 'chain-summary':
        chunk_size = 512
        fixed_output_size = 50
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))

    return inp_generator, inp_merger, outlen_generator



def _get_router_bench_data():
    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    # store the inp model of each inp seq for a model if it does not take all out seqs from each inp model
    inp_req_ids = dict()
    
    # store information if the output of a model need to be merged to generate new out reqs
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    # whether a base model's different input sources are independent or need to be merged, ...
    independent_srcs = dict()
    sampling_args_dict = dict()

    # inp/out len generator functions for the general setting
    model_paths = [
        'meta-llama/Llama-2-70b-chat-hf',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'WizardLMTeam/WizardLM-13B-V1.2',
        'meta-llama/CodeLlama-34b-Instruct-hf',
        'mistralai/Mistral-7B-Instruct-v0.2',     
    ]

    # 1. 
    in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}



    # 2. 
    import json
    prompt_dict = None
    # with open('/ssddata/jingzhi/vLLM/vllm/benchmarks/router_bench_not_multiple_choice_question_dataset.json', 'r') as f:
    dataset_type = 'multiple_choice_question'
    dataset_type = 'not_multiple_choice_question'
    with open(f'/ssddata/jingzhi/vLLM/vllm/benchmarks/router_bench_{dataset_type}_dataset.json', 'r') as f:
        prompt_dict = json.load(f)
    
    tot_inp_prompts = list()

    inp_seq_ids_dict = dict()
    req_num = 0
    for i, model_path in enumerate(model_paths):
        prompts = prompt_dict[model_path]
        inp_seq_ids_dict[i] = list(range(req_num, req_num + len(prompts)))
        inp_seq_ids_dict[-(i+1)] = list(range(req_num, req_num + len(prompts)))
        req_num+=len(prompts)
        tot_inp_prompts.extend(prompts)


    print(f"new inp_seq_ids_dict: ")
    for i, v in inp_seq_ids_dict.items():
        print(f"model {i}, ratio: {len(v)/req_num}")


    # 3. 
    # inp_generator = get_inplens_router_bench
    # inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
    # # we control the max output length here
    # outlen_generator = lambda model_name, inp_lens: np.minimum(8192, output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens))
    
    inp_generator, inp_merger, outlen_generator = _get_req_len_funcs('router', dataset_type)
    
    node_dataset_chunk_mapping = {-(i+1): (None, 0, -1) \
                                    for i in range(len(model_paths))}


    # 4. we need to prepare the dummpy requests here
    _init_dummy_requests(None, tot_inp_prompts, model_path=model_paths[0])


    # 5. 
    independent_srcs = {i:False for i in range(len(model_paths))}
    max_tokens = 8192 # int(1e9)
    if dataset_type == 'multiple_choice_question':
        max_tokens = 1
    sampling_args2 = {                    
        "n":1,
        # <jingzhi> change to greedy sampling to check correctness.
        "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
        "top_p":1.0,
        "use_beam_search":False,
        "ignore_eos":False, # False, # True (original),
        "max_tokens":max_tokens, #int(1e9)
        }
    sampling_args_dict = {base_model_id:SamplingParams(**sampling_args2) for base_model_id in range(len(model_paths))}

    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
    print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")

    check_gap = 16
    sort_input = True

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping, \
        inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs, sampling_args_dict


def _get_schedule_setting_with_real_data(test_case: str, ratio_seed:int, ratio_set:int):
    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    # store the inp model of each inp seq for a model if it does not take all out seqs from each inp model
    inp_req_ids = dict()
    
    # store information if the output of a model need to be merged to generate new out reqs
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    # whether a base model's different input sources are independent or need to be merged, ...
    independent_srcs = dict()
    sampling_args_dict = dict()
    if test_case == 'router':
        return _get_router_bench_data()
    elif (test_case == 'general'):
        # inp/out len generator functions for the general setting
        model_paths = get_model_path_list()
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}
        req_num = 10000
        inp_seq_ids_dict = {i: list(range(req_num)) for i in range(len(model_paths))}
        if test_case == 'router':
            ratios = np.arange(1, len(model_paths)+1)
            if ratio_set == 2:
                ratios = np.asarray([2**i for i in range(len(model_paths)//2+1) for j in range(2)][:len(model_paths)])
            rng = np.random.default_rng(seed=ratio_seed)
            rng.shuffle(ratios)

            ratios = ratios/sum(ratios)
            cumnums = np.cumsum(np.concatenate(([0], (ratios*req_num).astype(int))))
            cumnums[-1] = req_num
            rand_seq_ids = np.arange(req_num)
            rng = np.random.default_rng(seed=0)
            rng.shuffle(rand_seq_ids)
            print(f"ratios: {ratios}, cumnums: {cumnums}, rand_seq_ids: {rand_seq_ids}")
            inp_seq_ids_dict = {i:sorted(rand_seq_ids[cumnums[i]:cumnums[i+1]]) for i in range(len(model_paths))}
            inp_seq_ids_dict.update({-(i+1):inp_seq_ids_dict[i] for i in range(len(model_paths))})
            print(f"new inp_seq_ids_dict: ")
            for i, v in inp_seq_ids_dict.items():
                print(f"model {i}, ratio: {ratios[i]} : {v}")

        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
        outlen_generator = output_length_sampler.sample_out_len_for_given_model
        node_dataset_chunk_mapping = {-(i+1): (None, 0, -1) \
                                      for i in range(len(model_paths))}



        # simply use the tokenizer of llama2 7b to check the lengths of the prompts
        # TODO: we simply use the llama 7b tokenizer here --> may change this
        args = InferenceArgs(model='NousResearch/Llama-2-7b-hf', num_prompts=req_num)
        from transformers import AutoTokenizer
        # Sample the requests.
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        requests = benchmark_throughput.sample_requests(
            "ShareGPT_V3_unfiltered_cleaned_split.json", args.num_prompts, tokenizer, args.output_len, 
            random_seed=args.seed)
        inp_prompts = [req[0] for req in requests]
        # # we need to prepare the dummpy requests here
        _init_dummy_requests(None, inp_prompts, model_path=model_paths[0])



        independent_srcs = {i:False for i in range(len(model_paths))}

        sampling_args2 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, # False, # True (original),
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args2) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
        print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")

    elif test_case == 'map-reduce':
        # NOTE: we have changed the computation graph to directly horizontally fuse all ``map`` models together
        req_num = 10
        chunk_size = 512
        fixed_output_size = 50
        model_paths = ['NousResearch/Llama-2-13b-hf'] * 2
        # out_edge_dict = {i:[len(model_paths)-1] for i in range(len(model_paths)-1)}
        in_edge_dict_with_dummy_inp_nodes = {0: [-1], 1:[0]}
        # 
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))
        node_dataset_chunk_mapping = {-1: (None, 0, chunk_size)}


        # sample the real data from dataset
        dataset_path = 'train-00000-of-00001-b334c773bce22cb2.parquet'
        sampled_inps: List[List[int]] = _get_document_prompts(dataset_path=dataset_path, model_path=model_paths[0], num_requests=req_num)
        inp_lens = np.asarray([len(prompt_token_ids) for prompt_token_ids in sampled_inps])
        req_num = min(req_num, len(inp_lens))

        # leave it later: for the case where we horizontally fuse all ``map`` models
        out_req_id_mapping = {0: dict()}
        tot_req_num = 0
        inp_seq_ids_dict = {1:[]}
        sampled_inp_chunks = list()
        for i, inp_len in enumerate(inp_lens):
            chunk_num = (inp_len+chunk_size-1)//chunk_size
            out_req_id_mapping[0].update({chunk_i+tot_req_num:(i, chunk_i) for chunk_i in range(chunk_num) })
            tot_req_num += chunk_num
            inp_seq_ids_dict[1].append(tot_req_num-1)
            sampled_inp_chunks.extend([sampled_inps[i][chunk_i*chunk_size:(chunk_i+1)*chunk_size] for chunk_i in range(chunk_num)])

        inp_seq_ids_dict.update({0:list(out_req_id_mapping[0].keys())})
        # inp_seq_ids_dict.update({-(i+1):inp_seq_ids_dict[i] for i in [0]})


        new_out_req_part_num = { 0: { i:(inp_len+chunk_size-1)//chunk_size for i, inp_len in enumerate(inp_lens)} }
        independent_srcs = {i:False for i in range(len(model_paths))}

        # we need to prepare the dummpy requests here
        _init_dummy_requests([chunk_size]*tot_req_num, sampled_inp_chunks, model_path=model_paths[0])

        req_num = tot_req_num

        sampling_args1 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, # False, # True (original),
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, # False, # True (original),
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")



    elif test_case == 'chain-summary':
        # # chain summary
        req_num = 1000
        chunk_size = 2048 # 512
        fixed_output_size = 900 # 50

        # sample the real data from dataset
        dataset_path = 'train-00000-of-00001-b334c773bce22cb2.parquet'
        model_path = 'NousResearch/Llama-2-13b-hf'
        # model_path = 'NousResearch/Llama-2-7b-hf'
        sampled_inps: List[List[int]] = _get_document_prompts(dataset_path=dataset_path, model_path=model_path, num_requests=req_num)
        # sort the sampled inps
        sampled_inps = sorted(sampled_inps, key=lambda i: len(i), reverse=True)
        inp_lens = np.asarray([len(prompt_token_ids) for prompt_token_ids in sampled_inps])
        req_num = min(req_num, len(inp_lens))

        print(f"inp_lens: {inp_lens}")

        max_length = max(inp_lens)

        print(f"max chunk num: {(max_length + chunk_size - 1) // chunk_size}")

        # model_paths = ['NousResearch/Llama-2-13b-hf'] * ((max_length + chunk_size - 1) // chunk_size)
        model_paths = [model_path] * ((max_length + chunk_size - 1) // chunk_size)
        print(f"model_paths: {model_paths}")
        # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
        # out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
        # in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] + list(range(i)) for i in range(len(model_paths))}
        in_edge_dict_with_dummy_inp_nodes = {0: [-1]}
        in_edge_dict_with_dummy_inp_nodes.update({i:[-(i+1)] + [i-1] for i in range(1, len(model_paths))})

        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))
        # here ``None`` means we use our own dummpy request dataset
        node_dataset_chunk_mapping = {-(i+1): (None, i, chunk_size)\
                                      for i in range(len(model_paths))}
        
        inp_seq_ids_dict = defaultdict(list)
        # inp_lens = np.asarray(inp_generator(req_num))
        # inp_lens = np.asarray([chunk_size]*int(0.2*req_num)+[2*chunk_size]*int(0.2*req_num)\
        #                     +[3*chunk_size]*int(0.2*req_num)+[4*chunk_size]*int(0.2*req_num)\
        #                         +[5*chunk_size]*int(0.2*req_num))
        # inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))
        inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*i)))) for i in range(len(model_paths))})
        print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")
        # inp_seq_ids_dict.update({-(i+1):inp_seq_ids_dict[i] for i in range(len(model_paths))})

        # add another model after the chain summary
        # model_paths.append('NousResearch/Llama-2-7b-hf')
        model_paths.append('NousResearch/Llama-2-70b-hf')
        in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = list(range(len(model_paths)-1)) # [len(model_paths)-2, len(model_paths)-3]
        # in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = [19, 49]
        # out_edge_dict[3].append(5)
        # out_edge_dict[4] = [5]
        # inp_seq_ids_dict[len(model_paths)-1] = sorted(set(inp_seq_ids_dict[len(model_paths)-2] + inp_seq_ids_dict[len(model_paths)-3]))
        inp_seq_ids_dict[len(model_paths)-1] = inp_seq_ids_dict[0]

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")

        
        # TODO: leave it later: for the case where we horizontally fuse all ``map`` models
        # inp_req_ids = dict()
        # independent_srcs = {i:False for i in range(len(model_paths))}
        inp_req_ids = {len(model_paths)-1: {i:sorted(set(inp_seq_ids_dict[i])-set(inp_seq_ids_dict[i+1])) for i in range(len(model_paths)-2)}}
        inp_req_ids[len(model_paths)-1][len(model_paths)-2] = inp_seq_ids_dict[len(model_paths)-2]
        independent_srcs[len(model_paths)-1] = True

        print(f"\nreal inp_req_ids: {inp_req_ids}\n")
        print(f"\nreal independent_srcs: {independent_srcs}\n")

        # we need to prepare the dummpy requests here
        # _init_dummy_requests(inp_lens)
        _init_dummy_requests(inp_lens, sampled_inps, model_path=model_paths[0])
        sampling_args1 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, # False, # True (original),
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, # False, # True (original),
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths)-1)}
        sampling_args_dict.update({len(model_paths)-1:SamplingParams(**sampling_args1)})




    # # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # # search_method_baseline = 'ours' # 'naive'  'ours'
    # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # search_method_baseline = 'ours' # 'naive'  'ours'
    check_gap = 16
    sort_input = True

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping, \
        inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs, sampling_args_dict






def get_schedule_setting(test_case:str, use_real_dataset:bool, ratio_seed:int, ratio_set:int):

    if use_real_dataset:
        return _get_schedule_setting_with_real_data(test_case=test_case, ratio_seed=ratio_seed, ratio_set=ratio_set)
    


    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    # store the inp model of each inp seq for a model if it does not take all out seqs from each inp model
    inp_req_ids = dict()
    
    # store information if the output of a model need to be merged to generate new out reqs
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    # whether a base model's different input sources are independent or need to be merged, ...
    independent_srcs = dict()
    sampling_args_dict = dict()
    if test_case == 'general':
        # inp/out len generator functions for the general setting
        model_paths = get_model_path_list()
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}
        req_num = 1000
        inp_seq_ids_dict = {i: list(range(req_num)) for i in range(len(model_paths))}
        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
        outlen_generator = output_length_sampler.sample_out_len_for_given_model
        node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", 0, -1) \
                                      for i in range(len(model_paths))}


        # inp/out len generator functions for the general setting
        # 假设是下面的这种拓扑结构：model 1 -> model 2
        # in_edge_dict_with_dummy_inp_nodes = {0:[-1], 1:[0]}
        # req_num = 20
        # inp_generator = get_inplens
        # inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] # concat all inputs from input models together
        # outlen_generator = output_length_sampler.sample_out_len_for_given_model
        # node_dataset_chunk_mapping = {-1: ("ShareGPT_V3_unfiltered_cleaned_split.json", 0, -1)}

        independent_srcs = {i:False for i in range(len(model_paths))}


        # # we need to prepare the dummpy requests here
        # sampled_prompts = 
        # _init_dummy_requests([], )

        # req_num = tot_req_num

        # sampling_args1 = {                    
        #     "n":1,
        #     # <jingzhi> change to greedy sampling to check correctness.
        #     "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
        #     "top_p":1.0,
        #     "use_beam_search":False,
        #     "ignore_eos":True, # False, # True (original),
        #     "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, # False, # True (original),
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args2) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
        print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")

    elif test_case == 'map-reduce':
        # inp/out len generator functions for the map-reduce or chain summary scenario
        # map-reduce
        # 如果input sequence的长度差别很大的话，可能会导致chunk的数量差别很大，可能会有很多个LLM instance，但是每个instance的inp workload数量不一样，这样会有影响吗？
        # 试试再说吧
        # TODO: 还有一个问题，我们现在判断redundancy不会把相同的model但是不同instance看成是相同的，这样可能会导致大量redundancy。
        # req_num = 10
        # chunk_size = 512
        # model_paths = ['NousResearch/Llama-2-13b-hf'] * (20000 // chunk_size)
        # model_paths = model_paths + ['NousResearch/Llama-2-13b-hf']
        # # out_edge_dict = {i:[len(model_paths)-1] for i in range(len(model_paths)-1)}
        # in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths)-1)}
        # in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = list(range(len(model_paths)-1))
        # # 
        # inp_generator = lambda req_num: [512]*req_num
        # inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
        # outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        # node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", i, chunk_size) \
        #                               for i in range(len(model_paths)-1)}
        


        # NOTE: we have changed the computation graph to directly horizontally fuse all ``map`` models together
        req_num = 10
        chunk_size = 512
        model_paths = ['NousResearch/Llama-2-13b-hf'] * 2
        # out_edge_dict = {i:[len(model_paths)-1] for i in range(len(model_paths)-1)}
        in_edge_dict_with_dummy_inp_nodes = {0: [-1], 1:[0]}
        # 
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [512]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] # not consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        node_dataset_chunk_mapping = {-1: (None, 0, chunk_size)}

        inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))

        # leave it later: for the case where we horizontally fuse all ``map`` models
        out_req_id_mapping = {0: dict()}
        tot_req_num = 0
        inp_seq_ids_dict = {1:[]}
        for i, inp_len in enumerate(inp_lens):
            chunk_num = (inp_len+chunk_size-1)//chunk_size
            out_req_id_mapping[0].update({chunk_i+tot_req_num:(i, chunk_i) for chunk_i in range(chunk_num) })
            tot_req_num += chunk_num
            inp_seq_ids_dict[1].append(tot_req_num-1)

        inp_seq_ids_dict.update({0:list(out_req_id_mapping[0].keys())})
        # inp_seq_ids_dict.update({-(i+1):inp_seq_ids_dict[i] for i in [0]})


        new_out_req_part_num = { 0: { i:(inp_len+chunk_size-1)//chunk_size for i, inp_len in enumerate(inp_lens)} }
        independent_srcs = {i:False for i in range(len(model_paths))}

        # we need to prepare the dummpy requests here
        _init_dummy_requests([chunk_size]*tot_req_num)

        req_num = tot_req_num

        sampling_args1 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, # False, # True (original),
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, # False, # True (original),
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")


    elif test_case == 'chain-summary':
        # # chain summary
        # req_num = 1000
        # chunk_size = 512
        # model_paths = ['NousResearch/Llama-2-13b-hf'] * (20000 // chunk_size)
        # # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
        # # # out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
        # in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] + list(range(i)) for i in range(len(model_paths))}
        
        # inp_generator = lambda req_num: [512]*req_num
        # inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
        # outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        # node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", i, chunk_size)\
        #                               for i in range(len(model_paths))}


        req_num = 10
        chunk_size = 512
        max_length = chunk_size*50 # 20000
        model_paths = ['NousResearch/Llama-2-13b-hf'] * (max_length // chunk_size)
        print(f"model_paths: {model_paths}")
        # out_edge_dict = {i:list(range(i+1, len(model_paths))) for i in range(len(model_paths)-1)}
        # out_edge_dict = {i:[i+1] for i in range(len(model_paths)-1)}
        # in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] + list(range(i)) for i in range(len(model_paths))}
        in_edge_dict_with_dummy_inp_nodes = {0: [-1]}
        in_edge_dict_with_dummy_inp_nodes.update({i:[-(i+1)] + [i-1] for i in range(1, len(model_paths))})

        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] # consider model original inplens
        outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        # here ``None`` means we use our own dummpy request dataset
        node_dataset_chunk_mapping = {-(i+1): (None, i, chunk_size)\
                                      for i in range(len(model_paths))}
        
        inp_seq_ids_dict = defaultdict(list)
        # inp_lens = np.asarray(inp_generator(req_num))
        # inp_lens = np.asarray([chunk_size]*int(0.2*req_num)+[2*chunk_size]*int(0.2*req_num)\
        #                     +[3*chunk_size]*int(0.2*req_num)+[4*chunk_size]*int(0.2*req_num)\
        #                         +[5*chunk_size]*int(0.2*req_num))
        inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))
        inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*i)))) for i in range(len(model_paths))})
        print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")
        # inp_seq_ids_dict.update({-(i+1):inp_seq_ids_dict[i] for i in range(len(model_paths))})

        # add another model after the chain summary
        model_paths.append('NousResearch/Llama-2-7b-hf')
        # in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = [len(model_paths)-2, len(model_paths)-3]
        in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = [19, 49]
        # out_edge_dict[3].append(5)
        # out_edge_dict[4] = [5]
        inp_seq_ids_dict[len(model_paths)-1] = sorted(set(inp_seq_ids_dict[len(model_paths)-2] + inp_seq_ids_dict[len(model_paths)-3]))

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")

        
        # TODO: leave it later: for the case where we horizontally fuse all ``map`` models
        inp_req_ids = dict()
        independent_srcs = {i:False for i in range(len(model_paths))}
        independent_srcs[len(model_paths)-1] = True

        # we need to prepare the dummpy requests here
        _init_dummy_requests(inp_lens)
        sampling_args1 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, # False, # True (original),
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            # <jingzhi> change to greedy sampling to check correctness.
            "temperature":1.0, # 0 or 1e-6 (greedy), #1.0
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, # False, # True (original),
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths)-1)}
        sampling_args_dict.update({len(model_paths)-1:SamplingParams(**sampling_args1)})



    # # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # # search_method_baseline = 'ours' # 'naive'  'ours'
    # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # search_method_baseline = 'ours' # 'naive'  'ours'
    check_gap = 16
    sort_input = True

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping, \
        inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs, sampling_args_dict




if __name__ == "__main__":
    print("start")
    # --------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="args of end 2 end test")
    parser.add_argument("--gen-execplans-baseline",
                        type=str,
                        choices=["ours", "naive"],
                        default="ours")

    parser.add_argument("--test-case",
                        type=str,
                        choices=["general", "map-reduce", "chain-summary", "router"],
                        default="router")
    
    parser.add_argument("--ratio-seed",
                        type=int)    
    
    parser.add_argument("--ratio-set",
                        type=int)    
    
    args = parser.parse_args()

    # # gen_execplans_baseline = 'ours' # 'naive'  'ours'
    # # search_method_baseline = 'ours' # 'naive'  'ours'
    gen_execplans_baseline = 'ours' # 'naive'  'ours'
    search_method_baseline = 'ours' # 'naive'  'ours'
    test_case = 'router' # 'general' 'map-reduce' 'chain-summary', 'router'

    gen_execplans_baseline = args.gen_execplans_baseline
    test_case = args.test_case
    ratio_seed = args.ratio_seed
    ratio_set = args.ratio_set


    model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        num_prompts, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping, \
             inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs, sampling_args_dict = \
        get_schedule_setting(test_case=test_case, use_real_dataset=True, ratio_seed=ratio_seed, ratio_set=ratio_set)
    
    asyncio.run(main_with_preemption(
        test_case=test_case,
        model_paths=model_paths,
        gen_execplans_baseline=gen_execplans_baseline,
        search_method_baseline=search_method_baseline,
        # 
        # support model-level pipeline parallel
        in_edge_dict_with_dummy_inp_nodes=in_edge_dict_with_dummy_inp_nodes,
        node_dataset_chunk_mapping=node_dataset_chunk_mapping,
        check_gap=check_gap, sort_input=sort_input,
        num_prompts=num_prompts, 
        # 
        sampling_args_dict=sampling_args_dict,
        # 
        inp_seq_ids_dict=inp_seq_ids_dict, 
        inp_req_ids=inp_req_ids, out_req_id_mapping=out_req_id_mapping, 
        new_out_req_part_num=new_out_req_part_num, independent_srcs=independent_srcs,
        inp_generator=inp_generator, inp_merger=inp_merger, outlen_generator=outlen_generator,
        # 
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num=4,
        max_group_seq_num=20,
        top_k=20,
        similar_threshold=0.2,
        # NOTE: 1. for DSF servers: fully_connected_gpu_unit=2, for lccpus, fully_connected_gpu_unit=4.
        fully_connected_gpu_unit=2,
        machine_name='zxcpu'))
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