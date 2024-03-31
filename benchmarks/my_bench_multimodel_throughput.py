""" Benchmark offline multi-model inference throughput. """

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

from vllm.core.multimodel_scheduler import SHARED_CONTECT
import benchmark_throughput

import time
import numpy as np
from typing import List, Optional, Tuple
import itertools

# shared_counter: Array # = Array('d', [-1, -1])





# define the args we need
class InferenceArgs:
    """Arguments for vLLM single model inference."""
    def __init__(self, 
        model:str="huggyllama/llama-7b", 
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
        self.num_prompts: int = 2000
        self.seed: int = 0
        self.hf_max_batch_size: int = None
        self.trust_remote_code: bool = False
        self.max_model_len: int = None
        self.dtype: str = 'auto'
        self.enforce_eager: bool = True
        self.kv_cache_dtype: str = "auto"
        self.device: str = "cuda"

        # added parameters
        self.weight_load_degree: str = '16'
        self.gpu_use_ratio: float = 0.9

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





# init shared setting
# class SHARED_CONTECT():
#     shared_setting: Array = None

# def init_shared_vars(model_setting: Array):
#     SHARED_CONTECT.shared_setting = model_setting
#     # benchmark_throughput.shared_setting = model_setting



# def increment():
#     with SHARED_CONTECT.shared_setting.get_lock():
#         SHARED_CONTECT.shared_setting[0] += 1


# start a model for inference
def start_a_model_inference_child_process(gpus: str, model_id: int, model: str = "huggyllama/llama-7b"):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    os.environ['USE_VLLM']='False'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
    # os.environ['RUN_MULTI_MODEL'] = 'True'
    args = InferenceArgs(model)
    # set os.environ['CUDA_VISIBLE_DEVICES'] before importing benchmark_throughput
    # benchmark_throughput.SHARED_CONTECT.shared_setting = SHARED_CONTECT.shared_setting
    # set shared id for each model
    SHARED_CONTECT.shared_id = model_id
    try:
        benchmark_throughput.main(args)
    except Exception as e:
        print(e)





# start a model for inference
def start_a_model_inference(gpus: str, model_id: int, model: str = "huggyllama/llama-7b"):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(start_a_model_inference_child_process, gpus, model_id, model)


 
# def init(counter: Array):
#     global shared_counter
#     shared_counter = counter
 
 
# def increment():
#     with shared_counter.get_lock():
#         shared_counter[0] += 1
 
 
# async def main():
#     counter = Array('d', [0, 0]) # 'd' is for double
#     with ProcessPoolExecutor(initializer=benchmark_throughput.init,
#                              initargs=(counter,)) as pool:
#         await asyncio.get_event_loop().run_in_executor(pool, benchmark_throughput.increment)
#         print(counter[0])
#         for v in counter:
#             print(v)






# async def main_test():
#     loop = asyncio.get_running_loop()
#     tasks = []

#     counter = Array('d', [0, 0]) # 'd' is for double

#     with ProcessPoolExecutor(initializer=benchmark_throughput.init,
#                              initargs=(counter,)) as executor:
#         for gpus in ['2,1,3,0', '3,0,2,1']:
#             tasks.append(loop.run_in_executor(executor, benchmark_throughput.increment))
        
#         # # Or we can just use the method asyncio.gather(*tasks)
#         # for done in asyncio.as_completed(tasks):
            
#         #     result = await done
#         #     print(f"sum_to_num got a result which is {result}")

#         done_list, pending_list = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
#         for task in done_list:
#             await task
#             print(f">got :")
#             for v in counter:
#                 print(v)
#         print(f"the length of pending is {len(pending_list)}")

#         # deal with the pending tasks
#         asyncio.gather(*pending_list)
#         print(f"final >got :")
#         for v in counter:
#             print(v)        








# async def main():
#     import os
#     os.environ['RUN_MULTI_MODEL'] = 'True'

#     loop = asyncio.get_running_loop()
#     tasks = []

#     counter = Array('d', [0, 0]) # 'd' is for double
#     # all child processors will inherit this event
#     SHARED_CONTECT.events = [Event() for _ in range(2+2)]
#     # set the event to allow models to run
#     SHARED_CONTECT.events[1].set()
#     SHARED_CONTECT.shared_setting = counter

#     # with ProcessPoolExecutor(initializer=SHARED_CONTECT.init_shared_vars,
#     #                          initargs=(counter,)) as executor:
#     with ProcessPoolExecutor() as executor:
#         for model_id, gpus in enumerate(['2,1,3,0', '3,0,2,1']):
#             tasks.append(loop.run_in_executor(executor, start_a_model_inference, gpus, model_id))
        
#         # # Or we can just use the method asyncio.gather(*tasks)
#         # for done in asyncio.as_completed(tasks):
            
#         #     result = await done
#         #     print(f"sum_to_num got a result which is {result}")


#         # wait a moment
#         sleep(30)
#         # start all issued tasks
#         print('Setting event.', flush=True)
#         SHARED_CONTECT.events[0].set()
#         SHARED_CONTECT.events[1].set()


#         done_list, pending_list = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
#         for task in done_list:
#             await task
#             print(f">got :")
#             for v in counter:
#                 print(v)
#         print(f"the length of pending is {len(pending_list)}")

#         # deal with the pending tasks
#         asyncio.gather(*pending_list)
#         print(f"final >got :")
#         for v in counter:
#             print(v)
















async def main_with_preemption():
    import os
    import ctypes
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'True'
    os.environ['NO_PREEMPT'] = 'False'

    loop = asyncio.get_running_loop()
    tasks = []

    
    model_list = ['NousResearch/Llama-2-13b-hf'] + ['NousResearch/Llama-2-70b-hf']
    # model_list = ['huggyllama/llama-7b']*5 + ['NousResearch/Llama-2-70b-hf']


    counter = Array('i', [0 for i in range(len(model_list)*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+len(model_list))]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(len(model_list))])




    # set the initial execution plan
    # TODO (jingzhi) add a function to compute the current best execution plan for all models
    # new_setting = [2, 7, 2, 0,1,2,3] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    # new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    new_setting = [2, 8, 2, 1,0,2,3] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    SHARED_CONTECT.set_execution_plan(new_setting)

    # test more iterations of multi-model inference
    # first two 7b model run on 1 gpu respectively, then next 2 7b, then 1 7b taking 2 gpus; while 1 70b taking 2 gpus all the time.
    # new_setting = [1, 5, 2, 0,1,2,3] + [1, 5, 2, 1,0,2,3] + [1, 5, 2, 0,1,2,3] + [1, 5, 2, 1,0,2,3] + [2, 5, 2, 0,1,2,3] + [2, 9, 16, 2,3,0,1]
    # SHARED_CONTECT.set_execution_plan(new_setting)


    with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
        # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
        for model_id, model in enumerate(model_list):
            tasks.append(loop.run_in_executor(executor, start_a_model_inference, '2,1,3,0', model_id, model))


        start = time.perf_counter()
        print(f"Outer iter start time ---abs: {start}")

        pending_list = tasks
        model_schedule_iter = 0
        while len(pending_list) > 0:
            # repeat this loop until all models are finished
            print(f"a new iteration==================", flush=True)

            done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)

            # <jingzhi> For Profiling
            start_waiting = time.perf_counter()
            print(f"total time to launch processes (just the value of iter 0 is useful) {model_schedule_iter}: {start_waiting-start}s ---abs: {start_waiting}")

            # now at least one model is finished, make all models stop
            SHARED_CONTECT.events[0].set()
            # TODO (jingzhi) try to make the finished processes release their resources
            for task in done_list:
                await task
            SHARED_CONTECT.wait_all_models_to_finish()
            # determine the new execution plan     
            # TODO (jingzhi) support automatic rescheduling
            # new_setting = [4, 9, 2, 2,3,0,1]
            new_setting = [4, 9, 2, 2,3,0,1]
            SHARED_CONTECT.set_execution_plan(new_setting, [1])
            # reset event statuses
            SHARED_CONTECT.restart_all_models()
            print(f"len(pending_list): {len(pending_list)}", flush=True)

            # <jingzhi> For Profiling
            end_waiting = time.perf_counter()
            print(f"total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
            model_schedule_iter += 1

        end = time.perf_counter()
        print(f"total running time: {end-start}s ---abs: {end}")





# <jingzhi> this function will triger models to start when conditions meet; models will run with no preemption
async def main_No_preemption():
    import os
    import ctypes
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'True'
    os.environ['NO_PREEMPT'] = 'True'

    loop = asyncio.get_running_loop()
    tasks = []

    
    # model_list = ['NousResearch/Llama-2-13b-hf'] + ['NousResearch/Llama-2-70b-hf']
    model_list = ['huggyllama/llama-7b']*5 + ['NousResearch/Llama-2-70b-hf']


    counter = Array('i', [0 for i in range(len(model_list)*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+len(model_list))]
    SHARED_CONTECT.started_status = [Event() for _ in range(len(model_list))]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(len(model_list))])




    # set the initial execution plan
    # TODO (jingzhi) add a function to compute the current best execution plan for all models
    # new_setting = [2, 8, 2, 1,0,2,3] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    # new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    # SHARED_CONTECT.set_execution_plan(new_setting)

    # test more iterations of multi-model inference
    # first two 7b model run on 1 gpu respectively, then next 2 7b, then 1 7b taking 2 gpus; while 1 70b taking 2 gpus all the time.
    new_setting = [1, 8, 2, 0,1,2,3] + [1, 8, 2, 1,0,2,3] + [1, 8, 2, 0,1,2,3] + [1, 8, 2, 1,0,2,3] + [2, 8, 2, 0,1,2,3] + [2, 9, 16, 2,3,0,1]
    SHARED_CONTECT.set_execution_plan(new_setting)

    # block 3 7b models at the beginning
    SHARED_CONTECT.start_specific_models([0, 1, 5])


    with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
        # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
        for model_id, model in enumerate(model_list):
            tasks.append(loop.run_in_executor(executor, start_a_model_inference, '2,1,3,0', model_id, model))


        start = time.perf_counter()
        print(f"Outer iter start time ---abs: {start}")

        pending_list = tasks
        model_schedule_iter = 0
        while len(pending_list) > 0:
            # repeat this loop until all models are finished
            print(f"a new iteration==================", flush=True)

            done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)

            # <jingzhi> For Profiling
            start_waiting = time.perf_counter()
            print(f"total time to launch processes (just the value of iter 0 is useful) {model_schedule_iter}: {start_waiting-start}s ---abs: {start_waiting}")

            # # now at least one model is finished, make all models stop
            # SHARED_CONTECT.events[0].set()
            # # TODO (jingzhi) try to make the finished processes release their resources
            # for task in done_list:
            #     await task
            # SHARED_CONTECT.wait_all_models_to_finish()
            # # determine the new execution plan     
            # # TODO (jingzhi) support automatic rescheduling
            # # new_setting = [4, 9, 2, 2,3,0,1]
            # new_setting = [4, 9, 2, 2,3,0,1]
            # SHARED_CONTECT.set_execution_plan(new_setting, [1])
            # # reset event statuses
            # SHARED_CONTECT.restart_all_models()
            print(f"len(pending_list): {len(pending_list)}", flush=True)

            # <jingzhi> For Profiling
            end_waiting = time.perf_counter()
            print(f"total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
            model_schedule_iter += 1


            # start remaining models
            if SHARED_CONTECT.query_finish_status(0) and (not SHARED_CONTECT.query_finish_status(2)):
                SHARED_CONTECT.start_specific_models([2])
            if SHARED_CONTECT.query_finish_status(1) and (not SHARED_CONTECT.query_finish_status(3)):
                SHARED_CONTECT.start_specific_models([3])
            if SHARED_CONTECT.query_finish_status(2) and SHARED_CONTECT.query_finish_status(3):
                SHARED_CONTECT.start_specific_models([4])


        end = time.perf_counter()
        print(f"total running time: {end-start}s ---abs: {end}")











async def main_with_preemption_debug():
    import os
    import ctypes
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'True'

    loop = asyncio.get_running_loop()
    tasks = []

    model_list = ['NousResearch/Llama-2-70b-hf']
    num_models = len(model_list)

    counter = Array('i', [0 for i in range(num_models*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+num_models)]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(num_models)])



    # set the initial execution plan
    # TODO (jingzhi) add a function to compute the current best execution plan for all models
    # new_setting = [2, 7, 2, 0,1,2,3] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    # new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    new_setting = [] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    SHARED_CONTECT.set_execution_plan(new_setting)

    # test more iterations of multi-model inference
    # new_setting = [1, 5, 2, 0,1,2,3] + [1, 5, 2, 1,0,2,3] + [1, 5, 2, 0,1,2,3] + [1, 5, 2, 1,0,2,3] + [2, 9, 16, 2,3,0,1]


    with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
        for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
            tasks.append(loop.run_in_executor(executor, start_a_model_inference, gpus, model_id, model))


        start = time.perf_counter()
        print(f"Outer iter start time ---abs: {start}")

        pending_list = tasks
        model_schedule_iter = 0
        done_list = list()
        while len(pending_list) > 0:
            # repeat this loop until all models are finished
            print(f"a new iteration==================", flush=True)

            if model_schedule_iter>0:
                done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)
            else:
                time.sleep(65)


            # <jingzhi> For Profiling
            start_waiting = time.perf_counter()
            print(f"total time to launch processes (just the value of iter 0 is useful) {model_schedule_iter}: {start_waiting-start}s ---abs: {start_waiting}")

            # now at least one model is finished, make all models stop
            SHARED_CONTECT.events[0].set()
            # TODO (jingzhi) try to make the finished processes release their resources
            for task in done_list:
                await task
            SHARED_CONTECT.wait_all_models_to_finish()
            # determine the new execution plan     
            # TODO (jingzhi) support automatic rescheduling
            # new_setting = [4, 9, 2, 2,3,0,1]
            new_setting = [4, 9, 2, 2,3,0,1]
            SHARED_CONTECT.set_execution_plan(new_setting, [0])
            # reset event statuses
            SHARED_CONTECT.restart_all_models()
            print(f"len(pending_list): {len(pending_list)}", flush=True)

            # <jingzhi> For Profiling
            end_waiting = time.perf_counter()
            print(f"total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
            model_schedule_iter += 1

        end = time.perf_counter()
        print(f"total running time: {end-start}s ---abs: {end}")




# def sleep_for_some_time(delay):
#     import torch
#     a = torch.Tensor(range(2)).to(0)
#     import time
#     time.sleep(delay)
#     # import sys
#     # import os
#     # os._exit(0)
#     # sys.exit(f"sleep for {delay}")


# def child_proc_execution(delay):
#     with ProcessPoolExecutor(max_workers=1) as executor:
#         executor.submit(sleep_for_some_time, delay)




# async def main_test():
#     loop = asyncio.get_running_loop()
#     tasks = []

#     with ProcessPoolExecutor() as executor:
#         for delay in [5,10]:
#             tasks.append(loop.run_in_executor(executor, child_proc_execution, delay))


#         pending_list = tasks
#         while len(pending_list) > 0:
#             # repeat this loop until all models are finished
#             print(f"a new iteration==================", flush=True)

#             done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)

#             print(f"len(pending_list): {len(pending_list)}", flush=True)















# # 测试一下不用manager，直接继承global 变量的event是否可行
    
# from random import random
# from time import sleep
# # from multiprocessing import set_start_method

# # from multiprocessing.pool import Pool
 
# # task executed in a worker process
# def task(identifier):
#     # wait for the event to be set
#     print(f'Task {identifier} waiting...', flush=True)
#     event.wait()
#     # generate a value
#     value = random()
#     # block for a moment
#     # sleep(value)
#     # report a message
#     print(f'Task {identifier} completed with {value}', flush=True)
 




# async def main():
#     loop = asyncio.get_running_loop()
#     tasks = []

#     counter = Array('d', [0, 0]) # 'd' is for double

#     # # create the shared event
#     # event = Event()



#     with ProcessPoolExecutor() as executor:
#     # with ProcessPoolExecutor() as executor:
#         for model_id, gpus in enumerate(['1,2,3,0', '3,0,2,1']):
#             tasks.append(loop.run_in_executor(executor, SHARED_CONTECT.test_task, model_id))
        
#         # # Or we can just use the method asyncio.gather(*tasks)
#         # for done in asyncio.as_completed(tasks):
            
#         #     result = await done
#         #     print(f"sum_to_num got a result which is {result}")

#         # wait a moment
#         sleep(2)
#         # start all issued tasks
#         print('Setting event.', flush=True)
#         SHARED_CONTECT.event.set()
#         # wait for all tasks to finish
#         asyncio.gather(*tasks)
#         print('All done.', flush=True)




# import os
# os.environ['CUDA_VISIBLE_DEVICES']='2,1,3,0'
# os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False'
# os.environ['USE_VLLM']='True'
# os.environ['RUN_MULTI_MODEL'] = 'False'

# import time

# from vllm import LLM
# from ray.util import remove_placement_group

# os.environ['WEIGHT_LOAD_DEGREE'] = '16'

# start_time = time.perf_counter()
# llm = LLM(model='NousResearch/Llama-2-70b-hf', enforce_eager=True, tensor_parallel_size=4)
# end_time = time.perf_counter()
# print(f"total initialization time: {end_time - start_time}")

# llm.llm_engine.delete_workers()
# # Import placement group APIs.
# from ray.util.placement_group import (
#     placement_group,
#     placement_group_table,
#     remove_placement_group,
#     get_placement_group,
# )

# pg_info = list(placement_group_table().values())[0]
# pg = get_placement_group(pg_info['name'])
# remove_placement_group(pg)

# from vllm.core.block_manager import KVBlkPerLayerWeight
# KVBlkPerLayerWeight.reset()

# os.environ['WEIGHT_LOAD_DEGREE'] = '2'
# llm = LLM(model="huggyllama/llama-7b", enforce_eager=True, tensor_parallel_size=4)






async def main_test_load_model():
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,0'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
    os.environ['USE_VLLM']='False'
    os.environ['RUN_MULTI_MODEL'] = 'False'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'True'

    import time

    from vllm import LLM
    from ray.util import remove_placement_group

    os.environ['WEIGHT_LOAD_DEGREE'] = '2'

    start_time = time.perf_counter()
    # huggyllama/llama-7b   NousResearch/Llama-2-70b-hf   NousResearch/Llama-2-13b-hf
    llm = LLM(model='NousResearch/Llama-2-13b-hf', enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.9,)
    end_time = time.perf_counter()
    print(f"total initialization time: {end_time - start_time}")




























































































































































































# <jingzhi> about the decoding phase
# TODO (jingzhi): NOTE the sum(contexts) here
def cal_flops(T,V,h,I,L,contexts):
    return 2*T*V*h+L*(4*T*h*h+2*sum(contexts)*h+3*I*h*T)




def cal_param_num(V,h,I,L):
    return 2*V*h+L*(4*h*h+3*I*h+2*h)+h



model_param_configs = {'llama_7b': {'V':32000,'h':4096,'I':11008, 'L':32},
                 'llama_13b':{'V':32000,'h':5120,'I':13824, 'L':40},
                 'llama_70b':{'V':32000,'h':8192,'I':28672, 'L':80}}


def cal_extra_param_bytes(V,h,v_byte):
    return (2*V*h+h) * v_byte

def cal_each_layer_param_bytes(V,h,I,L,v_byte):
    return 4*h*h+3*I*h+2*h




def get_sample_dataset():
    import os
    import random
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
    os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,0'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
    os.environ['USE_VLLM']='False'
    os.environ['RUN_MULTI_MODEL'] = 'False'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'True'
    os.environ['WEIGHT_LOAD_DEGREE'] = '2'
    
    args = InferenceArgs('huggyllama/llama-7b')

    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = benchmark_throughput.sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)





class MyModelInfor:
    """ My model information class. Contains basic model information. """
    def __init__(
        self,
        model_name, flops_per_token,
        layer_num, param_byte_per_layer, extra_param_byte # model_info
    ) -> None:
        self.model_name = model_name
        self.flops_per_token = flops_per_token
        self.layer_num = layer_num
        self.param_byte_per_layer = param_byte_per_layer
        self.extra_param_byte = extra_param_byte
        self.left_flops_per_token = flops_per_token
    
    def get_model_info(self):
        return self.layer_num, self.param_byte_per_layer, self.extra_param_byte
    def get_name(self):
        return self.model_name
    def get_flops_per_token(self):
        return self.flops_per_token
    def get_left_flops_per_token(self):
        return self.left_flops_per_token
    def is_finished(self):
        return self.left_flops_per_token == 0
    def update_left_flops_per_token(self, computed_ratio):
        # computed_ratio: time taken for computation / total time needed to finish computation
        self.left_flops_per_token = self.left_flops_per_token * (1-computed_ratio)
    def reset_left_flops_per_token(self, v):
        self.left_flops_per_token = v
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
        mem_per_comp_gpu: float, 
        param_byte_per_comp_gpu: int, 
        param_byte_per_cache_gpu: int
    ) -> None:
        self.model: MyModelInfor = model
        self.num_worker = num_worker
        self.wld_degree = wld_degree
        self.cache_gpu_num = cache_gpu_num
        self.mem_per_comp_gpu = mem_per_comp_gpu
        self.param_byte_per_comp_gpu = param_byte_per_comp_gpu
        self.param_byte_per_cache_gpu = param_byte_per_cache_gpu
    
    def estimate_exec_time(self, gpu_name: str):
        throughput = get_throughput(gpu_name, self)
        return self.model.get_left_flops_per_token()/throughput
    
    def __str__(self) -> str:
        return f"{str(self.model)}, "\
            f"tp:{self.num_worker}, wld:{self.wld_degree}, "\
            f"cache_gpu:{self.cache_gpu_num}, mem_r:{self.mem_per_comp_gpu}, "\
            # f"param_byte_per_comp_gpu:{self.param_byte_per_comp_gpu}, param_byte_per_cache_gpu:{self.param_byte_per_cache_gpu}"



class MyExecPlanGroup:
    """ My execution plan group definition. """
    def __init__(
        self,
        exec_plans: List[MyExecPlan], 
    ) -> None:
        self.exec_plans = exec_plans
        self.throughput = None
        self.tot_worker_num = None
    def comp_throughput(self, gpu_name: str):
        '''
        Get the total throughput of the given plan group.
        '''
        self.throughput = sum([get_throughput(gpu_name, exec_plan) for exec_plan in self.exec_plans])
    def get_tot_worker_num(self):
        self.tot_worker_num = sum([exec_plan.num_worker for exec_plan in self.exec_plans])
        return self.tot_worker_num
    def __str__(self):
        '''
        Get the string to represent a plan group
        '''
        return str(sorted([str(exec_plan) + str(exec_plan.model.get_left_flops_per_token()) for exec_plan in self.exec_plans]))
    def __len__(self):
        return len(self.exec_plans)




class MyExecPlanGroupSeq:
    """ My execution plan group sequence definition. """
    def __init__(
        self,
        plan_group_seq: List[MyExecPlanGroup], 
        time_seq: List[float]
    ) -> None:
        self.plan_group_seq = plan_group_seq
        self.time_seq = time_seq
    def get_tot_time(self):
        return sum(self.time_seq)
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
            return f"{[[str(exec_plan) for exec_plan in group.exec_plans] for group in self.plan_group_seq], self.time_seq}"



def get_factors(v):
    return [i for i in range(1, v+1) if v%i==0]



def is_valid_exec_plan(exec_plan: MyExecPlan, byte_per_gpu):
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
    # 
    # 0. is mem_per_comp_gpu and wld_degree consistent with each other
    if (wld_degree > 0) and (mem_per_comp_gpu < 0.9):
        return False
    # 1. whether mem is enough
    # check comp gpu mem
    if mem_per_comp_gpu * byte_per_gpu < param_byte_per_comp_gpu:
        return False
    # check cache gpu mem
    if byte_per_gpu < param_byte_per_cache_gpu:
        return False
    # 2. whether weight loading bandwidth is enough
    # compare the peak comp throughput with the weight loading speed, leave it to the cost model? I think it is ok.
    return True






def get_possible_exec_plans(model: MyModelInfor, tot_gpu_num, byte_per_gpu):
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
    if model.left_flops_per_token == 0:
        return exec_plans
    layer_num, param_byte_per_layer, extra_param_byte = model.get_model_info()
    tot_param_byte = layer_num * param_byte_per_layer + extra_param_byte
    for i in range(int(math.log(tot_gpu_num, 2)+1)):
        num_worker = 2**i
        for wld_degree in [2, 16]: # get_factors(layer_num): # TODO
            if wld_degree < 2:
                # if wld_degree <= 2, then we do not need to use cache gpu, so <2 will be the same as ==2.
                continue
            param_byte_per_comp_gpu = (tot_param_byte - (wld_degree - 2) * param_byte_per_layer) / num_worker
            for cache_gpu_num in range(tot_gpu_num-num_worker+1):
                if (wld_degree > 2) and (cache_gpu_num == 0):
                    # no cache gpu but have layers cached on other gpus, inconsistent
                    continue
                if (wld_degree == 2) and (cache_gpu_num > 0):
                    # no layer cached but has cache gpus
                    continue
                for mem_per_comp_gpu in [0.9]: # TODO [j/10 for j in range(1, 10)]:
                    # we can compute param_byte_per_cache_gpu
                    param_byte_per_cache_gpu = 0
                    if cache_gpu_num > 0:
                        param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / num_worker / cache_gpu_num
                    exec_plan = MyExecPlan(model,
                        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu)
                    # check whether exec_plan is valid
                    if is_valid_exec_plan(exec_plan, byte_per_gpu):
                        exec_plans.append(exec_plan)

    return exec_plans






def get_throughput(gpu_name:str, exec_plan:MyExecPlan):
    '''
    Get the peak decoding throughput for the given model and the given execution plan.
    Input:
        exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    num_worker = exec_plan.num_worker
    wld_degree = exec_plan.wld_degree
    cache_gpu_num = exec_plan.cache_gpu_num
    mem_per_comp_gpu = exec_plan.mem_per_comp_gpu
    # 
    cost_dict_key = (exec_plan.model.get_name(), gpu_name, num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu)
    cost_dict = {('llama_70b', 'A100-80G', 2, 16, 2, 0.9): 16.772150315187663, 
                 ('llama_7b', 'A100-80G', 1, 2, 0, 0.9): 5.021288393914332, 
                 ('llama_7b', 'A100-80G', 2, 2, 0, 0.9): 8.077960270014238, 
                 ('llama_7b', 'A100-80G', 4, 2, 0, 0.9): 13.637959291700502, 
                 ('llama_70b', 'A100-80G', 4, 2, 0, 0.9): 31.92388966289352, 
                 ('llama_70b', 'A100-80G', 2, 2, 0, 0.9): 14.395333220602716}
    if cost_dict_key not in cost_dict:
        # temporarily return -inf for this case, but we will measure every possible execution plan throughput all models
        return 1e-9
    else:
        return cost_dict[cost_dict_key]




# def get_one_stage_total_throughput(plan_group: List[MyExecPlan], gpu_name: str):
#     '''
#     Get the total throughput of the given plan group.
#     '''
#     return sum([get_throughput(gpu_name, exec_plan) for exec_plan in plan_group])


def comp_throughput_given_plan_groups(plan_groups: List[MyExecPlanGroup], gpu_name: str):
    for plan_group in plan_groups:
        plan_group.comp_throughput(gpu_name)





# def get_failed_gpu_request_pair_num(resources: np.ndarray, requests):
#     return sum([sum(resources < r_group[0]) * len(r_group) for r_group in requests])



# # we should consider the request combination 
# # as cache requests from 1 model should be assigned to different GPUs.
# def select_best_gpu_for_cache_request(r, resources: np.ndarray, requests):
#     best_i = None
#     best_fail_num = float('inf')
#     tmp_resources = resources.copy()
#     for i, cache in enumerate(resources):
#         if cache < r:
#             continue
#         tmp_resources[i] = tmp_resources[i] - r
#         fail_num = get_failed_gpu_request_pair_num(tmp_resources, requests)
#         if fail_num < best_fail_num:
#             best_fail_num = fail_num
#             best_i = i
#         tmp_resources[i] = tmp_resources[i] + r
#     return best_i, best_fail_num




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
        fail_ratio = sum([sum((tmp_resources < r[0])*tmp_resources) for r in requests]) / sum(tmp_resources)
        if fail_ratio < best_fail_ratio:
            best_fail_ratio = fail_ratio
            best_choice = choice_list
        tmp_resources[choice_list] = tmp_resources[choice_list] + request
    return best_choice, best_fail_ratio





def get_tot_worker_num(exec_plans: List[MyExecPlan]):
    return sum([exec_plan.num_worker for exec_plan in exec_plans])


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
    use_cache_gpu_plans = []
    without_cache_gpu_plans = []
    cache_gpu_remaining_bytes = []
    cache_gpu_required_bytes = []
    for exec_plan in exec_plans:
        num_worker = exec_plan.num_worker
        cache_gpu_num = exec_plan.cache_gpu_num
        mem_per_comp_gpu = exec_plan.mem_per_comp_gpu
        param_byte_per_cache_gpu = exec_plan.param_byte_per_cache_gpu
        if cache_gpu_num > 0:
            use_cache_gpu_plans.append(exec_plan)
            cache_gpu_required_bytes.append(np.asarray([param_byte_per_cache_gpu]*cache_gpu_num))
        else:
            without_cache_gpu_plans.append(exec_plan)
            cache_gpu_remaining_bytes.extend([byte_per_gpu * (1 - mem_per_comp_gpu)] * num_worker)
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
    
    requests = sorted(cache_gpu_required_bytes, key=lambda i: i[0], reverse=True)
    resources = np.asarray(cache_gpu_remaining_bytes)
    for i, request in enumerate(requests):
        remaining_requests = []
        if i < len(requests) - 1:
            remaining_requests = requests[i+1:]
        # get the best choice of cache gpus for request
        gpus_choice, _ = select_best_gpus_for_cache_request(request, resources, remaining_requests)
        if gpus_choice == None:
            # this exec plan combination is not valid
            return False
        # update the resources
        resources[gpus_choice] = resources[gpus_choice] - request


    # 3. delete the exec plan combinations which does not fully utilize the resources, i.e., the mem usage is less than 90%.
    if (resources > 0.1 * byte_per_gpu).any():
        return False
    
    return True





def _append_exec_plan(plan_groups, exec_plans_list, depth_i, tot_gpu_num, byte_per_gpu):
    '''
    Get all the possible exec plans with depth-first search.
    The initial plan_groups is [[]], i.e., containing a group with no exec plan.
    All plan groups are valid if they are put into plan_groups and returned.
    '''
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
    plan_groups.extend(new_plan_groups)
    _append_exec_plan(plan_groups, exec_plans_list, depth_i+1, tot_gpu_num, byte_per_gpu)




# def get_plan_group_str(plan_group):
#     '''
#     Get the string to represent a plan group
#     '''
#     return str(sorted([str(exec_plan) + str(exec_plan.model.get_left_flops_per_token()) for exec_plan in plan_group]))



def get_one_stage_exec_plans_sorted(model_list: List[MyModelInfor], gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    '''
    Get a set of exec plans which can work corrently on the given multi-GPU environment.
    '''
    # 1. first get the candidate exec plans for each model
    exec_plans_list = list()
    for model in model_list:
        exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu)
        exec_plans_list.append(exec_plans)
    # 
    plan_groups = [[]]
    _append_exec_plan(plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu)
    plan_groups = [MyExecPlanGroup(plan_group) for plan_group in plan_groups]

    # 2. delete plan_groups which do not occupy all comp gpus when there are models not executed
    # also for cases where there are idle comp resources, check whether increase comp resources can improve throughput, if yes, delete it.
    not_finished_model_num = sum([not model.is_finished() for model in model_list])
    useful_plan_groups = list()
    idle_comp_plan_groups = dict() # {models executed: (best_throughput, best_plan_group)}
    comp_throughput_given_plan_groups(plan_groups, gpu_name)
    for plan_group in plan_groups:
        key = tuple(sorted([(str(exec_plan.model), exec_plan.model.get_left_flops_per_token()) for exec_plan in plan_group.exec_plans]))
        if key not in idle_comp_plan_groups:
            idle_comp_plan_groups[key] = (plan_group.throughput, plan_group)
        else:
            if plan_group.throughput > idle_comp_plan_groups[key][0]:
                idle_comp_plan_groups[key] = (plan_group.throughput, plan_group)
    for _, plan_group in idle_comp_plan_groups.values():
        useful_plan_groups.append(plan_group)

        # else:
        #     print(f"inefficient plan group: {str(plan_group)}")

    # 3. delete redundant plan_groups (in case there are models that are the same)
    uniq_plan_groups_strs = set()
    uniq_plan_groups = list()
    for plan_group in useful_plan_groups:
        plan_group_str = str(plan_group)
        if plan_group_str not in uniq_plan_groups_strs:
            uniq_plan_groups_strs.add(plan_group_str)
            uniq_plan_groups.append(plan_group)
        # else:
        #     print(f"redundant plan group: {str(plan_group)}")


    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    for plan_group in uniq_plan_groups:
        print(str(plan_group))


    # sort plan groups according to the overall throughput
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.throughput, reverse=True)

    return uniq_plan_groups




# TODO move to class MyExecPlanGroup
def update_model_state(plan_group: MyExecPlanGroup, gpu_name: str) -> float:
    '''
    Update the model states after a comp stage.
    '''
    times = [exec_plan.estimate_exec_time(gpu_name) for exec_plan in plan_group.exec_plans]
    comp_time = min(times)
    for exec_plan, tot_time_required in zip(plan_group.exec_plans, times):
        exec_plan.model.update_left_flops_per_token(comp_time/tot_time_required)
    return comp_time


def recover_model_state(model_list:List[MyModelInfor], left_flops):
    for model, reset_v in zip(model_list, left_flops):
        model.reset_left_flops_per_token(reset_v)



def models_are_finished(model_list: List[MyModelInfor]):
    return False not in [model.is_finished() for model in model_list]


def get_model_states(model_list: List[MyModelInfor]):
    return str(sorted([(model.get_name(), model.get_left_flops_per_token()) for model in model_list]))



# we compute the best model execution plan for the given model list.
# assumption: the output lengths of all the models are the same.
# we can directly use a table for the cost model.
def _get_best_model_schedule(
        model_list: List[MyModelInfor], 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: set,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''
    if len(curr_group_seq.plan_group_seq) > len(model_list):
        assert False, f'{[model.left_flops_per_token for model in model_list]},'\
                    f'{[[str(_) for _ in group] for group in curr_group_seq]}'


    # exit the loop when all model are finished
    if models_are_finished(model_list):
        # update best plan seq info
        # print(f"curr_plan_seq return: {curr_plan_seq}")
        if curr_group_seq.get_tot_time() < best_group_seq.get_tot_time():
            best_group_seq.set_plan_group_and_time(curr_group_seq.plan_group_seq, curr_group_seq.time_seq)
        return 
    # 
    model_states = get_model_states(model_list)
    if model_states in uniq_model_states:
        # do not need to check this plan group, as we have check an equivalent one
        # print(f"redundant plan group due to model states: {model_states, str(curr_plan_seq)}")
        return
    else:
        # update uniq_model_states
        uniq_model_states.add(model_states)
    # print(f"curr_plan_seq: {curr_group_seq}")
    # print(f"model_states: {model_states}, uniq_model_states: {uniq_model_states}")
    plan_groups = get_one_stage_exec_plans_sorted(model_list, gpu_name, tot_gpu_num, byte_per_gpu)
    ori_left_flops = [model.get_left_flops_per_token() for model in model_list]
    for plan_group in plan_groups:
        # print(f"plan_group: {plan_group}")
        if len(plan_group) == 0:
            continue
        # update the remaining workload of all models after this stage
        recover_model_state(model_list, ori_left_flops)
        comp_time = update_model_state(plan_group, gpu_name)
        curr_group_seq.append_plan_group(plan_group)
        curr_group_seq.append_exec_time(comp_time)
        _get_best_model_schedule(model_list, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu)
        curr_group_seq.pop_one_stage()
    # print(f"best_group_seq: {best_group_seq}")





def get_best_model_schedule(
        model_list: List[MyModelInfor], 
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    curr_group_seq = MyExecPlanGroupSeq([], [])
    best_group_seq = MyExecPlanGroupSeq([None], [float('inf')])
    _get_best_model_schedule(model_list, curr_group_seq, best_group_seq, set(), gpu_name, tot_gpu_num, byte_per_gpu)
    return best_group_seq


        
'''
我们的整体计算逻辑如下：
1. 在一开始假定所有model的output length都一样，找到下一个stage的最佳exec plan组合。
2. 当一个stage结束之后，根据已有的model的运行信息，对不同model的output length进行重新评估。然后再重复step 1，计算当前的最佳exec plan组合。
试试效果吧，should be OK.
'''

# 吃完饭test一下。

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














# import time
# start = time.perf_counter()
# time.sleep(10)
# end = time.perf_counter()
# print(f"time sleep: {end-start}")



 
if __name__ == "__main__":
    # create the shared event
    # asyncio.run(main_with_preemption())
    # asyncio.run(main_test())
    # asyncio.run(main_test_load_model())
    # asyncio.run(main_with_preemption_debug())
    # asyncio.run(main_No_preemption())
    get_sample_dataset()