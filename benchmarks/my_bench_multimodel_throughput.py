""" Benchmark offline multi-model inference throughput. """

'''
Basic idea: we use asyncio with process pool + shared variable among process.

When there is a model finished, the main process here will determine the new best execution plan for the remaining
models and notify them through the shared variables.

In each model inference process, it will check the shared varibles to see whether it need to change its execution plan.


from my_bench_multimodel_throughput import *


'''




from concurrent.futures import ProcessPoolExecutor
import asyncio
from multiprocessing import Array, Event

from vllm.core.multimodel_scheduler import SHARED_CONTECT
import benchmark_throughput

import time

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

    loop = asyncio.get_running_loop()
    tasks = []

    counter = Array('i', [0 for i in range(2*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
    # all child processors will inherit this event
    SHARED_CONTECT.events = [Event() for _ in range(2+2)]
    # set the event to allow models to run
    # SHARED_CONTECT.events[1].set()
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(2)])


    model_list = ['NousResearch/Llama-2-13b-hf', 'NousResearch/Llama-2-70b-hf']


    # set the initial execution plan
    # TODO (jingzhi) add a function to compute the current best execution plan for all models
    # new_setting = [2, 7, 2, 0,1,2,3] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    # new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    new_setting = [2, 8, 2, 1,0,2,3] # model 0: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    new_setting.extend([2, 9, 16, 2,3,0,1]) # model 1
    SHARED_CONTECT.set_execution_plan(new_setting)



    with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
        for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
            tasks.append(loop.run_in_executor(executor, start_a_model_inference, gpus, model_id, model))


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
    os.environ['CUDA_VISIBLE_DEVICES']='2,1,3,0'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
    os.environ['USE_VLLM']='False'
    os.environ['RUN_MULTI_MODEL'] = 'False'

    import time

    from vllm import LLM
    from ray.util import remove_placement_group

    os.environ['WEIGHT_LOAD_DEGREE'] = '16'

    start_time = time.perf_counter()
    # huggyllama/llama-7b   NousResearch/Llama-2-70b-hf
    llm = LLM(model='huggyllama/llama-7b', enforce_eager=True, tensor_parallel_size=1)
    end_time = time.perf_counter()
    print(f"total initialization time: {end_time - start_time}")





# import time
# start = time.perf_counter()
# time.sleep(10)
# end = time.perf_counter()
# print(f"time sleep: {end-start}")



 
if __name__ == "__main__":
    # create the shared event
    asyncio.run(main_with_preemption())
    # asyncio.run(main_test())
    # asyncio.run(main_test_load_model())

