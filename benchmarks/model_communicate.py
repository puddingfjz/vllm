"""
    This file contains functions about the communication between models, e.g., a model's output is another model's input.

    Class definition has been removed to multimodel_scheduler.py    
"""
# from typing import List, Optional, Tuple, Dict
# from collections import defaultdict

# from multiprocessing.managers import BaseManager





# class LLM_COMMUNICATOR:
#     """A ray actor class used to communicate between different LLMs for input/output passing.
#     """
#     def __init__(self, model_num: int):
#         # prepare a request pipe for each model in the LLM system
#         self.req_pool: Dict[int, List[str]] = {model_id: list() for model_id in range(model_num)}
#         self.fetched_nums: Dict[Tuple[int,int], int] = defaultdict(int)
#         # self.req_pool[0] = ['this', 'is']

#     def add_seqs(self, model_id: int, seqs: List[str]):
#         self.req_pool[model_id].extend(seqs)
    
#     def get_seqs(self, from_model_id: int, to_model_id):
#         start = self.fetched_nums[(from_model_id, to_model_id)]
#         reqs = self.req_pool[from_model_id]
#         ret = reqs[start:]
#         self.fetched_nums[(from_model_id, to_model_id)] = len(reqs)
#         return ret
#     def get_info(self):
#         return self.req_pool, self.fetched_nums
    




# class MyManager(BaseManager):
#     pass

# MyManager.register('Communicator', LLM_COMMUNICATOR)















# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# BELOW IS TEST CODE
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================

from concurrent.futures import ProcessPoolExecutor
import asyncio
from multiprocessing import Array, Event, Manager
from multiprocessing.managers import BaseManager
import ctypes


from vllm.engine.ray_utils import ray
from typing import List, Optional, Tuple, Dict
from collections import defaultdict


import model_communicate_utils
from model_communicate_utils import MyManager, TEST

import torch






# # @ray.remote
# class REMOTE_LLM_COMMUNICATOR:
#     """A ray actor class used to communicate between different LLMs for input/output passing.
#     """
#     def __init__(self, model_num: int):
#         # prepare a request pipe for each model in the LLM system
#         self.req_pool: Dict[int, List[str]] = {model_id: list() for model_id in range(model_num)}
#         self.fetched_nums: Dict[Tuple[int,int], int] = defaultdict(int)
#         self.req_pool[0] = ['this', 'is']

#     def add_seqs(self, model_id: int, seqs: List[str]):
#         self.req_pool[model_id].extend(seqs)
    
#     def get_seqs(self, from_model_id: int, to_model_id):
#         start = self.fetched_nums[(from_model_id, to_model_id)]
#         reqs = self.req_pool[from_model_id]
#         ret = reqs[start:]
#         self.fetched_nums[(from_model_id, to_model_id)] = len(reqs)
#         return ret
#     def get_info(self):
#         return self.req_pool, self.fetched_nums
    


def add_seqs(req_pool, model_id: int, seqs: List[str]):
    req_pool[model_id].extend(seqs)
    print("1 req_pool ", req_pool)


def get_seqs(req_pool, fetched_nums, from_model_id: int, to_model_id):
    start = fetched_nums[(from_model_id, to_model_id)]
    reqs = req_pool[from_model_id]
    ret = reqs[start:]
    fetched_nums[(from_model_id, to_model_id)] = len(reqs)
    print("2 ret ", ret)
    return ret


def init_communicator(req_pool, fetched_nums, out_edge_dict):
    for src, tgts in out_edge_dict.items():
        req_pool[src] = list()
        for tgt in tgts:
            fetched_nums[(src, tgt)] = 0
    


# 下面的代码先来测试一下multiprocessing起的进程能不能继承ray worker，
# 还是说我们要在init multiprocessing的进程的时候把ray worker当做参数传进去。

# communicator = REMOTE_LLM_COMMUNICATOR.remote(2)

# def init_task(task_i: int, req_pool, fetched_nums, l):
#     try:
#         if task_i == 0:
#             # communicator.add_seqs(0, ['this', 'is'])
#             # ret = communicator.get_seqs.remote(0, 1)
#             # print(ray.get(ret))
#             add_seqs(req_pool, 0, ['this', 'is'])
#             print(task_i, req_pool, fetched_nums)
#         else:
#             # ret = communicator.get_seqs.remote(0, 1)
#             # print(ray.get(ret))
#             ret = get_seqs(req_pool, fetched_nums, 0, 1)
#             print(task_i, req_pool, fetched_nums)
#     except Exception as e:
#         print(task_i, e)
    
#     l.append(task_i)




def init_task(task_i: int, communicator):
    try:
        model_communicate_utils.communicator = communicator
        if task_i == 0:
            # communicator.add_seqs(0, ['this', 'is'])
            # ret = communicator.get_seqs.remote(0, 1)
            # print(ray.get(ret))
            # communicator.add_seqs(0, ['this', 'is'])
            communicator.add_seqs(0, [TEST(1,2,3), TEST(4,5,6)])
            print(f"{task_i}, finished")
        else:
            # ret = communicator.get_seqs.remote(0, 1)
            # print(ray.get(ret))
            ret = communicator.get_seqs(0, 1)
            print(f"{task_i}, ret: {ret}")
    except Exception as e:
        print(task_i, e)






@ray.remote
class MODEL_MAIN_PROCESS():
    """A ray actor class used to communicate between different LLMs for input/output passing.
    """
    def __init__(self, main_model_process_id: int, communicator):
        # prepare a request pipe for each model in the LLM system
        from model_communicate_utils import shared_finish_status
        print(f"main_model_process_id: {main_model_process_id}, {shared_finish_status[0]}")

        import os
        os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4'

        a = torch.Tensor([1, 2, 3]).to(0)
        a = torch.Tensor([1, 2, 3]).to(1)
        a = torch.Tensor([1, 2, 3]).to(2)
        a = torch.Tensor([1, 2, 3]).to(3)
        self.communicator = communicator
        self.process_i = main_model_process_id

        
    def init_task(self):
        try:
            if self.process_i == 0:
                self.communicator.add_seqs.remote(0, ['this', 'is'])
            else:
                ret = self.communicator.get_seqs.remote(0, 1)
                print(ray.get(ret))
        except Exception as e:
            print(self.process_i, e)        



# async def test():


#     loop = asyncio.get_running_loop()
#     tasks = []
#     # communicator = REMOTE_LLM_COMMUNICATOR.remote(2)
    
#     with Manager() as manager:
#         req_pool = dict()
#         fetched_nums = dict()
#         out_edge_dict = {0:[1]}
#         for src, tgts in out_edge_dict.items():
#             req_pool[src] = list()
#             for tgt in tgts:
#                 fetched_nums[(src, tgt)] = 0
#         print(req_pool)
#         print(fetched_nums)
#         req_pool = manager.dict(req_pool)
#         fetched_nums = manager.dict(fetched_nums)
#         # init_communicator(req_pool, fetched_nums, {0:[1]})
#         print(req_pool)
#         print(fetched_nums)

#         l = manager.list()

#         # launch the exec_plans in order
#         with ProcessPoolExecutor(max_workers=2) as executor:
#             # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
            
#             # start a process for each model, no matter it is in launched_exec_plan_states or not
#             # NOTE: we will use os.environ['TOT_ORDERED_GPUS'] to control the actual gpu order in each model to support reschedule
#             # for model_id, model_path in enumerate(model_paths):
#             for i in range(2):
#                 tasks.append(
#                     loop.run_in_executor(
#                         executor, init_task, i, req_pool, fetched_nums, l,
#                     )        
#                 )
        



#         print(l)




# class MyManager(BaseManager):
#     pass

# MyManager.register('Communicator', REMOTE_LLM_COMMUNICATOR)




# 看看能不能注册一个自己的class作为manager，用起来能更方便一点
async def test():


    loop = asyncio.get_running_loop()
    tasks = []
    # communicator = REMOTE_LLM_COMMUNICATOR.remote(2)
    
    with MyManager() as manager:
        communicator = manager.Communicator(2)
        print(f"at the beginning: {communicator.get_info()}")
        print(f"at the beginning: {[str(i) for i in communicator.get_info()[0][0]]}")

        # launch the exec_plans in order
        with ProcessPoolExecutor(max_workers=2) as executor:
            # for model_id, (gpus, model) in enumerate(zip(['2,1,3,0', '3,0,2,1'], model_list)):
            
            # start a process for each model, no matter it is in launched_exec_plan_states or not
            # NOTE: we will use os.environ['TOT_ORDERED_GPUS'] to control the actual gpu order in each model to support reschedule
            # for model_id, model_path in enumerate(model_paths):
            for i in range(2):
                tasks.append(
                    loop.run_in_executor(
                        executor, init_task, i, communicator
                    )        
                )
        
        print(f"in the end: {communicator.get_info()}")
        print(f"in the end: {[str(i) for i in communicator.get_info()[0][0]]}")





# def test():


#     from model_communicate_utils import shared_finish_status
#     shared_finish_status[0] = True
    
#     communicator = REMOTE_LLM_COMMUNICATOR.remote(2)


#     # create ray worker for model main process
#     processes = [MODEL_MAIN_PROCESS.remote(process_i, communicator) for process_i in range(2)]
#     ray.get([process.init_task.remote() for process in processes])

#     # import time
#     # time.sleep(60)

'''
from model_communicate import *
asyncio.run(test())
# test()
'''