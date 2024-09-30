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
from vllm.core.multimodel_scheduler import MyManager

import torch




@ray.remote
class RAY_WORKER():
    """A ray actor class used to communicate between different LLMs for input/output passing.
    """
    def __init__(self, communicator):
        # prepare a request pipe for each model in the LLM system
        print(f"\ninit a ray worker--------------\n")
        self.communicator = communicator
        communicator.add_seqs(0, [(2, 'this'), (3, 'is')])
        print(f"\nfinish init a ray worker--------------\n")





def init_task(task_i: int, communicator):
    try:
        model_communicate_utils.communicator = communicator
        if task_i == 0:
            # communicator.add_seqs(0, ['this', 'is'])
            # ret = communicator.get_seqs.remote(0, 1)
            # print(ray.get(ret))
            # communicator.add_seqs(0, ['this', 'is'])
            communicator.reset_state_for_model(0, 2)
            communicator.add_seqs(0, [(0, 'this'), (1, 'is')])
            print(f"\nRAY_WORKER.remote(communicator)--------------\n")
            worker = RAY_WORKER.remote(communicator)
            # communicator.add_seqs(0, [TEST(1,2,3), TEST(4,5,6)])
            print(f"{task_i}, finished")
        else:
            # ret = communicator.get_seqs.remote(0, 1)
            # print(ray.get(ret))
            ret = communicator.get_seqs(1, 1, 1)
            print(f"{task_i}, ret: {ret}")
    except Exception as e:
        print(task_i, e)







# 看看能不能注册一个自己的class作为manager，用起来能更方便一点
async def test():

    import time
    print(time.perf_counter())
    loop = asyncio.get_running_loop()
    print(time.perf_counter())
    tasks = []
    # communicator = REMOTE_LLM_COMMUNICATOR.remote(2)
    
    with MyManager() as manager:
        communicator = manager.Communicator(2, in_edge_dict_with_dummy_inp_nodes={1:[0]})
        # communicator.add_seqs(0, [(2, 'this'), (3, 'is')])
        # communicator.add_seqs(0, ["hh"])
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


# from model_communicate import *
asyncio.run(test())

'''
from model_communicate import *
asyncio.run(test())
# test()
'''