"""
    This file contains functions about the communication between models, e.g., a model's output is another model's input.
"""

from concurrent.futures import ProcessPoolExecutor
import asyncio
from multiprocessing import Array, Event, Manager
from multiprocessing.managers import BaseManager
import ctypes


from vllm.engine.ray_utils import ray
from typing import List, Optional, Tuple, Dict
from collections import defaultdict


import torch





# @ray.remote
class TEST:
    """A ray actor class used to communicate between different LLMs for input/output passing.
    """
    def __init__(self, a, b, c):
        # prepare a request pipe for each model in the LLM system
        self.a = a
        self.b = b
        self.c = c

    def __str__(self) -> str:
        return f"{self.a, self.b, self.c}"


# @ray.remote
class REMOTE_LLM_COMMUNICATOR:
    """A ray actor class used to communicate between different LLMs for input/output passing.
    """
    def __init__(self, model_num: int):
        # prepare a request pipe for each model in the LLM system
        self.req_pool: Dict[int, List[TEST]] = {model_id: list() for model_id in range(model_num)}
        self.fetched_nums: Dict[Tuple[int,int], int] = defaultdict(int)
        self.req_pool[0] = ['this', 'is']

    def add_seqs(self, model_id: int, seqs: List[TEST]):
        self.req_pool[model_id].extend(seqs)
    
    def get_seqs(self, from_model_id: int, to_model_id):
        start = self.fetched_nums[(from_model_id, to_model_id)]
        reqs = self.req_pool[from_model_id]
        ret = reqs[start:]
        self.fetched_nums[(from_model_id, to_model_id)] = len(reqs)
        return ret
    def get_info(self):
        return self.req_pool, self.fetched_nums
    

class MyManager(BaseManager):
    pass

MyManager.register('Communicator', REMOTE_LLM_COMMUNICATOR)


communicator = None