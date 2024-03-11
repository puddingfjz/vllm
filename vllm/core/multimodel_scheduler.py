""" Code for determine multimodel scheduling. """

# <jingzhi> To support multi-model inference
from multiprocessing import Array, Event
import os
from typing import List
from collections import deque
import time

from ray.util.placement_group import (
            placement_group_table,
            remove_placement_group,
            get_placement_group,
        )



from vllm.core.block_manager import KVBlkPerLayerWeight



class SHARED_CONTECT():
    shared_setting: Array = None
    shared_id: int = 0
    # should be 2+#model events, the first two events set by main process (stop all models, start all models), 
    # the other events set by the model processes.
    events: List[Event] = None
    gened_outputs = deque()
    remaining_requests = deque()
    shared_finish_status = [False] # store whether each model is finished
    # TODO (jingzhi) need to set this variable automatically
    execution_plan_size: int = 7


    @classmethod
    def should_reschedule(cls) -> bool:
        # check whether this model should stop inference and rescheduled
        # TODO (jingzhi) complete the schedule logic here.
        # 暂时先这么写，先不管reschedule怎么实现吧，因为我们需要知道是否需要允许最灵活的reschedule。
        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return False
        return cls.events[0].is_set()
    

    @classmethod
    def prepare_for_reschedule(cls, outputs, remaining_requests, llm_engine) -> None:
        # get the requests which have not been finished
        # get the new execution plan from shared_setting

        # <jingzhi> For Profiling
        start_wrapup = time.perf_counter()

        # print(f"event list status in prepare--model {cls.shared_id} 1 outputs {outputs}", flush=True)

        cls.gened_outputs.extend(outputs)

        # print(f"event list status in prepare--model {cls.shared_id} 1 remaining_requests {remaining_requests}", flush=True)

        cls.remaining_requests = remaining_requests

        print(f"event list status in prepare--model {cls.shared_id} 2", flush=True)

        # release occupied resources
        llm_engine.delete_workers()

        print(f"event list status in prepare--model {cls.shared_id} 3", flush=True)

        if llm_engine.parallel_config.world_size > 1:
            # do not need to remove placement group if there is only one worker
            pg_info = list(placement_group_table().values())[0]
            pg = get_placement_group(pg_info['name'])
            remove_placement_group(pg)

        print(f"event list status in prepare--model {cls.shared_id} 4", flush=True)

        # update global variable
        KVBlkPerLayerWeight.reset()

        print(f"event list status in prepare--model {cls.shared_id} 5", flush=True)

        # # set the event to notify the preparation is finished
        cls.events[cls.shared_id + 2].set()

        print(f"event list status in prepare--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")


        # <jingzhi> For Profiling
        end_wrapup = time.perf_counter()
        print(f"total time to wrap up for reschedule: {end_wrapup-start_wrapup}s")


    @classmethod
    def update_execution_plan(cls, tensor_parallel_size, gpu_memory_utilization):

        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return tensor_parallel_size, gpu_memory_utilization

        # currently, the execution plan of a model is controlled by four parameters
        # TODO (jingzhi) we may add another parameter to control the number of cache gpus

        offset = cls.shared_id * cls.execution_plan_size

        # for v in cls.shared_setting:
        #     print(v)

        tensor_parallel_size: int = cls.shared_setting[offset]
        gpu_memory_utilization: float = cls.shared_setting[offset+1]/10
        weight_load_degree: int = cls.shared_setting[offset+2]
        gpus: str = ','.join([str(i) for i in cls.shared_setting[offset+3:offset+7]])

        os.environ['WEIGHT_LOAD_DEGREE'] = str(weight_load_degree)

        # TODO (jingzhi) this may not work because we have initialized torch.cuda when we reload the model
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        return tensor_parallel_size, gpu_memory_utilization

    
    @classmethod
    def set_execution_plan(cls, setting, model_ids=list()) -> None:
        '''
            If model_ids is given, only change the setting for specific models
        '''
        iter_range = None
        if len(model_ids) == 0:
            iter_range = range(len(cls.shared_setting))
        else:
            iter_range = list()
            for model_i in model_ids:
                iter_range.extend(range(cls.execution_plan_size * model_i, cls.execution_plan_size * (model_i+1)))
        
        # print(f"setting: {setting}, len(setting): {len(setting)}  len(shared_setting): {len(cls.shared_setting)} iter_range:{iter_range}")
        
        for i, v in zip(iter_range, setting):
            cls.shared_setting[i] = v



    @classmethod
    def sync_before_loading_model(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return
        
        print(f"model_i: {SHARED_CONTECT.shared_id}, os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}, SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)

        # # set the event to notify the preparation is finished
        # cls.events[cls.shared_id + 2].set()
        # wait for the signal to start a new execution plan
        print(f"event list status in sync before loading--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")
        cls.events[1].wait()
    




    @classmethod
    def is_finished(cls) -> None:
        return cls.shared_finish_status[cls.shared_id]
    

    @classmethod
    def set_finished(cls, finished: bool) -> None:
        if finished:
            cls.shared_finish_status[cls.shared_id] = finished
            if os.environ['RUN_MULTI_MODEL'] == 'True':
                cls.events[cls.shared_id + 2].set()

        if os.environ['RUN_MULTI_MODEL'] == 'True':
            print(f"event list status in set finish--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")




    @classmethod
    def wait_all_models_to_finish(cls) -> None:
        assert os.environ['RUN_MULTI_MODEL'] == 'True'
        # wait for all the models to finish

        print(f"event list status in waiting all to finish 1: {[event.is_set() for event in cls.events[2:]]}")

        for event in cls.events[2:]:
            event.wait()

        print(f"event list status in waiting all to finish 2: {[event.is_set() for event in cls.events[2:]]}")



    @classmethod
    def restart_all_models(cls) -> None:
        assert os.environ['RUN_MULTI_MODEL'] == 'True'
        # clear the stop signal
        cls.events[0].clear()
        # clear all model events
        for event, finished in zip(cls.events[2:], cls.shared_finish_status):
            if not finished:
                event.clear()
        # set the start signal
        cls.events[1].set()

        print(f"event list status in restart--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")



    # ===============================================================================
    # ===============================================================================
    # ===============================================================================
    # ===============================================================================
    # the functions below are not used
    @classmethod
    def init_shared_vars(cls, model_setting: Array) -> None:
        cls.shared_setting = model_setting


    @classmethod
    def increment(cls) -> None:
        with cls.shared_setting.get_lock():
            cls.shared_setting[cls.shared_id] += 1

    @classmethod
    def test_and_print(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] == 'True':
            print(cls.shared_setting[cls.shared_id])
            cls.increment()        



    @classmethod
    def test_task(cls, identifier=-1):
        # test whether event can work
        # wait for the event to be set
        if identifier == -1:
            identifier = cls.shared_id
        print(f'Task {identifier} waiting...', flush=True)
        cls.events[cls.shared_id].wait()
        # generate a value
        value = 1
        # block for a moment
        # sleep(value)
        # report a message
        print(f'Task {identifier} completed with {value}', flush=True)



# shared_setting: Array = None # = Array('d', [-1, -1])

# def init_shared_vars(model_setting: Array):
#     SHARED_CONTECT.shared_setting = model_setting
 
 
# def increment():
#     with SHARED_CONTECT.shared_setting.get_lock():
#         SHARED_CONTECT.shared_setting[SHARED_CONTECT.shared_id] += 1



