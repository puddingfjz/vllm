""" Code for determine multimodel scheduling. """

# <jingzhi> To support multi-model inference
from multiprocessing import Array, Event, Lock
import os
from typing import List, Optional, Tuple, Dict, Union
from collections import deque, defaultdict
import time

from ray.util.placement_group import (
            placement_group_table,
            remove_placement_group,
            get_placement_group,
        )



from vllm.core.block_manager import KVBlkPerLayerWeight


# <jingzhi> to support data parallel
from vllm.engine.ray_utils import ray



# <jingzhi> to communicate between models
from multiprocessing.managers import BaseManager
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams



class DummyLock:
    """
        A dummy lock which does nothing. Used by LLM_COMMUNICATOR.
    """
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self,*args):
        pass




class LLM_COMMUNICATOR:
    """
        A class used to communicate between different LLMs for input/output passing.
        NOTE:
            1. When a model only has one input model, the fused_inp_queue is the same as the queue in output_pool.
            2. for the models without input models, we also prepare their inputs in the output pool as the output of a dummy model.
            3. self._read_output_nums[model_1, model_2] records the number of seqs in self.output_pool of model_1 that has been read by model_2
            4. self.fetched_fused_inp_start[to_model_id][dp_i] records the number of complete inps that has been read by to_model's dp_i worker.
                this variable will be RESET every time the model is restarted.
            5. in ``in_edge_dict_with_dummy_inp_nodes``, we use NEGATIVE model_ids to represent DUMMY inp nodes.
    """
    def __init__(
            self, model_num: int, 
            in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]],
            # 
            base_model_ids_dict: Dict[int, List[int]],
            inp_req_ids: Dict[int, Dict[int, List[int]]],
            out_req_id_mapping: Dict[int, Dict[int, Dict[int, Tuple[int, int]]]],
            new_out_req_part_num: Dict[int, int],
            independent_srcs: Dict[int, bool],
            ):
        """
            NOTE:
                1. The input parameters: 
                    in_edge_dict_with_dummy_inp_nodes, inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs are all
                    about original base models.
                2. model_num: the number of models in the system which may contain fused models
        """
        # prepare a request pipe for each model in the LLM system
        # the output may be str or List[int] (i.e., token ids directly); we also record the seq id of each seq
        # self.output_pool: Dict[int, Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]] = {model_id: list() for model_id in range(model_num)}
        self.output_pool: Dict[int, Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]] = defaultdict(list)
        # _read_output_nums key: [from_model_id, to_model_id]
        self._read_output_nums: Dict[Tuple[int, int], int] = defaultdict(int)
        # records the number of avaialble srcs for each seq for a given model [to_model_id: [seq_id, (#available srcs, available src seqs)]]
        self._available_srcs: Dict[int, Dict[int, Tuple[int, Union[List[str],List[List[int]]]]]] = defaultdict(dict) #\
            # {model_id: dict() for model_id in range(model_num)}
        self.fused_inp_queues: Dict[int, Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]] = defaultdict(list) #\
            # {model_id: list() for model_id in range(model_num)} # value: a queue of [req_id, complete inp (str or token_id_list)]
        self.fetched_fused_inp_start: Dict[int, List[int]] = defaultdict(list) # key: (to_model_id) value: the fetched number for each dp worker
        self.in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]] = in_edge_dict_with_dummy_inp_nodes
        # self.output_pool[0] = ['this', 'is']
        self._unavailable_req_nums: Dict[int, int] = dict()
        self._ungened_out_req_nums: Dict[int, int] = dict()

        
        # ----------------- support fused models (horizontal and vertical fusion) ----------------------------------------------------
        
        # {fused_model shared_id: [base model ids]}
        self.base_model_ids_dict: Dict[int, List[int]] = base_model_ids_dict

        # store the req ids to be fetched from each inp model {to_model_id: {from_model_id: req_ids}}
        # NOTE: not every model has information stored in this dict, 
        # e.g., the models taking in all reqs from a from_model; the models which will process the reqs from a from_model before doing inference
        # init at the beginning
        # NOTE: req_ids is a set
        self.inp_req_ids: Dict[int, Dict[int, List[int]]] = inp_req_ids

        # store they way to reorganize out reqs from a model {model_id: {out_req_id: (new_out_req_id, ind)}}
        # NOTE: not every model has information stored in this dict, 
        # this is only used in the case of horizontal fusion, e.g., map reduce, where we different req ids are supporsed to be concat to be a new req
        # init at the beginning
        self.out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]] = out_req_id_mapping
        # stores how many ori out req can constitute a new out req {model_id: {new_out_req_id: tot_part_num}}
        self.new_out_req_part_num: Dict[int, Dict[int, int]] = new_out_req_part_num
        # stores the out reqs before merge them into new out reqs if necessary {model_id: {new_out_req_id: (#available out parts, )}}
        self._meta_output_pool: Dict[int, Union[Dict[int, Tuple[int, List[str]]], Dict[int, Tuple[int, List[List[int]]]]]] = dict()
        self._init_meta_output_pool()

        # stores whether a base model's input sources are independent, if independent, they will not be fused before inference
        self._independent_srcs: Dict[int, bool] = independent_srcs

        # used for the fused models, to store currently each req belongs to which base model: {fused_model_id: {req_id: base_model_id}}
        self.out_req_model_id_mapping: Dict[int, Dict[int, int]] = defaultdict(dict)


        # we need two kinds of locks: 
        # (1) lock for adding reqs to output_pool,
        # (2) lock for fetching reqs.
        # NOTE: for dummpy inp nodes, their model ids are negative numbers
        # NOTE: one lock for each model (base model or fused model)
        self._add_req_locks: Dict[int, Lock] = {_: Lock() for _ in in_edge_dict_with_dummy_inp_nodes}
        self._fetch_req_locks: Dict[int, Lock] = {_: Lock() for _ in in_edge_dict_with_dummy_inp_nodes}
        # self._add_req_locks: List[Lock] = [Lock() for _ in range(model_num)]
        # self._fetch_req_locks: List[Lock] = [Lock() for _ in range(model_num)]



        print(f"In INIT Communicator:")
        print(f"self._add_req_locks: {self._add_req_locks}")
        print(f"self._fetch_req_locks: {self._fetch_req_locks}")




    def fuse_inp_srcs(self, src_seqs):
        ret = src_seqs[0]
        for src in src_seqs[1:]:
            ret += src
        return ret




    def _init_meta_output_pool(self):
        """
            Init self._meta_output_pool according to self.out_req_id_mapping, self.new_out_req_part_num.
        """
        for model_id in self.new_out_req_part_num:
            self._meta_output_pool[model_id] = dict()
            for new_out_req_id, tot_part_num in self.new_out_req_part_num[model_id].items():
                self._meta_output_pool[model_id][new_out_req_id] = [0, [None for _ in range(tot_part_num)]]
    



    def _get_correct_lock(
            self, model_id: int, lock_dict: Dict[int, Lock], read_only: bool
        ) -> Union[DummyLock, Lock]:
        """
            Get the correct lock from the lock list (if any).
            NOTE:
                1. if to get read_only _add_req_locks and the inp models have sent out all out reqs,
                    return a dummy lock.
        """
        # print(f"in _get_correct_lock, model_id: {model_id}, lock_dict: {lock_dict}, read_only: {read_only}")

        if model_id < 0:
            return DummyLock()
        if (lock_dict == self._add_req_locks) and read_only \
            and (self._ungened_out_req_nums[model_id] == 0):
            return DummyLock()
        return lock_dict[model_id]
        



    def init_unavailable_req_nums_and_ungened_out_req_nums(
            self, unavailable_req_nums: Dict[int, int], ungened_out_req_nums: Dict[int, int]):
        """
            NOTE: do not need lock here, because it is called before launching LLM infer processes.
        """
        self._unavailable_req_nums.update(unavailable_req_nums)
        self._ungened_out_req_nums.update(ungened_out_req_nums)



    def _reset_state_for_model(self, model_id: int, dp_size: int):
        """
            When a model is restarted, we need to update some variables.
            Update: self.fused_inp_queues, self.fetched_fused_inp_start.
            NOTE: 
                1. do not need lock here, will add lock where this method is called.
                2. the input model_id can be of fused models or base models, but 
                ``fused_inp_queues`` and ``fetched_fused_inp_start`` are for base models only.
        """
        print(f"In reset_state_for_model, parameters: {model_id, dp_size}-----------------\n")
        # first, we need to get the remaining reqs
        old_dp_size = len(self.fetched_fused_inp_start[model_id])
        remaining = list()
        for dp_worker_fetched_start in self.fetched_fused_inp_start[model_id]:
            remaining.extend(self.fused_inp_queues[model_id][dp_worker_fetched_start::old_dp_size])
        
        remaining = sorted(remaining, key=lambda inp: inp[0])
        self.fused_inp_queues[model_id] = remaining
        self.fetched_fused_inp_start[model_id] = [i for i in range(dp_size)]
        # print(f"In reset_state_for_model, parameters: {model_id, dp_size} self.fetched_fused_inp_start: {self.fetched_fused_inp_start} self.fused_inp_queues: {self.fused_inp_queues}\n")




    def reset_state_for_model(self, model_id: int, dp_size: int):
        """
            When a model is restarted, we need to update some variables.
            Update: self.fused_inp_queues, self.fetched_fused_inp_start.
            NOTE: 
                1. do not need lock here, will add lock where this method is called.
                2. the input model_id can be of fused models or base models, but 
                ``fused_inp_queues`` and ``fetched_fused_inp_start`` are for base models only.
        """
        # print(f"In reset_state_for_model, parameters: {model_id, dp_size}-----------------\n")
        if model_id in self.base_model_ids_dict:
            # this model is a fused model
            for base_model_id in self.base_model_ids_dict[model_id]:
                self._reset_state_for_model(base_model_id, dp_size)
        else:
            self._reset_state_for_model(model_id, dp_size)





    def _process_outputs(self, model_id: int, seqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]):
        """
            Process the outputs, e.g., concat different ori seqs into new out seqs.
            May be called when horizontally fusing models, e.g., Map-reduce Summary.
        """
        ret = list()
        for req_id, req_content in seqs:
            new_out_id, ind = self.out_req_id_mapping[model_id][req_id]
            self._meta_output_pool[model_id][new_out_id][1][ind] = req_content
            self._meta_output_pool[model_id][new_out_id][0] = self._meta_output_pool[model_id][new_out_id][0] + 1
            if self._meta_output_pool[model_id][new_out_id][0] == len(self._meta_output_pool[model_id][new_out_id][1]):
                # this new out req is ready
                new_out_content = self.fuse_inp_srcs(self._meta_output_pool[model_id][new_out_id][1])
                ret.append((new_out_id, new_out_content))
        return ret





    def add_seqs_base_model(self, model_id: int, seqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]):
        """
            Send newly generated outputs to the model communicator.
            We all sort the input outputs by their req id.
            We may also consider to sort them by their lengths instead. 
            --> better not? because if there are more than one inp models for a target model, only receiving all outputs from 
            the input models can we obtain a complete input req for the target model.

            INPUT:
                seqs: list of (req_id, req_content); req_content may be generated text or generated token ids.
            Modify:
                self.output_pool, self._ungened_out_req_nums
            NOTE: 
                1. ensure the input ``seqs`` only contains seqs which have not been added before.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
            NOTE:
                1. this function is only for base models.
        """

        to_add = seqs        
        # # sort the seqs by their seq ids ==> we do not need to sort here, because we will sort when adding them to fused_inp_queues.
        # to_add = sorted(to_add, key=lambda x: int(x.request_id))

        with self._get_correct_lock(model_id, self._add_req_locks, read_only=False):

            if model_id in self.new_out_req_part_num:
                to_add = self._process_outputs(model_id=model_id, seqs=seqs)

            self.output_pool[model_id].extend(to_add)
            # update self._ungened_out_req_nums
            self._ungened_out_req_nums[model_id] = self._ungened_out_req_nums[model_id] - len(to_add)
        
            print(f"ADDING SEQS of model {model_id} TO THE POOL:")
            for _ in to_add:
                print(_)
            with open(f"./test_end2end_schedule/model_IO.log", 'a') as file:
                for _ in to_add:
                    file.write(f"Add: base model id: {model_id}: {str(_)}\n")

        # below is the old version which does not sort the seqs ----------------------
        # if contain_old_seqs:
        #     old_num = len(self.output_pool[model_id])
        #     self.output_pool[model_id].extend(seqs[old_num:])
        # else:
        #     self.output_pool[model_id].extend(seqs)



    def add_seqs(self, model_id: int, seqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]):
        """
            Send newly generated outputs to the model communicator.
            We all sort the input outputs by their req id.
            We may also consider to sort them by their lengths instead. 
            --> better not? because if there are more than one inp models for a target model, only receiving all outputs from 
            the input models can we obtain a complete input req for the target model.

            INPUT:
                seqs: list of (req_id, req_content); req_content may be generated text or generated token ids.
            Modify:
                self.output_pool, self._ungened_out_req_nums
            NOTE: 
                1. ensure the input ``seqs`` only contains seqs which have not been added before.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
            NOTE:
                1. this function is only for base models.
        """
        if model_id not in self.base_model_ids_dict:
            self.add_seqs_base_model(model_id=model_id, seqs=seqs)
        else:
            # 1. group seqs by base model ids
            base_model_seqs_dict = defaultdict(list)
            for seq in seqs:
                req_id = seq[0]
                base_model_id = self.out_req_model_id_mapping[model_id][req_id]
                base_model_seqs_dict[base_model_id].append(seq)
            
            # 2. add seqs to each base model
            for base_model_id, base_model_seqs in base_model_seqs_dict.items():
                self.add_seqs_base_model(model_id=base_model_id, seqs=base_model_seqs)



    

    def _get_selected_outputs(
            self, from_model_id: int, to_model_id: int,
    ):
        """
            Try to fetch available outputs from the model whose id is ``model_id``, 
            where the ids of the outputs are in req_ids_requested.
            NOTE: 1. assume the corresponding lock has been obtained.
        """
        end = len(self.output_pool[from_model_id])
        start = self._read_output_nums[(from_model_id, to_model_id)]
        self._read_output_nums[(from_model_id, to_model_id)] = end
        
        if (to_model_id in self.inp_req_ids) and (from_model_id in self.inp_req_ids[to_model_id]):
            inp_req_ids = self.inp_req_ids[to_model_id][from_model_id]
            ret = [req for req in self.output_pool[from_model_id][start:end] if req[0] in inp_req_ids]
            return ret
        else:
            # return all the avaliable reqs
            return self.output_pool[from_model_id][start:end]
    
        










    def _get_fused_inp_queues_limited_version(self, to_model_id: int):
        """
            Update self.fused_inp_queues based on the new outputs from inp models of the to_model.
            Modify:
                self._read_output_nums, self._available_srcs, self.fused_inp_queues, self._unavailable_req_nums
            NOTE:
                1. need lock here, as there may be multiple dp workers: one lock for one model.
            NOTE:
                1. this version does not support model horizontal fusion.
        """

        if self._unavailable_req_nums[to_model_id] == 0:
            # if all inp reqs have been fetched, directly return
            return

        # we need first update self.fused_inp_queues
        inps = self.in_edge_dict_with_dummy_inp_nodes[to_model_id]
        if len(inps) == 1:
            # first get a lock
            with self._get_correct_lock(inps[0], self._add_req_locks, read_only=True):
                end = len(self.output_pool[inps[0]])
                start = self._read_output_nums[(inps[0], to_model_id)]
                reqs = self.output_pool[inps[0]][start:end]
                # sort the reqs by their req ids
                reqs = sorted(reqs, key=lambda x: x[0])
            self.fused_inp_queues[to_model_id].extend(reqs)
            self._read_output_nums[(inps[0], to_model_id)] = end
            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)
        else:
            # there are more than one inp src for the to_model
            new_complete_req_ids = list()
            inp_src_num = len(inps)
            for inp_i, from_model_id in enumerate(inps):
                
                # first get a lock
                with self._get_correct_lock(from_model_id, self._add_req_locks, read_only=True):

                    end = len(self.output_pool[from_model_id])
                    start = self._read_output_nums[(from_model_id, to_model_id)]
                    for req in self.output_pool[from_model_id][start:end]:
                        if req[0] not in self._available_srcs[to_model_id]:
                            self._available_srcs[to_model_id][req[0]] = [0, [None for _ in range(inp_src_num)]]
                        self._available_srcs[to_model_id][req[0]][0] += 1
                        self._available_srcs[to_model_id][req[0]][1][inp_i] = req[1]
                        if self._available_srcs[to_model_id][req[0]][0] == inp_src_num:
                            new_complete_req_ids.append(req[0])
                self._read_output_nums[(from_model_id, to_model_id)] = end
            new_complete_req_ids = sorted(new_complete_req_ids)
            reqs = [(req_id, self.fuse_inp_srcs(self._available_srcs[to_model_id][req_id][1])) for req_id in new_complete_req_ids]
            self.fused_inp_queues[to_model_id].extend(reqs)
            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)






    def _get_fused_inp_queues(self, to_model_id: int):
        """
            Update self.fused_inp_queues based on the new outputs from inp models of the to_model.
            Modify:
                self._read_output_nums, self._available_srcs, self.fused_inp_queues, self._unavailable_req_nums
            NOTE:
                1. need lock here, as there may be multiple dp workers: one lock for one model.
                2. this version support model horizontal fusion, i.e., the reqs from different models may not be fused together.
        """


        # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 1")

        if self._unavailable_req_nums[to_model_id] == 0:
            # if all inp reqs have been fetched, directly return
            return
        

        # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 2, self.in_edge_dict_with_dummy_inp_nodes: {self.in_edge_dict_with_dummy_inp_nodes}")

        # we need first update self.fused_inp_queues
        inps = self.in_edge_dict_with_dummy_inp_nodes[to_model_id]

        # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 2.1, inps: {inps}")

        if (len(inps) == 1) or (self._independent_srcs[to_model_id]):
            # first get a lock

            # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 3")

            reqs = list()
            for from_model_id in inps:
                with self._get_correct_lock(from_model_id, self._add_req_locks, read_only=True):
                    reqs.extend( 
                        self._get_selected_outputs(from_model_id=from_model_id, to_model_id=to_model_id) )
                    
            # sort the reqs by their req ids
            reqs = sorted(reqs, key=lambda x: x[0])
            self.fused_inp_queues[to_model_id].extend(reqs)

            # print(f"self.fused_inp_queues[to_model_id]: {to_model_id}: {self.fused_inp_queues[to_model_id]}")

            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)
        else:
            # there are more than one inp src for the to_model
            new_complete_req_ids = list()
            inp_src_num = len(inps)
            for inp_i, from_model_id in enumerate(inps):

                # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 4, inp_i, from_model_id: {inp_i, from_model_id}")
                
                # first get a lock
                with self._get_correct_lock(from_model_id, self._add_req_locks, read_only=True):

                    # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 5, inp_i, from_model_id: {inp_i, from_model_id}")

                    reqs = self._get_selected_outputs(from_model_id=from_model_id, to_model_id=to_model_id)

                    # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 6, inp_i, from_model_id: {inp_i, from_model_id}")

                    for req in reqs:
                        if req[0] not in self._available_srcs[to_model_id]:
                            self._available_srcs[to_model_id][req[0]] = [0, [None for _ in range(inp_src_num)]]
                        self._available_srcs[to_model_id][req[0]][0] += 1
                        self._available_srcs[to_model_id][req[0]][1][inp_i] = req[1]
                        if self._available_srcs[to_model_id][req[0]][0] == inp_src_num:
                            new_complete_req_ids.append(req[0])
            
            # print(f"_get_fused_inp_queues, to_model_id: {to_model_id}, 7, inp_i, from_model_id: {inp_i, from_model_id}  self._available_srcs[to_model_id]: {self._available_srcs[to_model_id]}")
            
            new_complete_req_ids = sorted(new_complete_req_ids)
            reqs = [(req_id, self.fuse_inp_srcs(self._available_srcs[to_model_id][req_id][1])) for req_id in new_complete_req_ids]
            self.fused_inp_queues[to_model_id].extend(reqs)

            # print(f"self.fused_inp_queues[to_model_id]: {to_model_id}: {self.fused_inp_queues[to_model_id]}")

            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)









    def _get_seqs(self, to_model_id: int, dp_id: int, dp_size: int
        ) -> Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]:
        """
            Get the available inp reqs for a specific dp worker.
            Update:
                self.fetched_fused_inp_start
        """

        # print(f"int _get_seqs: self.fetched_fused_inp_start: {self.fetched_fused_inp_start}")


        start = self.fetched_fused_inp_start[to_model_id][dp_id]
        reqs = self.fused_inp_queues[to_model_id]
        ret = reqs[start::dp_size]
        end = start + len(ret)*dp_size
        self.fetched_fused_inp_start[to_model_id][dp_id] = end

        return ret

        # start = self.fetched_fused_inp_start[(from_model_id, to_model_id)]
        # reqs = self.output_pool[from_model_id]
        # end = len(reqs)
        # ret = reqs[start:end]
        # self.fetched_fused_inp_start[(from_model_id, to_model_id)] = end
        # return ret



    def _update_out_req_model_id_mapping(
            self, fused_model_id: int, base_model_id: int,
            reqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]], 
    ) -> None:
        
        if fused_model_id not in self.base_model_ids_dict:
            # this is not a fused model
            return
        
        new_mapping = {req[0]:base_model_id for req in reqs}
        self.out_req_model_id_mapping[fused_model_id].update(new_mapping)



    def get_seqs_base_model(self, fused_model_id: int, to_model_id: int, dp_id: int, dp_size: int
        ) -> Tuple[Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]], bool]:
        """
            Get the available inp reqs for a specific dp worker.
            Update:
                self.fetched_fused_inp_start, self.fused_inp_queues
            Output:
                1. ret: the reqs assigned to the dp worker
                2. possible_to_get_future_reqs: whether it is possible for the dp worker to get more new available reqs in the future
                    True if there are/will be remaining available inp reqs for this model.
            NOTE:
                1. When the dp worker cannot get available reqs, reorganize the fused_inp_queue and assign reqs to the worker again.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
                3. ``to_model_id`` is the id of a base model.
            NOTE:
                1. this function is for a base model.
        """    

        # print(f"in get_seqs_base_model: fused_model_id: {fused_model_id}, to_model_id: {to_model_id}")

        with self._get_correct_lock(to_model_id, self._fetch_req_locks, read_only=False):

            # print(f"_get_correct_lock successfully!")

            # first update fused inp queues
            self._get_fused_inp_queues(to_model_id)

            # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 1")

            ret = self._get_seqs(to_model_id, dp_id, dp_size)

            # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 2, ret: {ret}")

            possible_to_get_future_reqs: bool = True

            if len(ret) == 0:
                # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 3.1")

                # reorganize the fused_inp_queue
                self.reset_state_for_model(to_model_id, dp_size)

                # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 3")

                # check whether there is available request for this dp worker
                start = self.fetched_fused_inp_start[to_model_id][dp_id]

                # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 4, start: {start}")

                if start >= len(self.fused_inp_queues[to_model_id]):
                    # there is no req for this dp worker, directly pop the first req to this worker
                    # we do not need to change self.fetched_fused_inp_start in this case
                    if len(self.fused_inp_queues[to_model_id])>0:
                        ret = [self.fused_inp_queues[to_model_id][0]]
                        self.fused_inp_queues[to_model_id] = self.fused_inp_queues[to_model_id][1:]
                        
                        self._update_out_req_model_id_mapping(fused_model_id, base_model_id=to_model_id, reqs=ret)

                        # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 5.1, ret: {ret}")

                        return ret, possible_to_get_future_reqs
                    else:
                        if self._unavailable_req_nums[to_model_id] == 0:
                            # only when there is no unavailable reqs and all the available reqs have been assigned to dp workers
                            # we set possible_to_get_future_reqs to False
                            possible_to_get_future_reqs = False

                        assert self._unavailable_req_nums[to_model_id] >= 0, f"assert self._unavailable_req_nums[to_model_id] >= 0 wrong: {self._unavailable_req_nums[to_model_id]}"

                        # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 5.2, self._unavailable_req_nums: {self._unavailable_req_nums}")

                        return [], possible_to_get_future_reqs
                else:
                    # run the normal get_seq process
                    ret = self._get_seqs(to_model_id, dp_id, dp_size)

                    # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 6, ret: {ret}")

                    self._update_out_req_model_id_mapping(fused_model_id, base_model_id=to_model_id, reqs=ret)
                    return ret, possible_to_get_future_reqs

            else:

                # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 5.1")

                self._update_out_req_model_id_mapping(fused_model_id, base_model_id=to_model_id, reqs=ret)

                # print(f"fused_model_id: {fused_model_id}, to_model_id: {to_model_id}, 5")

                return ret, possible_to_get_future_reqs






    def get_seqs(self, to_model_id: int, dp_id: int, dp_size: int
        ) -> Tuple[Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]], bool, List[int]]:
        """
            Get the available inp reqs for a specific dp worker.
            Update:
                self.fetched_fused_inp_start, self.fused_inp_queues
            Output:
                1. ret: the reqs assigned to the dp worker
                2. possible_to_get_future_reqs: whether it is possible for the dp worker to get more new available reqs in the future
                    True if there are/will be remaining available inp reqs for this model.
                3. req_base_model_ids: the corresponding base model each req belongs to.
            NOTE:
                1. When the dp worker cannot get available reqs, reorganize the fused_inp_queue and assign reqs to the worker again.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
                3. ``to_model_id`` is the id of a base model.
            NOTE:
                1. this function is for any model: fused or base models.
        """ 
        import traceback
        try:

            # print(f"self.base_model_ids_dict: {self.base_model_ids_dict}")

            base_model_ids = list() 
            if to_model_id not in self.base_model_ids_dict:
                base_model_ids = [to_model_id]
            else:
                base_model_ids = self.base_model_ids_dict[to_model_id]


            # print(f"base_model_ids: {base_model_ids}  to_model_id: {to_model_id}  dp_id: {dp_id}")

            possible_to_get_future_reqs = False
            ret = list()
            req_base_model_ids = list()
            for base_model_id in base_model_ids:
                tmp_ret, tmp_possible_to_get_future_reqs = \
                    self.get_seqs_base_model(fused_model_id=to_model_id, to_model_id=base_model_id, dp_id=dp_id, dp_size=dp_size)                
                ret.extend(tmp_ret)
                # print(f"base_model_id: {base_model_id}: {tmp_possible_to_get_future_reqs, tmp_ret}")
                if tmp_possible_to_get_future_reqs:
                    possible_to_get_future_reqs = True
                
                # record the base model id of each req to return
                if len(base_model_ids) > 1:
                    # if this is a fused model
                    req_base_model_ids.extend([base_model_id]*len(tmp_ret))
            
            if len(base_model_ids) == 1:
                req_base_model_ids = [base_model_ids[0]]

            return ret, possible_to_get_future_reqs, req_base_model_ids
        
        except Exception as e:
            print(f"Exception in llm._run_engine: {e}")
            print(traceback.format_exc())
            assert False




    

    def get_info(self):
        return self.output_pool, self._read_output_nums, self._available_srcs, \
            self.fused_inp_queues, self.fetched_fused_inp_start, self.in_edge_dict_with_dummy_inp_nodes
    




class MyManager(BaseManager):
    pass

MyManager.register('Communicator', LLM_COMMUNICATOR)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





# class SCHEDULE_COMMUNICATOR():
#     def __init__(self, model_num: int, in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]]):
#         self.shared_setting: Array = None
#         # should be 2+#model events, the first two events set by main process (stop all models, start all models), 
#         # the other events set by the model processes: indicating whether the preparation for reschedule is finished.
#         events: List[Event] = None
#         started_status: List[Event] = None
#         shared_finish_status = [False] # store whether each model is finished

#         # to support multi-level LLM system -> used to communicate between LLMs
#         # set below 3 variables when initialized
#         # check_out_gap: int = int(1e9)
#         check_out_gaps: Array = None

#     SHARED_CONTECT.set_execution_plan_size(tot_gpu_num)
#     counter = Array('i', [0 for i in range(len(model_paths)*SHARED_CONTECT.execution_plan_size)]) # 'd' is for double
#     # all child processors will inherit this event
#     SHARED_CONTECT.events = [Event() for _ in range(2+len(model_paths))]
#     # set the event to allow models to run
#     # SHARED_CONTECT.events[1].set()
#     SHARED_CONTECT.started_status = [Event() for _ in range(len(model_paths))]
#     SHARED_CONTECT.shared_setting = counter
#     SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(len(model_paths))])
#     # add check_out_gaps
#     check_out_gaps = Array('i', [int(1e9)]*len(model_paths)) # 'd' is for double
#     SHARED_CONTECT.check_out_gaps = check_out_gaps
#     SHARED_CONTECT.check_in_gap = check_gap



# MyManager.register('Schedule_Communicator', SCHEDULE_COMMUNICATOR)




class SHARED_CONTECT():
    shared_setting: Array = None
    shared_id: int = 0
    # to support data parallelism
    dp_id: int = 0
    # should be 2+#model events, the first two events set by main process (stop all models, start all models), 
    # the other events set by the model processes: indicating whether the preparation for reschedule is finished.
    events: List[Event] = None
    started_status: List[Event] = None
    gened_outputs = deque()
    remaining_requests = deque()
    shared_finish_status = [False] # store whether each model is finished
    # TODO (jingzhi) need to set this variable automatically
    # execution_plan_size: int = 7
    execution_plan_size: int = None # this value will be set via self.set_execution_plan_size
    # execution_plan_non_gpu_para_num: int = 3 # 3 for tp_size, gpu_mem, wldegree
    execution_plan_non_gpu_para_num: int = 4 # 4 for tp_size, gpu_mem, wldegree, dp_size

    # to support data parallelism, use a message passer to stop ray dp actors if necessary
    # message_passer_for_dp = None


    # to support multi-level LLM system -> used to communicate between LLMs
    communicator: LLM_COMMUNICATOR = None
    # set below 3 variables when initialized
    return_str: bool = False
    # check_in: bool = False
    check_in_gap: int = int(1e9)
    # check_out_gap: int = int(1e9)
    check_out_gaps: Array = None
    tot_req_num_remained: int = None

    # to support different sampling args for different base models
    # will be set when initialized and not changed during the scheduling.
    # {base model id: (ignore_eos, inp_len, out_len)}
    sampling_args_dict: Dict[int, SamplingParams] = None


    # @classmethod
    # def should_reschedule(cls) -> bool:
    #     # check whether this model should stop inference and rescheduled
    #     # TODO (jingzhi) complete the schedule logic here.
    #     # 暂时先这么写，先不管reschedule怎么实现吧，因为我们需要知道是否需要允许最灵活的reschedule。
    #     if os.environ['RUN_MULTI_MODEL'] != 'True':
    #         return False
    #     return cls.events[0].is_set()


    # -------------------------------------------------------------------
    @classmethod
    def set_execution_plan_size(cls, tot_gpu_num):
        # 3 for tp_size, gpu_mem, wldegree
        cls.execution_plan_size = tot_gpu_num + cls.execution_plan_non_gpu_para_num


    # @classmethod
    # def init_SHARED_CONTECT_by_copy(cls, dp_id: int, to_copy):
    #     cls.shared_setting, cls.shared_id, cls.events, cls.started_status, cls.shared_finish_status, cls.execution_plan_size=\
    #         to_copy
        
    #     # to support data parallelism
    #     cls.dp_id = dp_id
        
    # @classmethod
    # def params_in_SHARED_CONTECT_to_copy(cls):
    #     return (cls.shared_setting, cls.shared_id, cls.events, \
    #         cls.started_status, cls.shared_finish_status, cls.execution_plan_size)
      
    

    @classmethod
    def get_tp_size(cls):
        # 3 for tp_size, gpu_mem, wldegree
        offset = cls.shared_id * cls.execution_plan_size
        tensor_parallel_size: int = cls.shared_setting[offset]
        return tensor_parallel_size

    @classmethod
    def get_gpu_mem_utilization(cls):
        # 3 for tp_size, gpu_mem, wldegree
        offset = cls.shared_id * cls.execution_plan_size
        gpu_memory_utilization: float = cls.shared_setting[offset+1]/10
        return gpu_memory_utilization      

    @classmethod
    def get_wldegree(cls):
        # 3 for tp_size, gpu_mem, wldegree
        offset = cls.shared_id * cls.execution_plan_size
        weight_load_degree: int = cls.shared_setting[offset+2]
        return weight_load_degree  
    
    @classmethod
    def get_dp_size(cls):
        # 4 for tp_size, gpu_mem, wldegree, dp_size
        offset = cls.shared_id * cls.execution_plan_size
        dp_size: int = cls.shared_setting[offset+3]
        return dp_size  


    @classmethod
    def has_dp_parallel(cls):
        # 4 for tp_size, gpu_mem, wldegree, dp_size
        if cls.dp_id > 0:
            return True
        offset = cls.shared_id * cls.execution_plan_size
        dp_size: int = cls.shared_setting[offset+3]
        return dp_size > 1


    @classmethod
    def get_gpus(cls):
        # 3 for tp_size, gpu_mem, wldegree
        offset = cls.shared_id * cls.execution_plan_size
        gpus = cls.shared_setting[offset+cls.execution_plan_non_gpu_para_num:offset+cls.execution_plan_size]
        return gpus
        # # deal with data parallelism
        # tp_size = cls.get_tp_size()
        # if cls.has_dp_parallel():
        #     gpus = gpus[cls.dp_id*tp_size:] + gpus[:cls.dp_id*tp_size]
        # return gpus


    @classmethod
    def get_sampling_args(cls, base_model_id: int)->SamplingParams:
        return cls.sampling_args_dict[base_model_id]




    # -------------------------------------------------------------------


    @classmethod
    def prepare_for_reschedule(cls, outputs, remaining_requests, llm_engine, mark_finish:bool=True) -> None:
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
        exit_actor = (os.environ['SOFT_RESCHEDULE'] == 'False')
        llm_engine.delete_workers(exit_actor)

        print(f"event list status in prepare--model {cls.shared_id} 3", flush=True)

        # we delete the placement group when we are doing hard reschedule
        if (os.environ['SOFT_RESCHEDULE'] == 'False') and (llm_engine.parallel_config.world_size > 1):
            # do not need to remove placement group if there is only one worker
            print(f"dp worker id {cls.dp_id}, placement_group_table(): {placement_group_table()}")

            # pg_info = list(placement_group_table().values())[0]
            # pg = get_placement_group(pg_info['name'])
            # remove_placement_group(pg)

            pg_info_name = 'my_pg'+os.environ['DP_WORKER_I']
            pg = get_placement_group(pg_info_name)
            remove_placement_group(pg)


        print(f"event list status in prepare--model {cls.shared_id} 4", flush=True)

        # update global variable
        KVBlkPerLayerWeight.reset()

        print(f"event list status in prepare--model {cls.shared_id} 5", flush=True)

        # # set the event to notify the preparation is finished
        # mark_finish == False when there is data parallel, and the finish status will be marked after all dp actors finish
        if mark_finish:
            cls.events[cls.shared_id + 2].set()

        if cls.events != None:
            # non-master data parallel ray actors do not print the message below
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

        # offset = cls.shared_id * cls.execution_plan_size

        # for v in cls.shared_setting:
        #     print(v)

        # tensor_parallel_size: int = cls.shared_setting[offset]
        # gpu_memory_utilization: float = cls.shared_setting[offset+1]/10
        # weight_load_degree: int = cls.shared_setting[offset+2]
        tensor_parallel_size: int = cls.get_tp_size()
        gpu_memory_utilization: float = cls.get_gpu_mem_utilization()
        weight_load_degree: int = cls.get_wldegree()
        # gpus: str = ','.join([str(i) for i in cls.shared_setting[offset+3:offset+7]])
        # gpus: str = ','.join([str(i) for i in cls.shared_setting[offset+3:offset+cls.execution_plan_size]])
        gpus: str = ','.join([str(i) for i in cls.get_gpus()])

        os.environ['WEIGHT_LOAD_DEGREE'] = str(weight_load_degree)

        # TODO (jingzhi) this may not work because we have initialized torch.cuda when we reload the model
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        # NOTE: to enable reschedule, we use TOT_ORDERED_GPUS to store the real available gpus in order
        os.environ['TOT_ORDERED_GPUS'] = gpus

        return tensor_parallel_size, gpu_memory_utilization

    
    @classmethod
    def set_execution_plan(cls, setting, shared_ids=list()) -> None:
        '''
            If model_ids is given, only change the setting for specific models
        '''
        iter_range = None
        if len(shared_ids) == 0:
            iter_range = range(len(cls.shared_setting))
        else:
            iter_range = list()
            for shared_id in shared_ids:
                iter_range.extend(range(cls.execution_plan_size * shared_id, cls.execution_plan_size * (shared_id+1)))
        
        # print(f"setting: {setting}, len(setting): {len(setting)}  len(shared_setting): {len(cls.shared_setting)} iter_range:{iter_range}")
        
        for i, v in zip(iter_range, setting):
            cls.shared_setting[i] = v



    # @classmethod
    # def get_comp_gpus(cls, model_id) -> List[int]:
    #     '''
    #         If model_ids is given, only change the setting for specific models
    #         setting format: tensor_parallel_size, gpu_memory_utilization*10, weight_load_degree, gpus
    #     '''
    #     setting = cls.shared_setting[ cls.execution_plan_size * model_id : cls.execution_plan_size * (model_id+1) ]
    #     tp_size = setting[0]
    #     gpus = setting[3:]
    #     return gpus[:tp_size]




    # @classmethod
    # def sync_before_loading_model(cls) -> None:
    #     if os.environ['RUN_MULTI_MODEL'] != 'True':
    #         return
        
    #     print(f"model_i: {SHARED_CONTECT.shared_id}, os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}, SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)

    #     # # set the event to notify the preparation is finished
    #     # cls.events[cls.shared_id + 2].set()
    #     # wait for the signal to start a new execution plan
    #     print(f"event list status in sync before loading--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")
    #     cls.events[1].wait()
    




    @classmethod
    def is_finished(cls) -> None:
        return cls.shared_finish_status[cls.shared_id]
    

    @classmethod
    def set_finished(cls, gened_output_num: int, set_event_state_anyway: bool = False) -> None:
        """
            set_event_state_anyway is True only when dp parallel is used, in which case, we will call set_finish 
            after all dp workers finish the preparation for re-scheduling (in the preparation, we will not set the event).
            --> so set_event_state_anyway is used to avoid set the event twice.
            但是其实这里也可以改掉的，以为dp worker我们目前不管有没有算完都会kill掉。不行，还是得等所有dp worker都至少停下来了才可以设置event？
        """
        print(f"in set_finished, model id: {cls.shared_id}, cls.tot_req_num_remained: {cls.tot_req_num_remained}, gened_output_num: {gened_output_num}, set_event_state_anyway: {set_event_state_anyway}")


        # support multi-level model system
        # first update tot_req_num_remained
        cls.tot_req_num_remained = cls.tot_req_num_remained - gened_output_num
        finished = (cls.tot_req_num_remained == 0)
        if finished:
            cls.shared_finish_status[cls.shared_id] = finished
        if finished or set_event_state_anyway:
            if os.environ['RUN_MULTI_MODEL'] == 'True':
                cls.events[cls.shared_id + 2].set()

        if os.environ['RUN_MULTI_MODEL'] == 'True':
            print(f"event list status in set finish--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")

        # if finished:
        #     cls.shared_finish_status[cls.shared_id] = finished
        #     if os.environ['RUN_MULTI_MODEL'] == 'True':
        #         cls.events[cls.shared_id + 2].set()

        # if os.environ['RUN_MULTI_MODEL'] == 'True':
        #     print(f"event list status in set finish--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")



    @classmethod
    def set_finish_preparation_before_init_LLM(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] == 'True':
            cls.events[cls.shared_id + 2].set()
            print(f"event list status in set finish before init LLM--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")



    # @classmethod
    # def set_finish_preparation_for_reschedule(cls) -> None:
    #     if os.environ['RUN_MULTI_MODEL'] == 'True':
    #         cls.events[cls.shared_id + 2].set()
    #         print(f"event list status in set finish prepare for reschedule--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")





    @classmethod
    def wait_all_models_to_finish_prepare_for_reschedule(cls, shared_ids: List[int]) -> None:
        assert os.environ['RUN_MULTI_MODEL'] == 'True'
        # wait for all the models to finish

        print(f"event list status in waiting all to finish 1: {[event.is_set() for event in cls.events[2:]]}")

        # for event in cls.events[2:]:
        #     event.wait()
        for shared_id in shared_ids:
            cls.events[2+shared_id].wait()

        print(f"event list status in waiting all to finish 2: {[event.is_set() for event in cls.events[2:]]}")



    @classmethod
    def wait_all_models_to_finish_preparation_before_init_LLM(cls, shared_ids: List[int]) -> None:
        '''
            Call this function in the multi-model scheduler process.
        '''
        assert os.environ['RUN_MULTI_MODEL'] == 'True'
        # wait for all the models to finish

        print(f"event list status in waiting all to finish 1: {[event.is_set() for event in cls.events[2:]]}")

        # for event in cls.events[2:]:
        #     event.wait()
        for shared_id in shared_ids:
            cls.events[2+shared_id].wait()

        print(f"event list status in waiting all to finish 2: {[event.is_set() for event in cls.events[2:]]}")


    # @classmethod
    # def restart_all_models(cls) -> None:
    #     assert os.environ['RUN_MULTI_MODEL'] == 'True'
    #     # clear the stop signal
    #     cls.events[0].clear()
    #     # clear all model events
    #     for event, finished in zip(cls.events[2:], cls.shared_finish_status):
    #         if not finished:
    #             event.clear()
    #     # set the start signal
    #     cls.events[1].set()

    #     print(f"event list status in restart--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")



    @classmethod
    def start_specific_models(cls, shared_ids) -> None:
        '''
            (1) set the started_status;
            (2) clear the finishing_preparation_for_reschedule event;
        '''

        assert os.environ['RUN_MULTI_MODEL'] == 'True'
        # if model event is set, this model will not be started
        for shared_id in shared_ids:
            cls.started_status[shared_id].set()
            cls.events[2+shared_id].clear()

        print(f"event list status in start_specific_models--model {cls.shared_id}: {[event.is_set() for event in cls.started_status]}")



    @classmethod
    def wait_to_be_started(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return
        cls.started_status[cls.shared_id].wait()

        print(f"event list status in start_specific_models--model {cls.shared_id}: {[event.is_set() for event in cls.started_status]}")



    @classmethod
    def query_finish_status(cls, shared_id) -> None:
        assert os.environ['RUN_MULTI_MODEL'] == 'True'
        # print(f"query_finish_status: shared_id: {shared_id} ")
        return cls.shared_finish_status[shared_id]



    # deal with data parallelism
    @classmethod
    def should_reschedule(cls) -> bool:
        '''
        check whether this model should stop inference and rescheduled
        Call this function after every inference iteration.
        '''
        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return False
        # NOTE: change to multiprocessing subprocesses for dp workers, so we do not need such message_passer
        # if cls.dp_id > 0:
        #     return ray.get(cls.message_passer_for_dp.query_stop.remote())
        return not cls.started_status[cls.shared_id].is_set()


    # deal with data parallelism
    # NOTE: change to multiprocessing subprocesses for dp workers, so we do not need such notify function
    # @classmethod
    # def notify_dp_actors_should_reschedule(cls) -> None:
    #     '''
    #     Notify other non-master dp ray actors that they should stop inference
    #     '''
    #     if os.environ['RUN_MULTI_MODEL'] != 'True':
    #         return
    #     if cls.has_dp_parallel() and (cls.dp_id == 0):
    #         cls.message_passer_for_dp.set_stop.remote()




    @classmethod
    def stop_specific_models(cls, shared_ids: List[int]) -> bool:
        '''
        Call this function in the model scheduler process.
        '''
        for shared_id in shared_ids:
            cls.started_status[shared_id].clear()



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



