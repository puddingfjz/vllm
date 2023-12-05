import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)




from vllm.core import my_swap_and_recompute


# <jingzhi>
import numpy as np
import copy



logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)





class MySchedulerConfig(object):
    """docstring for MySchedulerConfig"""
    def __init__(self, block_size):
        super(MySchedulerConfig, self).__init__()
        import os
        self.use_our_method = (os.environ['USE_OUR_METHOD']=='True') # False for vLLM

        # <jingzhi> DEBUG
        print(f"self.use_our_method: {self.use_our_method}")

        self.complete_swap_list = list() # the seq_groups to be swapped out for this iter group
        self.complete_swap_in_list = list() # the seq_groups to be swpped in for this iter group
        # the seq_groups to be recomputed or to be released for recomputation for this iter group
        self.complete_recompute_list = list()

        self.swap_list = list()
        self.swap_in_list = list()
        self.recompute_list = list()

        self.ready_to_run = list() # the seq_groups on card but have not been added to the running list

        self.blocks_to_free = list()
        self.new_blocks = list()
        self.current_effective_iter = 0 # the effective iteration number
        self.block_size = block_size

        self.swap_num_each_iter = None
        self.seq_group_dict = dict() # update every iteration group when the running request list is fixed
        


        self.blk_mem_size = None

        self.recompute_blk_num = 0

        self.DP_time = 0

        self.this_iter_recompute_dict = dict()



    def delete_finished_released_requests(self, request_ids):
        for request_id in request_ids:
            del self.seq_group_dict[request_id]

    def _update_current_iter(self):
        self.current_effective_iter += 1

    def _revoke_current_iter_update(self):
        self.current_effective_iter -= 1

    def _is_first_iter_in_group(self):
        return self.current_effective_iter % self.block_size == 1

    def _is_last_iter_in_group(self):
        return (self.current_effective_iter) % self.block_size == 0


    def no_interrupted_requests(self):
        return my_swap_and_recompute.mem_scheduler.no_interrupted_requests()


    def reset_status(self):
        self.complete_swap_list = list() # the seq_groups to be swapped out for this iter group
        self.complete_swap_in_list = list() # the seq_groups to be swpped in for this iter group
        # the seq_groups to be recomputed or to be released for recomputation for this iter group
        self.complete_recompute_list = list()

        self.swap_list = list()
        self.swap_in_list = list()
        self.recompute_list = list()

        self.blocks_to_free = list()
        self.new_blocks = list()

        self.swap_num_each_iter = None



class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []


        # <jingzhi>@new swap & recompute mechanism
        self.my_scheduler_config = MySchedulerConfig(self.block_manager.block_size)

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

                # <jingzhi> DEBUG
                if seq_group.request_id == '157':
                    print(f"load 157: {seq_group.request_id, seq_group.get_seqs()[0].get_len()}")


            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)

                    # <jingzhi> DEBUG
                    self.my_scheduler_config.recompute_blk_num = self.my_scheduler_config.recompute_blk_num + \
                        self.block_manager.get_gpu_blk_num(seq_group=victim_seq_group) * 2

                    print(f"to recompute 1: {victim_seq_group.request_id, victim_seq_group.get_seqs()[0].get_len()}")

                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # <jingzhi> DEBUG
                    self.my_scheduler_config.recompute_blk_num = self.my_scheduler_config.recompute_blk_num + \
                        self.block_manager.get_gpu_blk_num(seq_group=seq_group) * 2

                    print(f"to recompute 2: {seq_group.request_id, seq_group.get_seqs()[0].get_len()}")

                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)

            while self.swapped:
                seq_group = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

                # <jingzhi> For DEBUG
                print(f"load: {seq_group.request_id, seq_group.get_seqs()[0].get_len()}")


        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs
















    # <jingzhi> consider the output length of requests when managing the KV cache
    def _schedule_outlen_aware(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        old_waiting: List[SequenceGroup] = self.waiting
        self.waiting = []


        self.running = sorted(self.running, key=lambda seq_group: seq_group.sampling_params.max_tokens)
        curr_blk_num = 0
        for seq_group in self.running:
            requred_blk_num = (seq_group.get_seqs()[0].get_len() + self.block_manager.block_size - 1) // self.block_manager.block_size
            if requred_blk_num <= (self.block_manager.num_total_gpu_blocks - curr_blk_num):
                # can stay on card
                # self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
                curr_blk_num = curr_blk_num + requred_blk_num
            else:
                # need swap out
                # <jingzhi> DEBUG
                self.my_scheduler_config.recompute_blk_num = self.my_scheduler_config.recompute_blk_num + \
                    self.block_manager.get_gpu_blk_num(seq_group=seq_group) * 2

                self._preempt(seq_group, blocks_to_swap_out)
                preempted.append(seq_group)
        self.running = running
        for seq_group in running:
            self._append_slot(seq_group, blocks_to_copy)
        self.waiting = sorted(self.waiting, key=lambda seq_group: seq_group.sampling_params.max_tokens) + old_waiting



        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)

            while self.swapped:
                seq_group = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs
















    # <jingzhi> consider the future peak block demand of requests when managing the KV cache (selecting requests to release)
    # we will also do paged swapping and recomputation here
    def _schedule_peak_demand_aware_paged(self) -> SchedulerOutputs:

        def get_blk_num(seqlen):
            block_size = self.block_manager.block_size
            return (seqlen+block_size-1)//block_size

        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                # <jingzhi> For Profiling: temporarily remove this condition checking
                # if (num_batched_tokens >
                #         self.scheduler_config.max_num_batched_tokens * 15):
                #     break
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []


        # select the subset of requests to keep on GPU
        self.running = sorted(self.running, key=lambda seq_group: seq_group.get_seqs()[0].get_len()) # sort by increasing in_lens
        # curr_on_card： [(tot_in, future_out, idx)], tot_in=prompt+generated_out
        curr_on_card = [(seq_group.get_seqs()[0].get_len(), \
            seq_group.sampling_params.max_tokens + len(seq_group.prompt) - seq_group.get_seqs()[0].get_len(), seq_group_i) \
            for seq_group_i, seq_group in enumerate(self.running)]

        demand = sum(get_blk_num(np.asarray([info[0] for info in curr_on_card])))
        # print(f"demand: {demand}, tot_blk_num: {self.block_manager.num_total_gpu_blocks}")
        if demand > self.block_manager.num_total_gpu_blocks:
            # we cannot keep on self.running on GPU, and need to release some requests

            best_solution = [[], [], float('inf')]
            time1 = time.perf_counter()
            DP_select_requests_to_release(curr_on_card, curr_on_card, self.block_manager.num_total_gpu_blocks, [], best = best_solution)
            time2 = time.perf_counter()
            self.my_scheduler_config.DP_time+=(time2-time1)
            # print(f"best_solution: {best_solution}")

            # compute the number of blocks to release in this iteration
            to_release_this_iter = sum([get_blk_num(info[0]) for info in best_solution[1]]) - \
                sum([get_blk_num(info[0] - 1) for info in best_solution[1]])
            
            # assert (self.block_manager.gpu_allocator.get_num_free_blocks() == (self.block_manager.num_total_gpu_blocks - my_swap_and_recompute.cache_state.tot_blk_num_to_release - sum([get_blk_num(info[0] - 1) for info in best_solution[0]+best_solution[1]]))), (self.block_manager.gpu_allocator.get_num_free_blocks(), self.block_manager.num_total_gpu_blocks, my_swap_and_recompute.cache_state.tot_blk_num_to_release, sum([get_blk_num(info[0] - 1) for info in best_solution[0]+best_solution[1]]))
            to_release_this_iter = max(0, to_release_this_iter - self.block_manager.gpu_allocator.get_num_free_blocks())
            # 必定得把之前剩余的block全都release掉   为啥这里一定会把剩余的block全部消耗掉？不一定？实验里面出现了这个assertion为否的情况。理论上确实如此？
            # 因为否则的话，我们可以保留更多的request把blk_num_to_release 消耗完
            # assert (to_release_this_iter >= my_swap_and_recompute.cache_state.tot_blk_num_to_release), f"best_solution: {best_solution}, blk_num_to_release: {my_swap_and_recompute.cache_state.tot_blk_num_to_release}, to_release_this_iter: {to_release_this_iter}"

            print(f"to_release_this_iter: {to_release_this_iter}, free_blk_num: {self.block_manager.gpu_allocator.get_num_free_blocks()}")

            # update the KV cache state
            time1 = time.perf_counter()
            my_swap_and_recompute.update_KV_state_peak_demand_aware_paged(
                [(self.running[info[2]].request_id, get_blk_num(info[0]-1)) for info in best_solution[0]])

            to_swap, to_recompute, _ = my_swap_and_recompute.get_blocks_to_release(to_release_this_iter)
            time2 = time.perf_counter()
            self.my_scheduler_config.DP_time+=(time2-time1)
            self.my_scheduler_config.recompute_blk_num = self.my_scheduler_config.recompute_blk_num + to_release_this_iter

            # do preemption
            preempted = [self.running[info[2]] for info in best_solution[0]]
            self.swapped = self.swapped + preempted # ordered list
            self.my_scheduler_config.seq_group_dict.update({seq_group.request_id:seq_group for seq_group in preempted})
            self._my_do_swap_out_free_and_update_blocktables(to_swap, to_recompute, blocks_to_swap_out)
            
            # print(f"free_blk_num after swapping_out: {self.block_manager.gpu_allocator.get_num_free_blocks()}")

            for seq_group in preempted:        
                for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                    seq.status = SequenceStatus.SWAPPED



            running = [self.running[info[2]] for info in best_solution[1]]
            self.running = running
        
        


        # Swap in the sequence groups in the SWAPPED state if possible.
        # self.swapped = self.policy.sort_by_priority(now, self.swapped) 
        # we select requests from self.swapped from newly inserted ones to older ones
        if not preempted:

            # deal with the page-swap-out case
            if my_swap_and_recompute.cache_state.tot_blk_num_to_release + demand > self.block_manager.num_total_gpu_blocks:
                # print("2.1.0")
                # 需要接着挪出一部分之前没挪完的block，但是没有多余空间reload完整的request
                to_release_this_iter = sum(get_blk_num(np.asarray([info[0] for info in curr_on_card]))) - \
                    sum(get_blk_num(np.asarray([info[0] - 1 for info in curr_on_card])))
                
                # assert (self.block_manager.gpu_allocator.get_num_free_blocks() == (self.block_manager.num_total_gpu_blocks - sum(get_blk_num(np.asarray([info[0] - 1 for info in curr_on_card]))) - my_swap_and_recompute.cache_state.tot_blk_num_to_release))
                to_release_this_iter = to_release_this_iter - self.block_manager.gpu_allocator.get_num_free_blocks()
                # assert to_release_this_iter >= 0

                # do page-swapping-out
                time1 = time.perf_counter()
                to_swap, to_recompute, released_request_ids = my_swap_and_recompute.get_blocks_to_release(to_release_this_iter)
                time2 = time.perf_counter()
                self.my_scheduler_config.DP_time+=(time2-time1)

                self.my_scheduler_config.recompute_blk_num = self.my_scheduler_config.recompute_blk_num + to_release_this_iter
                self._my_do_swap_out_free_and_update_blocktables(to_swap, to_recompute, blocks_to_swap_out)
                # print(f"page-swap-in: {to_release_this_iter}, data_movement: {data_movement}")
                print(f"page-swap-out: {to_release_this_iter}")

            else:
                # 有多余的空间可以reload完整的request

                num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)


                curr_blk_num = demand
                if len(self.swapped) > 0:
                    # print("2.1.1")
                    # consider add in more requests  based on backtracking search?
                    # 但是不能直接用cost来当指标，因为我们希望cost最小，这样的话，就会自动选择到每个iteration只运行1个request。
                    # 但是目前并没有一个metric帮助选择request。先无脑选output length更短的吗？（但是这样就变成greedy了）之后再设计更好的算法。
                    # curr_released = sorted(curr_released, key=lambda info: info[1])
                    not_selected_ids = list()
                    reload_request_ids = list()
                    for i, seq_group in enumerate(self.swapped[::-1]):

                        # assert ((i == 0) or (len(set(my_swap_and_recompute.cache_state.to_release[seq_group.request_id])) == 1))

                        # The total number of sequences in the RUNNING state should not
                        # exceed the maximum number of sequences.
                        num_new_seqs = seq_group.get_max_num_running_seqs()
                        if (num_curr_seqs + num_new_seqs >
                                self.scheduler_config.max_num_seqs):
                            break

                        will_require = get_blk_num(seq_group.get_seqs()[0].get_len())
                        if will_require <= self.block_manager.num_total_gpu_blocks - curr_blk_num:
                            # print(f"blk num infor: {get_blk_num(info[0]), tot_blk_num, curr_blk_num}")
                            curr_blk_num = curr_blk_num + will_require
                            num_curr_seqs += num_new_seqs
                            reload_request_ids.append(seq_group.request_id)
                            self.running.append(seq_group)
                            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                                seq.status = SequenceStatus.RUNNING
                        else:
                            not_selected_ids.append(i)

                        if (i == 0) and (0 in not_selected_ids):
                            curr_blk_num = curr_blk_num + my_swap_and_recompute.cache_state.tot_blk_num_to_release


                    self.swapped = [self.swapped[-1-i] for i in not_selected_ids[::-1]] # keep the order of self.swapped

                    # do page-swap-in
                    time1 = time.perf_counter()
                    to_swap, to_recompute = my_swap_and_recompute.get_blocks_to_reload(reload_request_ids)
                    time2 = time.perf_counter()
                    self.my_scheduler_config.DP_time+=(time2-time1)

                    self._my_do_swap_in_free_and_update_blocktables(to_swap, to_recompute, blocks_to_swap_in)
                    # assert [seq_group.request_id for seq_group in self.swapped] == list(my_swap_and_recompute.cache_state.to_release.keys())

                    self.my_scheduler_config.recompute_blk_num = self.my_scheduler_config.recompute_blk_num + \
                        len(to_swap) + len(to_recompute)

                    print(f"reload_request_ids: {reload_request_ids}")


                    # store the recomputation information
                    self.my_scheduler_config.this_iter_recompute_dict = dict()
                    for info in to_recompute:
                        if info[0] not in self.my_scheduler_config.this_iter_recompute_dict:
                            self.my_scheduler_config.this_iter_recompute_dict = []
                        self.my_scheduler_config.this_iter_recompute_dict.append(info[1])


        # we can only append slot after releasing necessary blocks, append slot _for running requests
        # print(f"curren occupied: {sum(get_blk_num(np.asarray([seq_group.get_seqs()[0].get_len()-1 for seq_group in self.running])))},new_demand: {sum(get_blk_num(np.asarray([seq_group.get_seqs()[0].get_len() for seq_group in self.running])))}, blk_num_to_release: {my_swap_and_recompute.cache_state.tot_blk_num_to_release}, free_blk_num: {self.block_manager.gpu_allocator.get_num_free_blocks()}")
        for seq_group in self.running:
            self._append_slot(seq_group, blocks_to_copy)


        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs
















    # <jingzhi> : our method, page swap & page recompute
    # make decision (about swapping-in/out, recomputation) every block-size iterations
    # NOTE: for every request, we allocat enough blocks for it s.t. it can run block-size-iteration inference
    # blk_mem_size is the one cache block size, which can be obtained by get_cache_block_size() 
    def _my_schedule(self) -> SchedulerOutputs:
        # <jingzhi> update current_effective_iter
        self.my_scheduler_config._update_current_iter()

        # <jingzhi> DEBUG
        print('iter_num: ', self.my_scheduler_config.current_effective_iter)



        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        # if not self.swapped:
        # <jingzhi> try our page-swap and page-recompute idea
        # condition: (1) no swapped/released requests; 
        #            (2) it is the first iter in a group or the last iter is prompt
        if self.my_scheduler_config.no_interrupted_requests() and \
            self.my_scheduler_config._is_first_iter_in_group():
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                # if not self.block_manager.can_allocate(seq_group):
                #     break

                # <jingzhi>
                if not self.block_manager.my_can_allocate(seq_group, num_curr_seqs):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break


                # <jingzhi> question: why do we need this check? I think vllm do not need padding?
                # is it because we call torch algorithm with padding later?
                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                # self._allocate(seq_group)
                # <jingzhi> just allocate the exact required number of blocks
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )

                self.my_scheduler_config._revoke_current_iter_update()

                # <jingzhi> DEBUG
                print('iter_num after revoke: ', self.my_scheduler_config.current_effective_iter)


                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)



        # 我们需要按照上一个iteration group的结果更新self.running 和 block mapping 的信息
        # We do not maintain self.preempted

        if self.my_scheduler_config._is_first_iter_in_group():

            # free corresponding gpu blocks and update corresponding block tables
            # 但是这个操作可以之后再进行，因为可能新的一轮会撤销一些release的决定？好复杂，啊啊啊啊。暂时先不写这个撤销release决定的功能，太复杂了。之后再改进。
            # <jingzhi>@TODO: 可能需要撤销一些决定，不如有的gpu不需要再free了之类的。
            if self.my_scheduler_config.complete_swap_list:
                # deal with swapping out or directly releasing some gpu blocks

                running: List[SequenceGroup] = []
                preempted = [request_id for request_id, _ \
                    in self.my_scheduler_config.complete_swap_list]
                to_recompute = [request_id for request_id, _ \
                    in self.my_scheduler_config.complete_recompute_list]
                not_running = preempted+to_recompute

                # <jingzhi> DEBUG
                print(f"NOT RUNNING: {not_running}")


                finished_req_ids = list()

                for seq_group in self.running:
                    if seq_group.request_id not in not_running:
                        running.append(seq_group)
                    else:
                        # we should also delete the information for the requests which finish in last iter group
                        if seq_group.num_unfinished_seqs() == 0:
                            finished_req_ids.append(seq_group.request_id)

                        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                            seq.status = SequenceStatus.SWAPPED
                            # <jingzhi> DEBUG
                            print(seq_group.request_id, len(seq.logical_token_blocks), seq.get_len())

                self.running = running
                self.block_manager.my_update_block_status_swap_out(
                    self.my_scheduler_config.blocks_to_free, 
                    self.my_scheduler_config.new_blocks)

                # we should also update ready_to_run
                ready = list()
                for request_id in self.my_scheduler_config.ready_to_run:
                    if not my_swap_and_recompute.mem_scheduler.request_has_released_blks(request_id):
                        ready.append(request_id)
                self.my_scheduler_config.ready_to_run = ready


                # delete information of finished requests which are partially released
                self.my_scheduler_config.delete_finished_released_requests(finished_req_ids)
                my_swap_and_recompute.mem_scheduler.delete_finished_released_requests(finished_req_ids)

            else:
                # deal with swapping in or doing recomputation

                # <jingzhi> DEBUG
                if 139 in self.block_manager.block_tables:
                    print("self.block_manager.get_gpu_blk_num(seq_id='139')", self.block_manager.get_gpu_blk_num(seq_id=139))

                self.block_manager.my_update_block_status_swap_in(
                    self.my_scheduler_config.blocks_to_free, 
                    self.my_scheduler_config.new_blocks)

                # <jingzhi> DEBUG
                if 139 in self.block_manager.block_tables:
                    print("self.block_manager.get_gpu_blk_num(seq_id='139')",self.block_manager.get_gpu_blk_num(seq_id=139))


                # only append slot for those with all blocks on gpu
                cand_running = list(set([request_id for request_id, _ \
                    in self.my_scheduler_config.complete_swap_in_list\
                    + self.my_scheduler_config.complete_recompute_list]))

                # get quota to of self.running
                num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                    for seq_group in self.running)


                quota_to_run = self.scheduler_config.max_num_seqs - num_curr_seqs
                # we need to allocate one block for each running sequence
                quota_to_run = min(quota_to_run, 
                    self.block_manager.gpu_allocator.get_num_free_blocks()-num_curr_seqs-self.block_manager.watermark_blocks)

                run, ready, unready = my_swap_and_recompute.get_swapped_in_ready_to_run(
                    quota_to_run, self.my_scheduler_config.ready_to_run, cand_running)
                self.my_scheduler_config.ready_to_run = ready

                # <jingzhi> DEBUG
                print(f"Add to running from interrupted requests: {len(run), run}")


                for request_id in run:
                    seq_group = self.my_scheduler_config.seq_group_dict[request_id]
                    # self._append_slot(seq_group, blocks_to_copy)
                    self.running.append(seq_group)
                    for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                        seq.status = SequenceStatus.RUNNING
                    del self.my_scheduler_config.seq_group_dict[request_id]


                    # <jingzhi> DEBUG
                    if '139' == request_id:
                        assert 139 == seq_group.get_seqs()[0].seq_id, f"information: {type(seq_group.get_seqs()[0].seq_id), seq_group.get_seqs()[0].seq_id}"
                        print(self.block_manager.get_gpu_blk_num(seq_group=seq_group), self.block_manager.get_gpu_blk_num(seq_id=139))


                # delete running requests from on_card_info
                my_swap_and_recompute.mem_scheduler.delete_useless_on_card_info(run)


                # <jingzhi> DEBUG
                print(f"Updated on card infor after adding to running from interrupted requests: {my_swap_and_recompute.mem_scheduler.on_card_info}")

                # <jingzhi>@TODO: deal with the to-recompute blocks


            # <jingzhi> when finish dealing with the swap and recomputation of last iter group, 
            # reset the status of the current iter group
            self.my_scheduler_config.reset_status()


            # <jingzhi> DEBUG

            for seq_group in self.running:
                if seq_group.request_id == '357':
                    print("request 357 infor: ", len(seq_group.prompt_token_ids), seq_group.get_seqs()[0].get_len(), 
                        self.block_manager.block_size, len(seq_group.get_seqs()[0].logical_token_blocks))

            req_allocated = [(seq_group.request_id,\
                len(self.block_manager.block_tables[seq_group.get_seqs()[0].seq_id])*self.block_manager.block_size,\
                seq_group.get_seqs()[0].get_len() + self.block_manager.block_size) \
                    for seq_group in self.running]
            print(f"req_allocated: {req_allocated}")
            gpu_blk_num = [(seq_group.request_id,\
                len(self.block_manager.block_tables[seq_group.get_seqs()[0].seq_id]), \
                self.block_manager.get_gpu_blk_num(seq_group=seq_group)) \
                    for seq_group in self.running]
            print(f"blk info: {gpu_blk_num}")



        # <jingzhi> TODO: 这个地方计算seq_group_tot_lens的时候有点问题，因为没有考虑一个group有多个sequence的情况
        # determine which requests (more specific, which blocks) to keep on card, which to swap out and which to recompute (if any)
        seq_group_lens = [sum([seq.get_len() for seq in \
                seq_group.get_seqs(status=SequenceStatus.RUNNING)]) for seq_group in self.running]
        # assume we know the exact output lengths for the requests
        seq_group_tot_lens = [ len(seq_group.prompt_token_ids) + seq_group.sampling_params.max_tokens\
            for seq_group in self.running]

        # seq_group_lens_ready = [sum([seq.get_len() for seq in \
        #         seq_group.get_seqs(status=SequenceStatus.SWAPPED)]) \
        #         for seq_group in self.my_scheduler_config.ready_to_run]
        # seq_group_tot_lens_ready = [ len(seq_group.prompt_token_ids) + seq_group.sampling_params.max_tokens\
        #     for seq_group in self.my_scheduler_config.ready_to_run]
        

        # get the swap list and the release-for-recompute list for this round
        if self.my_scheduler_config._is_first_iter_in_group():
            swap_list, recompute_list, swap_num_each_iter = my_swap_and_recompute.determine_swap_recompute(
                self.block_manager.gpu_allocator.num_blocks, 
                self.block_manager.block_size, 
                self.my_scheduler_config.blk_mem_size, 
                seq_group_lens, 
                seq_group_tot_lens,
                [seq_group.request_id for seq_group in self.running]
                )

            # save the swap&recompute decisions for later reference
            self.my_scheduler_config.swap_num_each_iter = swap_num_each_iter
            self.my_scheduler_config.complete_swap_list = swap_list
            self.my_scheduler_config.complete_recompute_list = recompute_list

            self.my_scheduler_config.swap_list = swap_list
            self.my_scheduler_config.recompute_list = recompute_list

            for seq_group in self.running:
                if my_swap_and_recompute.mem_scheduler.request_is_interrupted(seq_group.request_id):
                    self.my_scheduler_config.seq_group_dict[seq_group.request_id] = seq_group


            # <jingzhi> DEBUG
            print("Swap list generated: ", self.my_scheduler_config.complete_swap_list)
            print("Recompute list: ", self.my_scheduler_config.complete_recompute_list)
            print("On card block nums: ", my_swap_and_recompute.mem_scheduler.on_card_info)
            print(f"seq_group_dict: {self.my_scheduler_config.seq_group_dict.keys()}")
            print(f"self.block_tables: {self.block_manager.block_tables.keys()}")
            print(f"current free gpu & cpu blocks: {self.block_manager.gpu_allocator.get_num_free_blocks(), self.block_manager.cpu_allocator.get_num_free_blocks()}")
            tot_on_card_num = sum([((seq_group.get_seqs()[0].get_len() + self.block_manager.block_size - 1)//self.block_manager.block_size) \
                for seq_group in self.running])
            tot_on_card_num = tot_on_card_num + sum(my_swap_and_recompute.mem_scheduler.on_card_info.values())
            print(f"total on card blk num: {tot_on_card_num}")
        # do corresponding preemption
        # first get the blocks to swap out in this iteration

        # <jingzhi> TODO: 还是觉得可以实现一下撤销决定的work。但是暂时先不实现，先把整体的框架搭起来。
        # 所以下面这个last iter in group 的swap和recompute决定就暂时这么写，之后会再优化。
        
        swap_out_this_iter = list()
        for_recompute_this_iter = list()
        if self.my_scheduler_config._is_last_iter_in_group():
            # swap_out_this_iter = swap_list[:swap_num_each_iter]
            # swap_list = swap_list[swap_num_each_iter:]

            swap_out_this_iter = self.my_scheduler_config.swap_list
            self.my_scheduler_config.swap_list = list()

            for_recompute_this_iter = self.my_scheduler_config.recompute_list
        else:
            # we need to check whether a block is full when it is in self.running
            cur_lens = {seq_group.request_id: seq_group.get_seqs()[0].get_len() \
                for seq_group in self.running}


            print(f"cur_lens: {cur_lens}")
            tot_on_card_num = sum([((seq_group.get_seqs()[0].get_len() + self.block_manager.block_size - 1)//self.block_manager.block_size) \
                for seq_group in self.running])
            tot_on_card_num = tot_on_card_num + sum(my_swap_and_recompute.mem_scheduler.on_card_info.values())
            print(f"total on card blk num: {tot_on_card_num}")

            # print(f"cur_lens: {cur_lens}")
            # print(f"self.my_scheduler_config.swap_list: {self.my_scheduler_config.swap_list}")
            # print(f"my_swap_and_recompute.release_info: {my_swap_and_recompute.mem_scheduler.release_infor}")
            # print(f"my_swap_and_recompute.on_card_info: {my_swap_and_recompute.mem_scheduler.on_card_info}")

            select_ids = np.nonzero(
                [(cur_lens[request_id]>=blk_i*self.block_manager.block_size \
                    if request_id in cur_lens \
                    else True) \
                for request_id, blk_i in self.my_scheduler_config.swap_list])[0]\
                [:self.my_scheduler_config.swap_num_each_iter]
            swap_out_this_iter = [self.my_scheduler_config.swap_list[select_i] for select_i in select_ids]
            self.my_scheduler_config.swap_list = [\
                self.my_scheduler_config.swap_list[select_i] for select_i in \
                    range(len(self.my_scheduler_config.swap_list)) \
                if select_i not in select_ids]

        # # we need to check whether a block is full when it is in self.running
        # cur_lens = {seq_group.request_id: seq_group.get_seqs()[0].get_len() for seq_group in self.running}
        # select_ids = np.nonzero([cur_lens[request_id]<blk_i*self.block_manager.block_size for request_id, blk_i in swap_list])[:swap_num_each_iter]
        # swap_out_this_iter = [swap_list[select_i] for select_i in select_ids]
        # swap_list = [swap_list[select_i] for select_i in range(len(swap_list)) if select_i not in select_ids]


        # <jingzhi> DEBUG
        # print(f"swap_out_this_iter: {swap_out_this_iter}")

        # do preemption but do not update self.running and block mapping
        self._my_do_swap_out( swap_out_this_iter, for_recompute_this_iter, blocks_to_swap_out )
        for seq_group in self.running:
            self._append_slot(seq_group, blocks_to_copy)

        # ------------------------------------------------------------------------------

        # Swap in the sequence groups in the SWAPPED state if possible.

        # condition: (1) no gpu block release (2) the first iteration of the group
        if (not (self.my_scheduler_config.complete_swap_list or \
                self.my_scheduler_config.complete_recompute_list)) \
            and self.my_scheduler_config._is_first_iter_in_group() :
            swap_in_list, recompute_list = my_swap_and_recompute.determine_swap_in_do_recompute(
                self.block_manager.gpu_allocator.num_blocks, 
                self.block_manager.watermark_blocks,
                self.block_manager.block_size, 
                self.my_scheduler_config.blk_mem_size, 
                seq_group_lens, 
                seq_group_tot_lens
                )

            self.my_scheduler_config.complete_swap_in_list = swap_in_list
            self.my_scheduler_config.complete_recompute_list = recompute_list

            self.my_scheduler_config.swap_in_list = swap_in_list
            self.my_scheduler_config.recompute_list = recompute_list


            # <jingzhi> DEBUG
            print(f"swap_in_list: {swap_in_list}")
            print(f"recompute_list: {recompute_list}")
            print(f"total block number to add in: {len(swap_in_list) + len(recompute_list)}")

            self._my_do_swap_in( swap_in_list, recompute_list, blocks_to_swap_in )

        

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs






    # <jingzhi> page-swap and page-recompute
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.

        # <jingzhi>
        if self.my_scheduler_config.use_our_method:
            # scheduler_outputs = self._my_schedule()
            scheduler_outputs = self._schedule_outlen_aware()
        else:
            scheduler_outputs = self._schedule()
            # scheduler_outputs = self._schedule_peak_demand_aware_paged()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                # <jingzhi> for page-recompute
                to_recompute_blocks=\
                    self.my_scheduler_config.this_iter_recompute_dict[seq_group.request_id]\
                    if seq_group.request_id in self.my_scheduler_config.this_iter_recompute_dict\
                    else [],
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _my_allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.my_allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING


    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]





    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE

                # <jingzhi> For DEBUG
                preemption_mode = PreemptionMode.SWAP

            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED




    # <jingzhi> do swap out without update status information
    def _my_do_swap_out(
        self, 
        swap_out_this_iter: List[Tuple[str, int]],
        for_recompute_this_iter: List[Tuple[str, int]],
        blocks_to_swap_out: Dict[int, int]
        ) -> None:
        # we will do the checking when make swap out decisions
        if len(swap_out_this_iter) > self.block_manager.cpu_allocator.get_num_free_blocks():
            assert False, "Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error."
        mapping, blocks_to_free, new_blocks = self.block_manager.my_do_swap_out(
            swap_out_this_iter, for_recompute_this_iter,
            self.my_scheduler_config.seq_group_dict)
        blocks_to_swap_out.update(mapping)
        self.my_scheduler_config.blocks_to_free = self.my_scheduler_config.blocks_to_free + blocks_to_free
        self.my_scheduler_config.new_blocks = self.my_scheduler_config.new_blocks + new_blocks



    # <jingzhi> do swap out block by block and update status information
    def _my_do_swap_out_free_and_update_blocktables(
        self, 
        swap_out_this_iter: List[Tuple[str, int]],
        for_recompute_this_iter: List[Tuple[str, int]],
        blocks_to_swap_out: Dict[int, int]
        ) -> None:
        # we will do the checking when make swap out decisions
        if len(swap_out_this_iter) > self.block_manager.cpu_allocator.get_num_free_blocks():
            assert False, "Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error."
        mapping = self.block_manager.my_do_swap_out_free_and_update_blocktables(
            swap_out_this_iter, for_recompute_this_iter,
            self.my_scheduler_config.seq_group_dict)
        blocks_to_swap_out.update(mapping)





    def _my_do_swap_in(
        self, 
        swap_in_this_iter: List[Tuple[str, int]],
        recompute_list: List[Tuple[str, int]],
        blocks_to_swap_in: Dict[int, int]
        ) -> None:
        mapping, blocks_to_free, new_blocks = self.block_manager.my_do_swap_in(swap_in_this_iter, recompute_list, self.my_scheduler_config.seq_group_dict)
        blocks_to_swap_in.update(mapping)
        self.my_scheduler_config.blocks_to_free = self.my_scheduler_config.blocks_to_free + blocks_to_free
        self.my_scheduler_config.new_blocks = self.my_scheduler_config.new_blocks + new_blocks






    # do page-swap-in, free cpu blocks and update blocktables
    def _my_do_swap_in_free_and_update_blocktables(
        self, 
        swap_in_this_iter: List[Tuple[str, int]],
        recompute_list: List[Tuple[str, int]],
        blocks_to_swap_in: Dict[int, int]
        ) -> None:
        mapping = self.block_manager.my_do_swap_in_free_and_update_blocktables(
            swap_in_this_iter, recompute_list, self.my_scheduler_config.seq_group_dict)
        blocks_to_swap_in.update(mapping)











def DP_select_requests_to_release(curr_on_card, candidates, tot_blk_num, to_release, best = [[], [], float('inf')], run_simple_test=False):
    '''
    to_release: a list of already selected requests
    best: [the best to_release set, the best to_keep set, the future #block demand of the best solution]
    candidates: the candidate requests to be released
    '''
    def get_blk_num(seqlen:List[int])->int:
        block_size = 1 if run_simple_test else 16 # 16 # simple test
        return sum((seqlen+block_size-1)//block_size)

    # compute the total block demand in this iter
    demand = get_blk_num(np.asarray([info[0] for info in curr_on_card]))

    if demand <= tot_blk_num:
        # we do not need release requests
        # compare with best
        future_peak_demand = 0
        tmp_req_list = sorted(curr_on_card, key=lambda info: info[1])
        curr_lens = np.asarray([info[0] for info in tmp_req_list])
        
        for i, info in enumerate(tmp_req_list):
            blk_num = get_blk_num(info[1] - 1 + curr_lens[i:]) - demand
            if blk_num > future_peak_demand:
                future_peak_demand = blk_num


        # if run_simple_test:
        #     print(f"--In DP--: curr_on_card: {curr_on_card}, to_release: {to_release}, future_peak_demand: {future_peak_demand}, best: {best}, demand: {demand}, tot_blk_num: {tot_blk_num}")
        # else:
        #     print(f"--In DP--: to_release: {to_release}, future_peak_demand: {future_peak_demand}, best: {best[0],best[2]}, demand: {demand}, tot_blk_num: {tot_blk_num}")

        if future_peak_demand < best[2]:
            best[0] = copy.deepcopy(to_release)
            best[1] = copy.deepcopy(curr_on_card)
            best[2] = future_peak_demand
        return True

    # 

    # <jingzhi> For DEBUG
    # print(curr_on_card, candidates)
    assert (curr_on_card[-len(candidates):] == candidates) or (len(candidates) == 0) # candidates=[] means this solution is not feasible


    best_idx = None
    # to_release only need to be a set not an ordered sequence
    for info_i, info in enumerate(candidates):
        # print(f"to_release: {to_release}, info_i:{info_i}, best_idx:{best_idx}")
        if best_idx!=None:
            if (info_i != best_idx):
                continue
        to_release = to_release + [info]
        new_curr_on_card = curr_on_card[:len(curr_on_card)-len(candidates)+info_i] + curr_on_card[len(curr_on_card)-len(candidates)+info_i + 1: ]
        reach_end = DP_select_requests_to_release(new_curr_on_card, candidates[info_i+1:], tot_blk_num, to_release, best)
        to_release = to_release[:-1]

        # if find solution in this round, we can skip some candidates because their peak future demand must be higher
        # we already sort candidates by in_lens from large to small
        if reach_end:
            assert (best_idx == None) or (best_idx == info_i), best_idx # we will only enter this condition once
            if best_idx!=None:
                continue
            # check the candidate in this range with the largest output length and skip other candidates
            best_idx = np.argsort([tmp_info[1] for tmp_info in candidates[info_i:]])[-1]
            best_idx = info_i + best_idx
            # print(f"demand - tot_blk_num: {demand - tot_blk_num}, {[get_blk_num(np.asarray([info[0]])) for info in candidates]}")
            # print(f"best_idx: {best_idx}, info_i: {info_i}, to_release: {to_release}")


    return False