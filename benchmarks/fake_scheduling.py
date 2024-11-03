"""
This file does fake scheduling based on the given output lengths.
The output lengths can be obtained by sampling following the output length distribution 
we obtain from experiment results (on the no-robot dataset).
"""


from typing import List, Optional, Tuple
import numpy as np
from my_per_iter_latency_estimator import CostTable
from vllm.engine.metrics import MyThroughputLogger


def remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size):
    '''
        Output:
            Modify running_seqs.
            Return: running_seqs_num, unfinished_reqnum, token_num_stored
    '''
    indices = running_seqs[2][:running_seqs_num].nonzero()[0]
    new_running_seqs_num = len(indices)
    running_seqs[0][:new_running_seqs_num] = running_seqs[0][indices]
    running_seqs[1][:new_running_seqs_num] = running_seqs[1][indices]
    running_seqs[2][:new_running_seqs_num] = running_seqs[2][indices]
    new_unfinished_reqnum = unfinished_reqnum - (running_seqs_num - new_running_seqs_num)
    # new_token_num_stored = sum(running_seqs[1][:new_running_seqs_num] - 1)
    new_block_num_used = sum((running_seqs[1][:new_running_seqs_num] - 1 + block_size - 1) // block_size)
    return new_running_seqs_num, new_unfinished_reqnum, new_block_num_used




def kill_seqs_for_more_cache_space(
        running_seqs, 
        max_block_num, block_size, running_seqs_num, 
        inp_lens, out_lens, seq_ids, pointer):
    '''
        Kill seqs to release cache slots.
        Output:
            Update running_seqs.
            Return: new_block_num_used, running_seqs_num, inp_lens, out_lens, pointer.
    '''
    ori_running_seqs_num = running_seqs_num
    # while token_num_stored + running_seqs_num > max_cache_slot_num:
    #     # we cannot support all seqs running together
    #     token_num_stored = token_num_stored - running_seqs[1][running_seqs_num-1]
    #     running_seqs_num -= 1

    # =============================================================================================================
    # compute the running seq num we can support in the next iteration
    # new_block_nums = (running_seqs[1][:running_seqs_num] + block_size - 1) // block_size
    # if sum(new_block_nums) > max_block_num:
    #     # we cannot support all seqs running together
    #     running_seqs_num = np.nonzero(np.cumsum(new_block_nums) > max_block_num)[0][0]
    # =============================================================================================================

    # =============================================================================================================
    # we follow the heuristics in vllm: keep a request only if there is a block (not a slot) left for it.=========
    new_block_nums = (running_seqs[1][:running_seqs_num] + block_size - 1) // block_size
    approx_tot_block_nums = (running_seqs[1][:running_seqs_num] - 1 + block_size - 1) // block_size + 1
    approx_tot_block_nums[1:] = approx_tot_block_nums[1:] + np.cumsum(new_block_nums)[:-1]
    if max(approx_tot_block_nums) > max_block_num:
        # we cannot support all seqs running together
        running_seqs_num = np.nonzero(approx_tot_block_nums > max_block_num)[0][0]
    # =============================================================================================================

    # print(f"sum(new_block_nums): {sum(new_block_nums)}, max_block_num: {max_block_num}")
    # print(f"sum(new_block_nums[:running_seqs_num]): {sum(new_block_nums[:running_seqs_num])}, {sum(new_block_nums[:running_seqs_num+1])}")
    # print(f"ori_running_seqs_num: {ori_running_seqs_num}, running_seqs_num: {running_seqs_num}")
    new_block_num_used = sum((running_seqs[1][:running_seqs_num] - 1 + block_size - 1) // block_size)

    # now the cache slots are enough
    killed_seqs_num = ori_running_seqs_num - running_seqs_num
    seq_ids[pointer-killed_seqs_num : pointer] = running_seqs[0][running_seqs_num:ori_running_seqs_num]
    inp_lens[pointer-killed_seqs_num : pointer] = running_seqs[1][running_seqs_num:ori_running_seqs_num]
    out_lens[pointer-killed_seqs_num : pointer] = running_seqs[2][running_seqs_num:ori_running_seqs_num]
    pointer -= killed_seqs_num
    return new_block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer




def _add_one_infer_rng(rngs, start, end):
    if len(rngs) == 0:
        rngs.extend([start, end])
    else:
        if (rngs[-1] + 1) == start:
            # the range [start, end] can be merged to the last range
            rngs[-1] = end
        else:
            rngs.extend([start, end])


def _store_infer_state(start, end, infer_progress, seq_ids):
    # print(f"_store_infer_state: {start}-{end}, seq_ids: {seq_ids}")
    for seq_i in seq_ids:
        # infer_progress[seq_i].extend([tot_iter_num, tot_iter_num])
        _add_one_infer_rng(infer_progress[seq_i], start, end)



def update_prefill_logs_NO_max_infer_step_num_limit(
        prefill_logs: List[Tuple[int, int, int, int]], 
        prompt_lens, max_num_batched_tokens, 
        prompt_ids, infer_progress, tot_iter_num) -> int:
    '''
        Get the steps to complete the prefill stages for the given prompt_lens.
        INPUT:
            tot_iter_num: the current total iteration number.
        Output:
            1. Add new tuples to prefill_logs, 
            each tuple is (seqnum, tot_token_num, attention_sum, max_seqlen) of a prefill step.
            2. store per iter infer information in infer_progress, i.e., which prompt ids are involved for each iter.
            NOTE:
                we only consider the ``max_num_batched_tokens'' constraint here.
                the ``max_seq_num'' and `` cache space'' are not considered 
                because they are considered in ``fake_FCFS_schedule''.
            NOTE:
                all prompts are padded to the max len.
            NOTE:
                in this version, we generate all the prefill steps required to start all the given prompts.
    '''
    def get_prefill_step(seqs):
        seqnum = len(seqs)
        max_seqlen = max(seqs)
        tot_token_num = sum(seqs)
        attention_sum = sum([(1+si)*si for si in seqs])
        return [seqnum, tot_token_num, attention_sum, max_seqlen]
    # 
    if len(prompt_lens)==0:
        # no need to update prefill_logs
        return tot_iter_num
    # 
    seqs = list()
    seq_ids = list()
    for i, seq_id in zip(prompt_lens, prompt_ids):
        if max(seqs+[i]) * (len(seqs)+1) <= max_num_batched_tokens:
            # can add to the current step
            seqs.append(i)
            seq_ids.append(seq_id)
        else:
            prefill_logs.append(get_prefill_step(seqs))
            _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
            tot_iter_num += 1
            # move to the next step
            seqs = [i]
            seq_ids = [seq_id]
    # 
    # deal with the last step
    prefill_logs.append(get_prefill_step(seqs))
    _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
    tot_iter_num += 1
    
    return tot_iter_num



# model-level pipeline
def update_prefill_logs(
        prefill_logs: List[Tuple[int, int, int, int]], 
        prompt_lens, max_num_batched_tokens, 
        prompt_ids, infer_progress, tot_iter_num, 
        # below are parameters for model-level pipeline
        need_query_available_requests: bool, 
        check_gap: int,
        last_iter_seqs: List[int], 
        last_iter_seq_ids: List[int],
        must_record_first_step: bool,
        ) -> Tuple[List[int], List[int], int]:
    '''
        Get the steps to complete the prefill stages for the given prompt_lens.
        INPUT:
            tot_iter_num: the current total iteration number.
            last_iter_seqs/last_iter_seq_ids: the prompts from last iter which is remained because we need to check 
                if there is newly available input requests.
        Output:
            1. Add new tuples to prefill_logs, 
            each tuple is (seqnum, tot_token_num, attention_sum, max_seqlen) of a prefill step.
            2. store per iter infer information in infer_progress, i.e., which prompt ids are involved for each iter.
            NOTE:
                we only consider the ``max_num_batched_tokens'' constraint here.
                the ``max_seq_num'' and `` cache space'' are not considered 
                because they are considered in ``fake_FCFS_schedule''.
            NOTE:
                all prompts are padded to the max len.
            NOTE:
                in this version, we will stop the prefill step generation when 
                    (1) ``need_query_available_requests`` is True, i.e., with the last step, there will be no waiting
                    request in the current waiting list.
                    (2) the step % check_gap == 0
    '''
    def get_prefill_step(seqs):
        seqnum = len(seqs)
        max_seqlen = max(seqs)
        tot_token_num = sum(seqs)
        attention_sum = sum([(1+si)*si for si in seqs])
        return [seqnum, tot_token_num, attention_sum, max_seqlen]
    # 
    # print(f"last_iter_seqs: {last_iter_seqs}")
    # print(f"update_prefill_logs: prompt_lens, prompt_ids, last_iter_seqs, last_iter_seq_ids: {prompt_lens, prompt_ids, last_iter_seqs, last_iter_seq_ids}")

    # if len(prompt_lens)==0:
    if (len(prompt_lens)==0) and (len(last_iter_seqs)==0):
        # no need to update prefill_logs
        return list(), list(), tot_iter_num
    # 

    # must_record_first_step = False
    # if len(last_iter_seqs) > 0:
    #     must_record_first_step = True

    # seqs = list()
    # seq_ids = list()
    seqs = last_iter_seqs
    seq_ids = last_iter_seq_ids
    for i, seq_id in zip(prompt_lens, prompt_ids):
        if max(seqs+[i]) * (len(seqs)+1) <= max_num_batched_tokens:
            # can add to the current step
            seqs.append(i)
            seq_ids.append(seq_id)
        else:
            prefill_logs.append(get_prefill_step(seqs))
            _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
            tot_iter_num += 1
            # move to the next step
            seqs = [i]
            seq_ids = [seq_id]

            # reset must_record_first_step for non-first steps 
            must_record_first_step = False
    # 
    # deal with the last step
    
    # we may need to check whether there is newly available input requests
    # 只有input req全都用完了，但是还有空间剩余的情况下才需要看有没有新的request，所以只需要提前一个轮次查看即可；
    # 在所有input req全都用完的情况下，每个K轮查询一次。查询完不管有没有更新都继续应该做的inference scheduling。
    # 但是接着做应该做的inference的时候就有一个问题：我们在模拟的时候做了一下加速，这些加速还等价吗？感觉OK的，就是
    # 应该在不符合正常做inference的条件的时候跳出即可。
    # 查询条件是：inp request还剩最后一个轮次就用完了，或者已经用完了，并且当前轮次是K的倍数；
    # 不查询条件是：inp request做完当前轮次还有剩余，或者不是K的倍数
    # print(f"update_prefill_logs: must_record_first_step: {must_record_first_step}, need_query_available_requests: {need_query_available_requests}, tot_iter_num: {tot_iter_num}, seq_ids: {seq_ids}")
    if (not must_record_first_step) \
        and need_query_available_requests and (tot_iter_num % check_gap == 0):
        # record the last step 
        return seqs, seq_ids, tot_iter_num
    
    
    prefill_logs.append(get_prefill_step(seqs))
    _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
    tot_iter_num += 1
    
    return list(), list(), tot_iter_num




def _fake_FCFS_schedule_NO_continuous_model_level_pipeline(
        inp_lens: List[int], out_lens: List[int], 
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int):
    '''
        Do the fake scheduling using the first-come-first-serve policy.
        inp_lens: the input lengths of the requests.
        out_lens: the output lengths of the requests.
        max_seq_num: the maximum number of requests running concurrently.
        max_cache_slot_num: the maximum number of tokens whose KV cache can be stored at the same time.
        There is only two constraints when trying to add a running request:
            (1) max_seq_num; (2) max_cache_slot_num (consider watermark=0.01).
        NOTE:
            (1) We ignore the block size here to make the fake schedule faster.
                --> it seems there will be a lot of request kill when block size is 1, 
                --> so we HAVE TO CONSIDER block size!
            (2) For prefill stage, we also consider  
                ``max_num_batched_tokens'' and TODO ``scheduler_config.max_paddings''. [Try this first]
            (3) When killing seqs, consider if there is an extra block for each sequence.
    '''
    def has_enough_cache(block_num_used, new_token_num, consider_watermark=False):
        # return token_num_stored < max_cache_slot_num
        new_block_num = (new_token_num + block_size - 1) // block_size
        if consider_watermark:
            watermark_blocks = 0.01 * max_block_num
            return max_block_num - block_num_used - new_block_num >= watermark_blocks
        return (block_num_used + new_block_num) <= max_block_num
    def add_block_num_used(block_num_used, new_token_num):
        new_block_num = (new_token_num + block_size - 1) // block_size
        block_num_used = block_num_used + new_block_num
        return block_num_used
    def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
        # iter_num = (max_cache_slot_num - token_num_stored) // running_seqs_num
        # iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        # first compute how many blocks can be assigned to each running seq at most
        # print(f"in get_max_iter_num: block_num_used: {block_num_used}, running_seqs_num: {running_seqs_num}")
        iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
        # print(f"iter_num: {iter_num}")
        
        # then the running seqs cannot run >= 16 iters
        extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
        # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}")
        extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
        # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}, counts: {counts.tolist()}")
        block_num_left = (max_block_num - block_num_used) % running_seqs_num
        # print(f"in get_max_iter_num: block_num_left: {block_num_left}")
        # print(f"in get_max_iter_num: np.nonzero(np.cumsum(counts) > block_num_left): {np.nonzero(np.cumsum(counts) > block_num_left)}")
        extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
        iter_num += extra_iter_num
        # print(f"in get_max_iter_num: extra_iter_num: {extra_iter_num}")
        # print(f"iter_num: {iter_num}, {type(iter_num)}")
        # print(f"extra_iter_nums: {extra_iter_nums}, {type(extra_iter_nums[0])}")
        # print(f"{extra_iter_nums, counts, block_num_left, extra_iter_num}")

        # now consider the remaining iters for each running seq
        iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        # print(f"{min(running_seqs[2][:running_seqs_num])}")

        return iter_num
    def get_tot_token_num(running_seqs_num, running_seqs):
        return sum(running_seqs[1][:running_seqs_num])
    def get_max_seqlen(running_seqs_num, running_seqs):
        return max(running_seqs[1][:running_seqs_num])
    # 

    # TODO: remove this copy as it is only used for assertion
    ori_inplens = inp_lens.copy()
    ori_outlens = out_lens.copy()

    unfinished_reqnum = len(inp_lens)
    running_seqs = np.zeros((3, max_seq_num), dtype=np.int32) # three rows: index (i.e., seq_id), gened token num, remaining token num
    seq_ids = list(range(len(inp_lens)))
    pointer = 0 # pointing to the next index of requests to consider
    running_seqs_num = 0
    # token_num_stored = 0 # for a seq, the number of token stored is (seq - 1)
    block_num_used = 0
    logs = list()
    prefill_logs = list() # each item is (seqnum, tot_token_num, attention_sum, max_seqlen)
    
    # store the inference progress of each sequence, for each seq, we store its continuous infer iter ranges
    infer_progress = list([] for _ in range(len(inp_lens)))
    # stores whether each step is a prefill step or not
    is_prefill_steps: List[bool] = list()
    tot_iter_num: int = 0 # the current total iteration number, == len(logs) + len(prefill_logs)

    while unfinished_reqnum:
        # old_running_seqs_num = running_seqs_num
        new_prompt_lens: List[int] = list()
        new_prompt_ids: List[int] = list()
        while (pointer < len(inp_lens)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
            # we try to add new requests
            # if token_num_stored + inp_lens[pointer] <= max_cache_slot_num:
            if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
                running_seqs[0][running_seqs_num] = seq_ids[pointer] # pointer
                running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
                running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


                new_prompt_lens.append(inp_lens[pointer])
                new_prompt_ids.append(seq_ids[pointer])
                if running_seqs[2][running_seqs_num] == 0:
                    # this seq is finished
                    unfinished_reqnum -= 1
                    pointer += 1
                    continue


                # token_num_stored = token_num_stored + inp_lens[pointer]
                block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
                pointer += 1
                running_seqs_num += 1
            else:
                break


        # update prefill logs
        # new_prompt_lens = np.asarray(running_seqs[1][old_running_seqs_num:running_seqs_num])-1
        # new_prompt_ids = running_seqs[0][old_running_seqs_num:running_seqs_num]
        new_prompt_lens = np.asarray(new_prompt_lens)
        new_prompt_ids = np.asarray(new_prompt_ids)
        # if len(new_prompt_lens) > 0:
        #     print(f"pointer: {pointer}, start new seqs: {new_prompt_lens}")
        # print(f"in _fake_FCFS_schedule_NO_continuous_model_level_pipeline")
        _, _, tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
                            new_prompt_ids, infer_progress, tot_iter_num,
                            need_query_available_requests=False, check_gap=1,
                            last_iter_seqs=list(), last_iter_seq_ids=list(),
                            must_record_first_step=False)
        is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))
        # 
        # kill some running reqs if the cache is not enough
        block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
            kill_seqs_for_more_cache_space(
                running_seqs, 
                max_block_num, block_size, running_seqs_num, 
                inp_lens, out_lens, seq_ids, pointer)

        # 
        # collect decoding stage logs
        # 1. compute the number of iters the current running reqs can run
        # consider: available cache slots, seq remaining output tokens
        iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
        # print(f"iter_num: {iter_num}")
        # print(f"{running_seqs[0][:running_seqs_num].tolist()}")
        # print(f"{running_seqs[1][:running_seqs_num].tolist()}")
        # print(f"{running_seqs[2][:running_seqs_num].tolist()}")
        tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
        curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
        logs.extend([(running_seqs_num, 
                      tot_token_num + running_seqs_num*i,
                      tot_token_num + running_seqs_num*i,
                      curr_max_seqlen + i) \
                     for i in range(iter_num)])
        _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
        is_prefill_steps.extend([False]*iter_num)
        tot_iter_num += iter_num

        # 
        # 2. update the status of the running seqs
        running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
        running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

        # remove finished reqs
        running_seqs_num, unfinished_reqnum, block_num_used = \
            remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)
        # 
        # now go back to the top of the loop
    # here we finish the fake scheduling.
    # for i, step in enumerate(logs):
    #     print(f"step {i}: {step}")
    # for i, step in enumerate(prefill_logs):
    #     print(f"prefill step {i}: {step}")
    # 
    assert tot_iter_num == (len(logs) + len(prefill_logs)), (tot_iter_num, len(logs), len(prefill_logs), ori_inplens, ori_outlens, max_seq_num, max_block_num, max_num_batched_tokens, block_size) 
    return logs, prefill_logs, is_prefill_steps, infer_progress




def _check_new_input_requests(
        sort_input: bool, 
        # 
        seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],
        time_when_checking: float,
        # 
        pointer: int,
        running_seqs_num: int,
        ) -> Tuple[List[int], List[int], List[int], List[float], float]:
    """
        This function checks whether there are new input requests and update the input request array if any.
        INPUT:
            sort_input: controls whether we need to sort the waiting input list every time we add some new inputs.
            pointer: seq_ids[pointer:] are the sequences currently received and in the waiting list.
        UPDATE:
            seq_ids, inp_lens, out_lens, arrive_times, time_when_checking.
        NOTE: 
            1. if we want to sort the input request list, we need to reorder seq_ids, inp_lens, out_lens accordingly 
            (we do not change the seq_id of any input request).
            2. when there is no new request at ``time_when_checking'' but there are input requests we need to wait, 
            we need to update ``time_when_checking'' to the latest time we can receive a new input request.
    """
    if len(seq_ids) == len(inp_lens):
        # no more input requests to receive
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking
    
    if (running_seqs_num > 0) and (arrive_times[len(seq_ids)] > time_when_checking):
        # there are running requests but currently there is no available input requests
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking
    
    if arrive_times[len(seq_ids)] > time_when_checking:
        # no available input requests at time_when_checking
        # change time_when_checking and collect available input requests 
        time_when_checking = arrive_times[len(seq_ids)]


    seq_id = None
    new_inp_end = None
    # TODO: 这里可能有可以加速的机会，换成searchsorted之类的函数
    for seq_id in range(len(seq_ids), len(inp_lens)):
        if arrive_times[seq_id] > time_when_checking:
            new_inp_end = seq_id
            break
    
    if new_inp_end == None:
        # all inputs are available now
        new_inp_end = len(inp_lens)
    
    waiting_inp_lens = inp_lens[pointer:new_inp_end]
    seq_ids = np.concatenate((seq_ids, (range(pointer, new_inp_end))))

    # print(f"in _check_new_input_requests----------")
    # print(f"waiting_inp_lens: {waiting_inp_lens}")
    # print(f"pointer: {pointer}")
    # print(f"new_inp_end: {new_inp_end}")
    # print(f"seq_ids: {seq_ids}")

    if sort_input:
        order = np.argsort(-waiting_inp_lens, kind='stable')
        seq_ids[pointer:] = seq_ids[pointer:][order]
        inp_lens[pointer:new_inp_end] = inp_lens[pointer:new_inp_end][order]
        out_lens[pointer:new_inp_end] =out_lens[pointer:new_inp_end][order]
        arrive_times[pointer:new_inp_end] =arrive_times[pointer:new_inp_end][order]
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking
    else:
        # we do not reorder the waiting input requests
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking






def _update_seq_info_with_known_arrive_time_deprecated(
        time_when_check: float,
        pointer: int,
        new_out_seq_ids: List[int],
        # 
        ref_seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],        
        # 
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],
        # 
        infer_progress, full_infer_progress, curr_model_level_id: int,
        ):
    """
        This function update the info of the seqs whose arrive time become known every time there is 
        new output generated.
        NOTE:
            1. the seqs in ref_seq_ids_list, ..., are sorted by the seq ids.
            2. we need to sort the ready seqs by their arrive times.
    """
    if len(ref_seq_ids_list) == 0:
        return ref_seq_ids, inp_lens, out_lens, arrive_times, \
            ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list

    # some output seqs are not needed in the next round
    seq_ids_to_add = list(set(new_out_seq_ids).intersection(ref_seq_ids_list[0]))
    inds = np.searchsorted(ref_seq_ids_list[0], seq_ids_to_add)
    inp_lens_to_add = inp_lens_list[0][inds]
    out_lens_to_add = out_lens_list[0][inds]
    arrive_times_to_add = np.maximum(arrive_times_list[0][inds], time_when_check)
    
    # update infer_progress
    for ind, seq_id in zip(inds, seq_ids_to_add):
        infer_progress[seq_id] = full_infer_progress[curr_model_level_id][ind]
    
    # add the seqs to the ready lists
    ref_seq_ids = np.concatenate((ref_seq_ids, seq_ids_to_add))
    inp_lens = np.concatenate((inp_lens, inp_lens_to_add))
    out_lens = np.concatenate((out_lens, out_lens_to_add))
    arrive_times = np.concatenate((arrive_times, arrive_times_to_add))
    # sort the ready seqs by their arrive_times
    order = np.argsort(arrive_times[pointer:])+pointer
    ref_seq_ids[pointer:] = ref_seq_ids[pointer:][order]
    inp_lens[pointer:] = inp_lens[pointer:][order]
    out_lens[pointer:] = out_lens[pointer:][order]
    arrive_times[pointer:] = arrive_times[pointer:][order]


    # remove the seqs from the cand lists
    remaining_inds = set(range(len(ref_seq_ids_list[0]))).difference(inds)
    remaining_inds = sorted(remaining_inds)
    ref_seq_ids_list[0] = ref_seq_ids_list[0][remaining_inds]
    inp_lens_list[0] = inp_lens_list[0][remaining_inds]
    out_lens_list[0] = out_lens_list[0][remaining_inds]
    arrive_times_list[0] = arrive_times_list[0][remaining_inds]

    if len(remaining_inds) == 0:
        ref_seq_ids_list = ref_seq_ids_list[1:]
        inp_lens_list = inp_lens_list[1:]
        out_lens_list = out_lens_list[1:]
        arrive_times_list = arrive_times_list[1:]
        curr_model_level_id += 1

    return ref_seq_ids, inp_lens, out_lens, arrive_times, \
        ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list, curr_model_level_id
    








def _update_seq_info_with_known_arrive_time(
        time_when_check: float,
        running_seq_ids: List[int],
        pointer: int,
        # 
        ref_seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],        
        # 
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],
        # 
        infer_progress, full_infer_progress, fixed_ref_seq_ids_list,
        ):
    """
        This function update the info of the seqs whose arrive time become known every time there is 
        new output generated.
        NOTE:
            1. the seqs in ref_seq_ids_list, ..., and fixed_ref_seq_ids_list, are sorted by the seq ids. --> !! no such requirement !!
            2. we need to sort the ready seqs by their arrive times.
    """
    # if len(ref_seq_ids_list) == 0:
    #     return ref_seq_ids, inp_lens, out_lens, arrive_times, \
    #         ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list



    for i in range(len(ref_seq_ids_list)):

        # get the seqs whose arrive times are known and not finished
        running_or_pending_seq_ids = np.concatenate((running_seq_ids, ref_seq_ids[pointer:]))


        # 1. some output seqs are not needed in the next round
        seq_ids_to_add = np.asarray(sorted(set(ref_seq_ids_list[i]).difference(running_or_pending_seq_ids)), dtype=np.int64)
        # seq_ids_to_add = list(set(new_out_seq_ids).intersection(ref_seq_ids_list[0]))
        inds = np.searchsorted(ref_seq_ids_list[i], seq_ids_to_add)
        inp_lens_to_add = inp_lens_list[i][inds]
        out_lens_to_add = out_lens_list[i][inds]
        arrive_times_to_add = np.maximum(arrive_times_list[i][inds], time_when_check)
        
        # 2. update infer_progress
        _inds = np.searchsorted(fixed_ref_seq_ids_list[i], seq_ids_to_add)
        for ind, seq_id in zip(_inds, seq_ids_to_add):
            infer_progress[seq_id] = full_infer_progress[i+1][ind]
        
        # 3. add the seqs to the ready lists
        ref_seq_ids = np.concatenate((ref_seq_ids, seq_ids_to_add))
        inp_lens = np.concatenate((inp_lens, inp_lens_to_add))
        out_lens = np.concatenate((out_lens, out_lens_to_add))
        arrive_times = np.concatenate((arrive_times, arrive_times_to_add))

        # sort the ready seqs by their arrive_times
        order = np.argsort(arrive_times[pointer:], kind='stable')
        ref_seq_ids[pointer:] = ref_seq_ids[pointer:][order]
        inp_lens[pointer:] = inp_lens[pointer:][order]
        out_lens[pointer:] = out_lens[pointer:][order]
        arrive_times[pointer:] = arrive_times[pointer:][order]

        # 4. remove the seqs from the cand lists
        remaining_inds = set(range(len(ref_seq_ids_list[i]))).difference(inds)
        remaining_inds = sorted(remaining_inds)
        # print(f"inds: {inds}, remaining_inds: {remaining_inds}")
        ref_seq_ids_list[i] = ref_seq_ids_list[i][remaining_inds]
        inp_lens_list[i] = inp_lens_list[i][remaining_inds]
        out_lens_list[i] = out_lens_list[i][remaining_inds]
        arrive_times_list[i] = arrive_times_list[i][remaining_inds]

        # if len(remaining_inds) == 0:
        #     ref_seq_ids_list = ref_seq_ids_list[1:]
        #     inp_lens_list = inp_lens_list[1:]
        #     out_lens_list = out_lens_list[1:]
        #     arrive_times_list = arrive_times_list[1:]
        #     curr_model_level_id += 1

    return ref_seq_ids, inp_lens, out_lens, arrive_times, \
        ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list
    


















def _check_new_input_requests_support_vertical_fuse(
        sort_input: bool, 
        # 
        seq_ids: List[int],
        # 
        ref_seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],
        time_when_checking: float,
        # 
        pointer: int,
        running_seqs_num: int,
        # 
        ) -> Tuple[List[int], List[int], List[int], List[float], float]:
    """
        This function checks whether there are new input requests and update the input request array if any.
        INPUT:
            sort_input: controls whether we need to sort the waiting input list every time we add some new inputs.
            pointer: seq_ids[pointer:] are the sequences currently received and in the waiting list.
        UPDATE:
            seq_ids, inp_lens, out_lens, arrive_times, time_when_checking,
            ref_seq_ids; ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list
        NOTE: 
            1. if we want to sort the input request list, we need to reorder seq_ids, inp_lens, out_lens accordingly 
            (we do not change the seq_id of any input request).
            2. when there is no new request at ``time_when_checking'' but there are input requests we need to wait, 
            we need to update ``time_when_checking'' to the latest time we can receive a new input request.
            3. this version supports model vertical fusion.
                ``ref_seq_ids``, ``inp_lens``, ``out_lens``, ``arrive_times`` store the info of seqs whose arrive times are known
                ``ref_seq_ids_list``, ``inp_lens_list``, ``out_lens_list``, ``arrive_times_list`` 
                    store the info of seqs of each model which are fused together
                Both of these two groups of variables will be modified in this method.
    """
    
    # then try to fetch available seqs

    if len(seq_ids) == len(inp_lens):
        # no more input requests to receive
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking
    
    if (running_seqs_num > 0) and (arrive_times[len(seq_ids)] > time_when_checking):
        # there are running requests but currently there is no available input requests
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking
    
    if arrive_times[len(seq_ids)] > time_when_checking:
        # no available input requests at time_when_checking
        # change time_when_checking and collect available input requests 
        time_when_checking = arrive_times[len(seq_ids)]


    seq_id = None
    new_inp_end = None
    # TODO: 这里可能有可以加速的机会，换成searchsorted之类的函数
    for seq_id in range(len(seq_ids), len(inp_lens)):
        if arrive_times[seq_id] > time_when_checking:
            new_inp_end = seq_id
            break
    
    if new_inp_end == None:
        # all inputs are available now
        new_inp_end = len(inp_lens)
    
    waiting_inp_lens = inp_lens[pointer:new_inp_end]
    # seq_ids = np.concatenate((seq_ids, (range(pointer, new_inp_end))))
    # NOTE: we are given a list of ref_seq_ids now
    seq_ids = np.concatenate((seq_ids, ref_seq_ids[pointer:new_inp_end]))

    # print(f"in _check_new_input_requests----------")
    # print(f"waiting_inp_lens: {waiting_inp_lens}")
    # print(f"pointer: {pointer}")
    # print(f"new_inp_end: {new_inp_end}")
    # print(f"seq_ids: {seq_ids}")

    if sort_input:
        order = np.argsort(-waiting_inp_lens, kind='stable')
        seq_ids[pointer:] = seq_ids[pointer:][order]
        inp_lens[pointer:new_inp_end] = inp_lens[pointer:new_inp_end][order]
        out_lens[pointer:new_inp_end] =out_lens[pointer:new_inp_end][order]
        arrive_times[pointer:new_inp_end] =arrive_times[pointer:new_inp_end][order]
        # 
        ref_seq_ids[pointer:new_inp_end] =ref_seq_ids[pointer:new_inp_end][order]
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking
    else:
        # we do not reorder the waiting input requests
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking







# def _do_K_step_inferene(
#         inp_lens: List[int], out_lens: List[int], arrive_times: List[float], check_gap: int,
#         max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
#         block_size: int,
#         sort_input: bool,
#         # 
#         unfinished_reqnum: int,
#         running_seqs: List[int],
#         seq_ids: List[int],
#         pointer: int,
#         running_seqs_num: int,
#         block_num_used: int,
#         logs: List[Tuple[int, int, int, int]],
#         prefill_logs: List[Tuple[int, int, int, int]],
#         infer_progress: List[List[int]],
#         is_prefill_steps: List[bool],
#         tot_iter_num: int,
#         ):
#     """
#         This function runs K (i.e., check_gap) step fake scheduling starting from the given inference progress.
#         NOTE:
#             1. if before we finish the K inference steps we run out of requests, we stop the inference process 
#             and turn to waiting more available input requests.
#             2. in the current code, ``sort_input`` performs differently from the version that we must query every 
#             check_gap steps.
#     """
#     def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
#         # iter_num = (max_cache_slot_num - token_num_stored) // running_seqs_num
#         # iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

#         # first compute how many blocks can be assigned to each running seq at most
#         # print(f"in get_max_iter_num: block_num_used: {block_num_used}, running_seqs_num: {running_seqs_num}")
#         iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
#         # print(f"iter_num: {iter_num}")
        
#         # then the running seqs cannot run >= 16 iters
#         extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
#         # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}")
#         extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
#         # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}, counts: {counts.tolist()}")
#         block_num_left = (max_block_num - block_num_used) % running_seqs_num
#         # print(f"in get_max_iter_num: block_num_left: {block_num_left}")
#         # print(f"in get_max_iter_num: np.nonzero(np.cumsum(counts) > block_num_left): {np.nonzero(np.cumsum(counts) > block_num_left)}")
#         extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
#         iter_num += extra_iter_num
#         # print(f"in get_max_iter_num: extra_iter_num: {extra_iter_num}")
#         # print(f"iter_num: {iter_num}, {type(iter_num)}")
#         # print(f"extra_iter_nums: {extra_iter_nums}, {type(extra_iter_nums[0])}")
#         # print(f"{extra_iter_nums, counts, block_num_left, extra_iter_num}")

#         # now consider the remaining iters for each running seq
#         iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

#         # print(f"{min(running_seqs[2][:running_seqs_num])}")

#         return iter_num


#     last_iter_seqs = list()
#     last_iter_seq_ids = list()
#     need_query_available_requests = True
#     must_record_first_step = False
#     tot_inference_time = 0
#     while unfinished_reqnum:

#         # before getting new prompts, query if there is newly available requests
#         TODO: 这个地方好像写错了，没有考虑到tot_inference_time只有在所有requests都结束但是当前时刻没有新的request的时候才会更新
#         if need_query_available_requests and (tot_iter_num % check_gap == 0):
#             seq_ids, inp_lens, out_lens, arrive_times, tot_inference_time = _check_new_input_requests(
#                 sort_input, seq_ids, inp_lens, out_lens, arrive_times, tot_inference_time, pointer, running_seqs_num)

#         # old_running_seqs_num = running_seqs_num
#         new_prompt_lens: List[int] = list()
#         new_prompt_ids: List[int] = list()
#         while (pointer < len(inp_lens)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
#             # we try to add new requests
#             # if token_num_stored + inp_lens[pointer] <= max_cache_slot_num:
#             if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
#                 running_seqs[0][running_seqs_num] = seq_ids[pointer] # pointer
#                 running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
#                 running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


#                 new_prompt_lens.append(inp_lens[pointer])
#                 new_prompt_ids.append(seq_ids[pointer])
#                 if running_seqs[2][running_seqs_num] == 0:
#                     # this seq is finished
#                     unfinished_reqnum -= 1
#                     pointer += 1
#                     continue


#                 # token_num_stored = token_num_stored + inp_lens[pointer]
#                 block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
#                 pointer += 1
#                 running_seqs_num += 1
#             else:
#                 break


#         # reset need_query_available_requests
#         TODO: 这个地方好像会死循环，如果当前时刻没有接收新的request的话-->已经fix
#         if (pointer == len(seq_ids)) \
#             and has_enough_cache(block_num_used,1,consider_watermark=True) \
#                 and (running_seqs_num < max_seq_num):
#             # if only there is not enough seqs but other resources are enough
#             need_query_available_requests = True
#         else:
#             need_query_available_requests = False
            

#         # update prefill logs
#         # new_prompt_lens = np.asarray(running_seqs[1][old_running_seqs_num:running_seqs_num])-1
#         # new_prompt_ids = running_seqs[0][old_running_seqs_num:running_seqs_num]
#         new_prompt_lens = np.asarray(new_prompt_lens)
#         new_prompt_ids = np.asarray(new_prompt_ids)
#         # if len(new_prompt_lens) > 0:
#         #     print(f"pointer: {pointer}, start new seqs: {new_prompt_lens}")
#         last_iter_seqs, last_iter_seq_ids, tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
#                             new_prompt_ids, infer_progress, tot_iter_num, 
#                             need_query_available_requests, check_gap, last_iter_seqs, last_iter_seq_ids, 
#                             must_record_first_step)
#         is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))

#         # we need first estimate the costs of all the new steps
#         tot_latency, prefill_latencys, decode_latencys = \
#             estimate_prefill_and_decode_cost_from_predicted_logs(
#                 prefill_logs=prefill_logs, decode_logs=logs, cost_table=cost_table, 
#                 model_name=model_name, 
#                 exec_plan=exec_plan, sample_config=sample_config, 
#                 trust_remote_code=trust_remote_code, revision=revision)
#         tot_inference_time += tot_latency
        

#         # we may need to go back to query newly available input requests
#         if len(last_iter_seqs) > 0:
#             must_record_first_step = True
#             continue
#         elif len(new_prompt_ids) > 0:
#             # already add at least 1 new step
#             must_record_first_step = False
#         # ------------prefill stage ends---------------------------------------------------------------
#         # 
#         # kill some running reqs if the cache is not enough
#         block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
#             kill_seqs_for_more_cache_space(
#                 running_seqs, 
#                 max_block_num, block_size, running_seqs_num, 
#                 inp_lens, out_lens, seq_ids, pointer)

#         # 
#         # collect decoding stage logs
#         # 1. compute the number of iters the current running reqs can run
#         # consider: available cache slots, seq remaining output tokens

#         iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
#         # compare iter_num with check_gap
#         if need_query_available_requests:
#             iter_num = min(iter_num, (tot_iter_num + must_record_first_step + check_gap - 1) // check_gap * check_gap \
#                 - tot_iter_num)
#             must_record_first_step = False
#             # iter_num must < check_gap
#             if iter_num == 0:
#                 must_record_first_step = True
#                 continue

#         # print(f"iter_num: {iter_num}")
#         # print(f"{running_seqs[0][:running_seqs_num].tolist()}")
#         # print(f"{running_seqs[1][:running_seqs_num].tolist()}")
#         # print(f"{running_seqs[2][:running_seqs_num].tolist()}")
#         tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
#         curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
#         logs.extend([(running_seqs_num, 
#                       tot_token_num + running_seqs_num*i,
#                       tot_token_num + running_seqs_num*i,
#                       curr_max_seqlen + i) \
#                      for i in range(iter_num)])
#         _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
#         is_prefill_steps.extend([False]*iter_num)
#         tot_iter_num += iter_num

#         # 
#         # 2. update the status of the running seqs
#         running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
#         running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

#         # remove finished reqs
#         running_seqs_num, unfinished_reqnum, block_num_used = \
#             remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)
        
#         # we need update tot_inference_time
#         tot_latency, prefill_latencys, decode_latencys = \
#             estimate_prefill_and_decode_cost_from_predicted_logs(
#                 prefill_logs=prefill_logs, decode_logs=logs, cost_table=cost_table, 
#                 model_name=model_name, 
#                 exec_plan=exec_plan, sample_config=sample_config, 
#                 trust_remote_code=trust_remote_code, revision=revision)
#         tot_inference_time += tot_latency

#         # 
#         # now go back to the top of the loop







def _fake_FCFS_schedule_continuous_model_level_pipeline(
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    '''
        Do the fake scheduling using the first-come-first-serve policy.
        inp_lens: the input lengths of the requests.
        out_lens: the output lengths of the requests.
        max_seq_num: the maximum number of requests running concurrently.
        max_cache_slot_num: the maximum number of tokens whose KV cache can be stored at the same time.
        cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}

        Output: [not only output fake scheduling logs, but also output latency]
            cumsum_latencys, is_prefill_steps, infer_progress

        There is only two constraints when trying to add a running request:
            (1) max_seq_num; (2) max_cache_slot_num (consider watermark=0.01).
        NOTE:
            (1) We ignore the block size here to make the fake schedule faster.
                --> it seems there will be a lot of request kill when block size is 1, 
                --> so we HAVE TO CONSIDER block size!
            (2) For prefill stage, we also consider  
                ``max_num_batched_tokens'' and TODO ``scheduler_config.max_paddings''. [Try this first]
            (3) When killing seqs, consider if there is an extra block for each sequence.

        NOTE: for continuous model-level pipeline, e.g., we may have model A -> model B, but A, B run in 
            the same execution stage.
            In this function, we will check whether there is new input requests every k (``check_gap'') inference steps,
            according to ``arrive_times''.
            If yes, we will add the new requests into the waiting list; 
            else, we do nothing but keep doing inference.
        
        This function runs K (i.e., check_gap) step fake scheduling starting from the given inference progress.
        NOTE:
            1. if before we finish the K inference steps we run out of requests, we stop the inference process 
            and turn to waiting more available input requests.
            2. in the current code, ``sort_input`` performs differently from the version that we must query every 
            check_gap steps.
    '''
    def has_enough_cache(block_num_used, new_token_num, consider_watermark=False):
        # return token_num_stored < max_cache_slot_num
        new_block_num = (new_token_num + block_size - 1) // block_size
        if consider_watermark:
            watermark_blocks = 0.01 * max_block_num
            return max_block_num - block_num_used - new_block_num >= watermark_blocks
        return (block_num_used + new_block_num) <= max_block_num
    def add_block_num_used(block_num_used, new_token_num):
        new_block_num = (new_token_num + block_size - 1) // block_size
        block_num_used = block_num_used + new_block_num
        return block_num_used
    def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
        # iter_num = (max_cache_slot_num - token_num_stored) // running_seqs_num
        # iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        # first compute how many blocks can be assigned to each running seq at most
        # print(f"in get_max_iter_num: block_num_used: {block_num_used}, running_seqs_num: {running_seqs_num}")
        iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
        # print(f"iter_num: {iter_num}")
        
        # then the running seqs cannot run >= 16 iters
        extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
        # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}")
        extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
        # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}, counts: {counts.tolist()}")
        block_num_left = (max_block_num - block_num_used) % running_seqs_num
        # print(f"in get_max_iter_num: block_num_left: {block_num_left}")
        # print(f"in get_max_iter_num: np.nonzero(np.cumsum(counts) > block_num_left): {np.nonzero(np.cumsum(counts) > block_num_left)}")
        extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
        iter_num += extra_iter_num
        # print(f"in get_max_iter_num: extra_iter_num: {extra_iter_num}")
        # print(f"iter_num: {iter_num}, {type(iter_num)}")
        # print(f"extra_iter_nums: {extra_iter_nums}, {type(extra_iter_nums[0])}")
        # print(f"{extra_iter_nums, counts, block_num_left, extra_iter_num}")

        # now consider the remaining iters for each running seq
        iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        # print(f"{min(running_seqs[2][:running_seqs_num])}")

        return iter_num
    def get_tot_token_num(running_seqs_num, running_seqs):
        return sum(running_seqs[1][:running_seqs_num])
    def get_max_seqlen(running_seqs_num, running_seqs):
        return max(running_seqs[1][:running_seqs_num])
    # 

    # convert input list to np arrays
    inp_lens = np.asarray(inp_lens)
    out_lens = np.asarray(out_lens)
    arrive_times = np.asarray(arrive_times)


    # TODO: remove this copy as it is only used for assertion
    ori_inplens = inp_lens.copy()
    ori_outlens = out_lens.copy()

    unfinished_reqnum = len(inp_lens)
    running_seqs = np.zeros((3, max_seq_num), dtype=np.int32) # three rows: index (i.e., seq_id), gened token num, remaining token num
    # seq_ids = np.asarray(list(range(len(inp_lens))))
    # seq_ids: initial available seq_ids is empty
    seq_ids = np.asarray(list(), dtype=np.int64)
    pointer = 0 # pointing to the next index of requests to consider
    running_seqs_num = 0
    # token_num_stored = 0 # for a seq, the number of token stored is (seq - 1)
    block_num_used = 0
    logs = list()
    prefill_logs = list() # each item is (seqnum, tot_token_num, attention_sum, max_seqlen)
    
    # store the inference progress of each sequence, for each seq, we store its continuous infer iter ranges
    infer_progress = list([] for _ in range(len(inp_lens)))
    # stores whether each step is a prefill step or not
    is_prefill_steps: List[bool] = list()
    tot_iter_num: int = 0 # the current total iteration number, == len(logs) + len(prefill_logs)

    # parameter to control checking newly available requests
    last_iter_seqs = list()
    last_iter_seq_ids = list()
    need_query_available_requests = True
    must_record_first_step = False
    tot_inference_time = 0

    # store the accumulated latency values we get
    cumsum_latencys: List[float] = np.asarray(list())



    while unfinished_reqnum:

        # TODO: 这个地方如果当前所有input都available了，也不需要再query了。
        # before getting new prompts, query if there is newly available requests
        # tot_inference_time is only updated when all requests finish but no new requests currently
        # print(f"at the beginning of the round: need_query_available_requests: {need_query_available_requests}")
        
        # if (running_seqs_num == 0) or \
        #     (need_query_available_requests and (tot_iter_num % check_gap == 0)):
        if ((running_seqs_num == 0) and (pointer==len(seq_ids))) or \
            (need_query_available_requests and (tot_iter_num % check_gap == 0)):

            # print(f"before query available requests, tot_inference_time: {tot_inference_time}")

            # print(f"Going to check new input requests. Conditions to determine checking:")
            # print(f"running_seqs_num: {running_seqs_num}")
            # print(f"need_query_available_requests: {need_query_available_requests}")
            # print(f"tot_iter_num: {tot_iter_num}")
            # print(f"len(seq_ids): {len(seq_ids)}")
            # print(f"pointer: {pointer}")

            seq_ids, inp_lens, out_lens, arrive_times, tot_inference_time = _check_new_input_requests(
                sort_input, seq_ids, inp_lens, out_lens, arrive_times, tot_inference_time, pointer, running_seqs_num)
        
            # print(f"_check_new_input_requests: new seq_ids: {seq_ids}")
            # print(f"arrive_times: {arrive_times}")
            # print(f"query available requests: ____________________")
            # print(f"seq_ids: {seq_ids}")
            # print(f"arrive_times: {arrive_times}")
            # print(f"tot_inference_time: {tot_inference_time}")
            # print(f"running_seqs: {running_seqs}")
            # print(f"running_seqs_num: {running_seqs_num}")
            # print(f"inp_lens: {inp_lens}")
            # print(f"out_lens: {out_lens}")
            # print(f"pointer: {pointer}")


        # old_running_seqs_num = running_seqs_num
        new_prompt_lens: List[int] = list()
        new_prompt_ids: List[int] = list()
        while (pointer < len(seq_ids)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
            # we try to add new requests
            # if token_num_stored + inp_lens[pointer] <= max_cache_slot_num:
            if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
                running_seqs[0][running_seqs_num] = seq_ids[pointer] # pointer
                running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
                running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


                new_prompt_lens.append(inp_lens[pointer])
                new_prompt_ids.append(seq_ids[pointer])
                if running_seqs[2][running_seqs_num] == 0:
                    # this seq is finished
                    unfinished_reqnum -= 1
                    pointer += 1
                    continue


                # token_num_stored = token_num_stored + inp_lens[pointer]
                block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
                pointer += 1
                running_seqs_num += 1
            else:
                break



        # reset need_query_available_requests
        if (len(seq_ids)<len(inp_lens)) and (pointer == len(seq_ids)) \
            and has_enough_cache(block_num_used,1,consider_watermark=True) \
                and (running_seqs_num < max_seq_num):
            # if there are unavailable reqs 
            # and available reqs are used up but there are remaining other resources
            need_query_available_requests = True
        else:
            need_query_available_requests = False


        # print(f"set need_query_available_requests conditions")
        # print(f"need_query_available_requests: {need_query_available_requests}")
        # print(f"len(seq_ids)<len(inp_lens): {len(seq_ids),len(inp_lens)}")
        # print(f"pointer == len(seq_ids): {pointer , len(seq_ids)}")
        # print(f"has_enough_cache(block_num_used,1,consider_watermark=True): {has_enough_cache(block_num_used,1,consider_watermark=True) }")
        # print(f"running_seqs_num < max_seq_num: {running_seqs_num, max_seq_num }")


        # update prefill logs
        # new_prompt_lens = np.asarray(running_seqs[1][old_running_seqs_num:running_seqs_num])-1
        # new_prompt_ids = running_seqs[0][old_running_seqs_num:running_seqs_num]
        new_prompt_lens = np.asarray(new_prompt_lens)
        new_prompt_ids = np.asarray(new_prompt_ids)
        ori_prefill_logs_num = len(prefill_logs)

        # print(f"new_prompt_ids: {new_prompt_ids}")
        # print(f"must_record_first_step: {must_record_first_step}, tot_iter_num: {tot_iter_num}, last_iter_seqs: {last_iter_seqs}, last_iter_seq_ids: {last_iter_seq_ids}")

        # if len(new_prompt_lens) > 0:
        #     print(f"pointer: {pointer}, start new seqs: {new_prompt_lens}")
        # tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
        #                     new_prompt_ids, infer_progress, tot_iter_num)
        # print(f"in _fake_FCFS_schedule_continuous_model_level_pipeline")
        last_iter_seqs, last_iter_seq_ids, tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
                            new_prompt_ids, infer_progress, tot_iter_num, 
                            need_query_available_requests, check_gap, last_iter_seqs, last_iter_seq_ids, 
                            must_record_first_step)
        is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))


        # print(f"prefill_logs: {prefill_logs}")
        # print(f"running_seqs_num: {running_seqs_num}")

        # we need first estimate the costs of all the new steps
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=prefill_logs[ori_prefill_logs_num:], decode_logs=list(), **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(prefill_latencys)+tot_inference_time))
        if len(cumsum_latencys) > 0:
            tot_inference_time = cumsum_latencys[-1]
        
        # assert (len(cumsum_latencys) == len(prefill_logs) + len(logs)) and (len(cumsum_latencys) == tot_iter_num)
        # print(f"len(prefill_logs): {len(prefill_logs)}, len(is_prefill_steps): {len(is_prefill_steps)}, ori_prefill_logs_num: {ori_prefill_logs_num}")
        # print(f"len(prefill_latencys): {len(prefill_latencys)}: {np.cumsum(prefill_latencys)}")
        # print(f"len(cumsum_latencys): {len(cumsum_latencys)}, len(infer_progress): {len(infer_progress)}, tot_iter_num: {tot_iter_num}, len(prefill_logs): {len(prefill_logs)}, len(logs): {len(logs)}")

        # we may need to go back to query newly available input requests
        if len(last_iter_seqs) > 0:
            must_record_first_step = True
            continue
        elif len(new_prompt_ids) > 0:
            # already add at least 1 new step
            must_record_first_step = False

        if running_seqs_num == 0:
            # no need to run the code below
            continue

        # ------------prefill stage ends---------------------------------------------------------------
        # 
        # kill some running reqs if the cache is not enough
        
        # print(f"tot_iter_num: {tot_iter_num}")
        # print(f"before kill_seqs_for_more_cache_space: ")
        # print(f"running_seqs: {running_seqs}")
        # print(f"running_seqs_num: {running_seqs_num}")

        block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
            kill_seqs_for_more_cache_space(
                running_seqs, 
                max_block_num, block_size, running_seqs_num, 
                inp_lens, out_lens, seq_ids, pointer)

        # print(f"after kill_seqs_for_more_cache_space: running_seqs_num: {running_seqs_num}")

        # 
        # collect decoding stage logs
        # 1. compute the number of iters the current running reqs can run
        # consider: available cache slots, seq remaining output tokens
        iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
        # print(f"iter_num: {iter_num}")
        

        # because of kill_seqs_for_more_cache_space, the ``need_query_available_requests`` value may need to be updated
        if pointer < len(seq_ids):
            need_query_available_requests = False

        # compare iter_num with check_gap
        if need_query_available_requests:
            iter_num = min(iter_num, (tot_iter_num + must_record_first_step + check_gap - 1) // check_gap * check_gap \
                - tot_iter_num)
            must_record_first_step = False
            # iter_num must < check_gap
            if iter_num == 0:
                must_record_first_step = True
                continue

        # print(f"tot_iter_num: {tot_iter_num}")
        # print(f"iter_num: {iter_num}")
        # print(f"{running_seqs[0][:running_seqs_num].tolist()}")
        # print(f"{running_seqs[1][:running_seqs_num].tolist()}")
        # print(f"{running_seqs[2][:running_seqs_num].tolist()}")
        tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
        curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
        logs.extend([(running_seqs_num, 
                      tot_token_num + running_seqs_num*i,
                      tot_token_num + running_seqs_num*i,
                      curr_max_seqlen + i) \
                     for i in range(iter_num)])
        _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
        is_prefill_steps.extend([False]*iter_num)
        tot_iter_num += iter_num

        # 
        # 2. update the status of the running seqs
        running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
        running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

        # remove finished reqs
        running_seqs_num, unfinished_reqnum, block_num_used = \
            remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)

        # we need update tot_inference_time
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=list(), decode_logs=logs[-iter_num:], **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(decode_latencys)+tot_inference_time))
        tot_inference_time = cumsum_latencys[-1]

        # assert (len(cumsum_latencys) == len(prefill_logs) + len(logs)) and (len(cumsum_latencys) == tot_iter_num)
        # print(f"len(decode_latencys): {len(decode_latencys)}: {np.cumsum(decode_latencys)}")
        # print(f"len(cumsum_latencys): {len(cumsum_latencys)}, len(infer_progress): {len(infer_progress)}, tot_iter_num: {tot_iter_num}, len(prefill_logs): {len(prefill_logs)}, len(logs): {len(logs)}")

        
        # print(f"unfinished_reqnum: {unfinished_reqnum}")

        # 
        # now go back to the top of the loop
    # here we finish the fake scheduling.
    # for i, step in enumerate(logs):
    #     print(f"step {i}: {step}")
    # for i, step in enumerate(prefill_logs):
    #     print(f"prefill step {i}: {step}")
    # 
    assert tot_iter_num == (len(logs) + len(prefill_logs)), (tot_iter_num, len(logs), len(prefill_logs), ori_inplens, ori_outlens, max_seq_num, max_block_num, max_num_batched_tokens, block_size) 
    # return logs, prefill_logs, is_prefill_steps, infer_progress
    return cumsum_latencys, is_prefill_steps, infer_progress













def _fake_FCFS_schedule_continuous_model_level_pipeline_vertical_fuse(
        # info of seqs whose arrive_times are known
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], ref_seq_ids: List[int],
        # 
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],        
        # 
        check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    '''
        Do the fake scheduling using the first-come-first-serve policy.
        inp_lens: the input lengths of the requests.
        out_lens: the output lengths of the requests.
        max_seq_num: the maximum number of requests running concurrently.
        max_cache_slot_num: the maximum number of tokens whose KV cache can be stored at the same time.
        cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}

        Output: [not only output fake scheduling logs, but also output latency]
            cumsum_latencys, is_prefill_steps, infer_progress

        There is only two constraints when trying to add a running request:
            (1) max_seq_num; (2) max_cache_slot_num (consider watermark=0.01).
        NOTE:
            (1) We ignore the block size here to make the fake schedule faster.
                --> it seems there will be a lot of request kill when block size is 1, 
                --> so we HAVE TO CONSIDER block size!
            (2) For prefill stage, we also consider  
                ``max_num_batched_tokens'' and TODO ``scheduler_config.max_paddings''. [Try this first]
            (3) When killing seqs, consider if there is an extra block for each sequence.

        NOTE: for continuous model-level pipeline, e.g., we may have model A -> model B, but A, B run in 
            the same execution stage.
            In this function, we will check whether there is new input requests every k (``check_gap'') inference steps,
            according to ``arrive_times''.
            If yes, we will add the new requests into the waiting list; 
            else, we do nothing but keep doing inference.
        
        This function runs K (i.e., check_gap) step fake scheduling starting from the given inference progress.
        NOTE:
            1. if before we finish the K inference steps we run out of requests, we stop the inference process 
            and turn to waiting more available input requests.
            2. in the current code, ``sort_input`` performs differently from the version that we must query every 
            check_gap steps.
            3. this version supports the vertical fusion of models.
    '''
    ''' ！我们假设关于是否对两个模型进行vertical fusion的操作在一开始就决定，把它当成一种计算图层面的优化。
    如果两个模型被vertically地fuse了，那对应的model info object 也发生了变化，这个model info object 会有不止一个inp list，
    每个inp list都来自被fuse的model，然后我们也要对应修改 inp_edge_dict 和 out_edge_dict。
    '''


    def has_enough_cache(block_num_used, new_token_num, consider_watermark=False):
        # return token_num_stored < max_cache_slot_num
        new_block_num = (new_token_num + block_size - 1) // block_size
        if consider_watermark:
            watermark_blocks = 0.01 * max_block_num
            return max_block_num - block_num_used - new_block_num >= watermark_blocks
        return (block_num_used + new_block_num) <= max_block_num
    def add_block_num_used(block_num_used, new_token_num):
        new_block_num = (new_token_num + block_size - 1) // block_size
        block_num_used = block_num_used + new_block_num
        return block_num_used
    def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
        # iter_num = (max_cache_slot_num - token_num_stored) // running_seqs_num
        # iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        # first compute how many blocks can be assigned to each running seq at most
        # print(f"in get_max_iter_num: block_num_used: {block_num_used}, running_seqs_num: {running_seqs_num}")
        iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
        # print(f"iter_num: {iter_num}")
        
        # then the running seqs cannot run >= 16 iters
        extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
        # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}")
        extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
        # print(f"in get_max_iter_num: extra_iter_nums: {extra_iter_nums.tolist()}, counts: {counts.tolist()}")
        block_num_left = (max_block_num - block_num_used) % running_seqs_num
        # print(f"in get_max_iter_num: block_num_left: {block_num_left}")
        # print(f"in get_max_iter_num: np.nonzero(np.cumsum(counts) > block_num_left): {np.nonzero(np.cumsum(counts) > block_num_left)}")
        extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
        iter_num += extra_iter_num
        # print(f"in get_max_iter_num: extra_iter_num: {extra_iter_num}")
        # print(f"iter_num: {iter_num}, {type(iter_num)}")
        # print(f"extra_iter_nums: {extra_iter_nums}, {type(extra_iter_nums[0])}")
        # print(f"{extra_iter_nums, counts, block_num_left, extra_iter_num}")

        # now consider the remaining iters for each running seq
        iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        # print(f"{min(running_seqs[2][:running_seqs_num])}")

        return iter_num
    def get_tot_token_num(running_seqs_num, running_seqs):
        return sum(running_seqs[1][:running_seqs_num])
    def get_max_seqlen(running_seqs_num, running_seqs):
        return max(running_seqs[1][:running_seqs_num])
    # 


    # convert input list to np arrays
    # copy the input information so that we can modify them
    # TODO: 这里需要用copy吗
    # inp_lens_list = np.asarray(inp_lens_list.copy())
    # out_lens_list = np.asarray(out_lens_list.copy())
    # arrive_times_list = np.asarray(arrive_times_list.copy())
    # fixed_ref_seq_ids_list = ref_seq_ids_list
    # ref_seq_ids_list = np.asarray(ref_seq_ids_list.copy(), dtype=np.int64)

    inp_lens_list = [np.asarray(_.copy()) for _ in inp_lens_list]
    out_lens_list = [np.asarray(_.copy()) for _ in out_lens_list]
    arrive_times_list = [np.asarray(_.copy()) for _ in arrive_times_list]
    fixed_ref_seq_ids_list = ref_seq_ids_list
    ref_seq_ids_list = [np.asarray(_.copy(), dtype=np.int64) for _ in ref_seq_ids_list]

    
    ref_seq_ids = np.asarray(ref_seq_ids, dtype=np.int64)
    inp_lens = np.asarray(inp_lens.copy())
    out_lens = np.asarray(out_lens.copy())
    arrive_times = np.asarray(arrive_times.copy())

    # print(f"ref_seq_ids: {ref_seq_ids}")
    # print(f"ref_seq_ids_list: {ref_seq_ids_list}")
    # print(f"arrive_times: {arrive_times}")
    # print(f"arrive_times_list: {arrive_times_list}")

    # currently, the first cand model is the model 1 (not model 0) of the fused models
    curr_model_level_id = 1

    # these variables stores the seqs whose arrive times has been known
    # inp_lens = inp_lens_list[0].copy()
    # out_lens = out_lens_list[0].copy()
    # arrive_times = arrive_times_list[0].copy()
    # ref_seq_ids = ref_seq_ids_list[0].copy()


    # TODO: remove this copy as it is only used for assertion
    ori_inplens = inp_lens.copy()
    ori_outlens = out_lens.copy()

    # unfinished_reqnum = len(inp_lens)
    # NOTE: here we regard the same req running in differet models as individual reqs, 
    # but for other metadata, there is only one set of data (like infer_progress) for the same seq
    req_nums = [len(inp_lens)]+[len(_inp_lens) for _inp_lens in inp_lens_list]
    unfinished_reqnum = sum(req_nums)
    # uniq_reqnum = len(inp_lens)
    running_seqs = np.zeros((3, max_seq_num), dtype=np.int32) # three rows: index (i.e., seq_id), gened token num, remaining token num
    # seq_ids = np.asarray(list(range(len(inp_lens))))
    # seq_ids: initial available seq_ids is empty
    seq_ids = np.asarray(list(), dtype=np.int64)
    pointer = 0 # pointing to the next index of requests to consider
    running_seqs_num = 0
    # token_num_stored = 0 # for a seq, the number of token stored is (seq - 1)
    block_num_used = 0
    logs = list()
    prefill_logs = list() # each item is (seqnum, tot_token_num, attention_sum, max_seqlen)
    
    # store the inference progress of each sequence, for each seq, we store its continuous infer iter ranges
    # infer_progress = list([] for _ in range(uniq_reqnum))
    full_infer_progress = list([[[] for _ in range(req_num)] for req_num in req_nums])
    # infer_progress = [full_infer_progress[0][_] for _ in range(req_nums[0])]
    # NOTE: change to dict here
    infer_progress = {seq_id:full_infer_progress[0][i] for i, seq_id in enumerate(ref_seq_ids)}
    # stores whether each step is a prefill step or not
    is_prefill_steps: List[bool] = list()
    tot_iter_num: int = 0 # the current total iteration number, == len(logs) + len(prefill_logs)

    # parameter to control checking newly available requests
    last_iter_seqs = list()
    last_iter_seq_ids = list()
    need_query_available_requests = True
    must_record_first_step = False
    tot_inference_time = 0

    # store the accumulated latency values we get
    cumsum_latencys: List[float] = np.asarray(list())

    # NOTE: record the finished seq ids
    finished_seq_ids = list()

    while unfinished_reqnum:

        # TODO: 这个地方如果当前所有input都available了，也不需要再query了。
        # before getting new prompts, query if there is newly available requests
        # tot_inference_time is only updated when all requests finish but no new requests currently
        # print(f"at the beginning of the round: need_query_available_requests: {need_query_available_requests}")
        
        # if (running_seqs_num == 0) or \
        #     (need_query_available_requests and (tot_iter_num % check_gap == 0)):
        if ((running_seqs_num == 0) and (pointer==len(seq_ids))) or \
            (need_query_available_requests and (tot_iter_num % check_gap == 0)):

            # print(f"before query available requests, tot_inference_time: {tot_inference_time}")

            # print(f"Going to check new input requests. Conditions to determine checking:")
            # print(f"running_seqs_num: {running_seqs_num}")
            # print(f"need_query_available_requests: {need_query_available_requests}")
            # print(f"tot_iter_num: {tot_iter_num}")
            # print(f"len(seq_ids): {len(seq_ids)}")
            # print(f"pointer: {pointer}")
            
            
            # we first check whether there are seqs whose arrive times become known
            ref_seq_ids, inp_lens, out_lens, arrive_times, \
                ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list= \
                    _update_seq_info_with_known_arrive_time(
                        tot_inference_time, running_seqs[0][:running_seqs_num], pointer,
                        ref_seq_ids, inp_lens, out_lens, arrive_times,
                        ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list,
                        infer_progress, full_infer_progress, fixed_ref_seq_ids_list
                        )
            finished_seq_ids = list()

            # print(f"ref_seq_ids: {ref_seq_ids}")

            seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, tot_inference_time = \
                _check_new_input_requests_support_vertical_fuse(
                    sort_input, seq_ids, ref_seq_ids, inp_lens, out_lens, arrive_times, 
                    tot_inference_time, pointer, running_seqs_num)
            
            # print(f"seq_ids: {seq_ids}")
        
            # print(f"_check_new_input_requests: new seq_ids: {seq_ids}")
            # print(f"arrive_times: {arrive_times}")
            # print(f"query available requests: ____________________")
            # print(f"seq_ids: {seq_ids}")
            # print(f"arrive_times: {arrive_times}")
            # print(f"tot_inference_time: {tot_inference_time}")
            # print(f"running_seqs: {running_seqs}")
            # print(f"running_seqs_num: {running_seqs_num}")
            # print(f"inp_lens: {inp_lens}")
            # print(f"out_lens: {out_lens}")
            # print(f"pointer: {pointer}")


        # old_running_seqs_num = running_seqs_num
        new_prompt_lens: List[int] = list()
        new_prompt_ids: List[int] = list()
        while (pointer < len(seq_ids)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
            # we try to add new requests
            # if token_num_stored + inp_lens[pointer] <= max_cache_slot_num:
            if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
                running_seqs[0][running_seqs_num] = seq_ids[pointer] # pointer
                running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
                running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


                new_prompt_lens.append(inp_lens[pointer])
                new_prompt_ids.append(seq_ids[pointer])
                if running_seqs[2][running_seqs_num] == 0:
                    # this seq is finished
                    unfinished_reqnum -= 1
                    pointer += 1
                    continue


                # token_num_stored = token_num_stored + inp_lens[pointer]
                block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
                pointer += 1
                running_seqs_num += 1
            else:
                break



        # reset need_query_available_requests
        # TODO: 这个条件要改
        # if (len(seq_ids)<len(inp_lens)) \
        if (len(seq_ids)<sum(req_nums)) \
            and (pointer == len(seq_ids)) \
            and has_enough_cache(block_num_used,1,consider_watermark=True) \
                and (running_seqs_num < max_seq_num):
            # if there are unavailable reqs 
            # and available reqs are used up but there are remaining other resources
            need_query_available_requests = True
        else:
            need_query_available_requests = False


        # print(f"set need_query_available_requests conditions")
        # print(f"need_query_available_requests: {need_query_available_requests}")
        # print(f"len(seq_ids)<len(inp_lens): {len(seq_ids),len(inp_lens)}")
        # print(f"pointer == len(seq_ids): {pointer , len(seq_ids)}")
        # print(f"has_enough_cache(block_num_used,1,consider_watermark=True): {has_enough_cache(block_num_used,1,consider_watermark=True) }")
        # print(f"running_seqs_num < max_seq_num: {running_seqs_num, max_seq_num }")


        # update prefill logs
        # new_prompt_lens = np.asarray(running_seqs[1][old_running_seqs_num:running_seqs_num])-1
        # new_prompt_ids = running_seqs[0][old_running_seqs_num:running_seqs_num]
        new_prompt_lens = np.asarray(new_prompt_lens)
        new_prompt_ids = np.asarray(new_prompt_ids)
        ori_prefill_logs_num = len(prefill_logs)

        # print(f"new_prompt_ids: {new_prompt_ids}")
        # print(f"must_record_first_step: {must_record_first_step}, tot_iter_num: {tot_iter_num}, last_iter_seqs: {last_iter_seqs}, last_iter_seq_ids: {last_iter_seq_ids}")

        # if len(new_prompt_lens) > 0:
        #     print(f"pointer: {pointer}, start new seqs: {new_prompt_lens}")
        # tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
        #                     new_prompt_ids, infer_progress, tot_iter_num)
        # print(f"in _fake_FCFS_schedule_continuous_model_level_pipeline")
        ori_last_iter_seq_ids = last_iter_seq_ids.copy()
        last_iter_seqs, last_iter_seq_ids, tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
                            new_prompt_ids, infer_progress, tot_iter_num, 
                            need_query_available_requests, check_gap, last_iter_seqs, last_iter_seq_ids, 
                            must_record_first_step)
        is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))


        # get the seqs which finishes after the prefill stage
        run_prompt_ids = np.setdiff1d(
            np.concatenate((new_prompt_ids, ori_last_iter_seq_ids)), last_iter_seq_ids, 
            assume_unique=True)
        finished_seq_ids.extend(
            np.setdiff1d(run_prompt_ids, running_seqs[0][:running_seqs_num], assume_unique=True))
        


        # print(f"new prefill_logs: {prefill_logs[ori_prefill_logs_num:]}")
        # print(f"running_seqs_num: {running_seqs_num}")

        # we need first estimate the costs of all the new steps
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=prefill_logs[ori_prefill_logs_num:], decode_logs=list(), **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(prefill_latencys)+tot_inference_time))
        if len(cumsum_latencys) > 0:
            tot_inference_time = cumsum_latencys[-1]
        
        # assert (len(cumsum_latencys) == len(prefill_logs) + len(logs)) and (len(cumsum_latencys) == tot_iter_num)
        # print(f"len(prefill_logs): {len(prefill_logs)}, len(is_prefill_steps): {len(is_prefill_steps)}, ori_prefill_logs_num: {ori_prefill_logs_num}")
        # print(f"len(prefill_latencys): {len(prefill_latencys)}: {np.cumsum(prefill_latencys)}")
        # print(f"len(cumsum_latencys): {len(cumsum_latencys)}, len(infer_progress): {len(infer_progress)}, tot_iter_num: {tot_iter_num}, len(prefill_logs): {len(prefill_logs)}, len(logs): {len(logs)}")

        # we may need to go back to query newly available input requests
        if len(last_iter_seqs) > 0:
            must_record_first_step = True
            # print(f"go back to check inps\n")
            continue
        elif len(new_prompt_ids) > 0:
            # already add at least 1 new step
            must_record_first_step = False

        if running_seqs_num == 0:
            # no need to run the code below
            # print(f"go back to check inps\n")
            continue

        # ------------prefill stage ends---------------------------------------------------------------
        # 
        # kill some running reqs if the cache is not enough
        
        # print(f"tot_iter_num: {tot_iter_num}")
        # print(f"before kill_seqs_for_more_cache_space: ")
        # print(f"running_seqs: {running_seqs}")
        # print(f"running_seqs_num: {running_seqs_num}")

        block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
            kill_seqs_for_more_cache_space(
                running_seqs, 
                max_block_num, block_size, running_seqs_num, 
                inp_lens, out_lens, seq_ids, pointer)

        # print(f"after kill_seqs_for_more_cache_space: running_seqs_num: {running_seqs_num}")

        # 
        # collect decoding stage logs
        # 1. compute the number of iters the current running reqs can run
        # consider: available cache slots, seq remaining output tokens
        iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
        # print(f"iter_num: {iter_num}")
        

        # because of kill_seqs_for_more_cache_space, the ``need_query_available_requests`` value may need to be updated
        if pointer < len(seq_ids):
            need_query_available_requests = False

        # compare iter_num with check_gap
        if need_query_available_requests:
            iter_num = min(iter_num, (tot_iter_num + must_record_first_step + check_gap - 1) // check_gap * check_gap \
                - tot_iter_num)
            must_record_first_step = False
            # iter_num must < check_gap
            if iter_num == 0:
                must_record_first_step = True
                # print(f"go back to check inps --decoding 0 step\n")
                continue

        # print(f"tot_iter_num: {tot_iter_num}")
        # print(f"iter_num: {iter_num}")
        # print(f"{running_seqs[0][:running_seqs_num].tolist()}")
        # print(f"{running_seqs[1][:running_seqs_num].tolist()}")
        # print(f"{running_seqs[2][:running_seqs_num].tolist()}")
        tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
        curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
        logs.extend([(running_seqs_num, 
                      tot_token_num + running_seqs_num*i,
                      tot_token_num + running_seqs_num*i,
                      curr_max_seqlen + i) \
                     for i in range(iter_num)])
        _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
        is_prefill_steps.extend([False]*iter_num)
        tot_iter_num += iter_num

        # print(f"new decode logs: {logs[-iter_num:]}")

        # 
        # 2. update the status of the running seqs
        running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
        running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

        # get the finished seq ids
        finished_seq_ids.extend(running_seqs[0][:running_seqs_num][running_seqs[2][:running_seqs_num]==0])

        # remove finished reqs
        running_seqs_num, unfinished_reqnum, block_num_used = \
            remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)

        # we need update tot_inference_time
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=list(), decode_logs=logs[-iter_num:], **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(decode_latencys)+tot_inference_time))
        tot_inference_time = cumsum_latencys[-1]

        # assert (len(cumsum_latencys) == len(prefill_logs) + len(logs)) and (len(cumsum_latencys) == tot_iter_num)
        # print(f"len(decode_latencys): {len(decode_latencys)}: {np.cumsum(decode_latencys)}")
        # print(f"len(cumsum_latencys): {len(cumsum_latencys)}, len(infer_progress): {len(infer_progress)}, tot_iter_num: {tot_iter_num}, len(prefill_logs): {len(prefill_logs)}, len(logs): {len(logs)}")

        
        # print(f"unfinished_reqnum: {unfinished_reqnum}")

        # 
        # now go back to the top of the loop
    # here we finish the fake scheduling.
    # for i, step in enumerate(logs):
    #     print(f"step {i}: {step}")
    # for i, step in enumerate(prefill_logs):
    #     print(f"prefill step {i}: {step}")
    # 
    assert tot_iter_num == (len(logs) + len(prefill_logs)), (tot_iter_num, len(logs), len(prefill_logs), ori_inplens, ori_outlens, max_seq_num, max_block_num, max_num_batched_tokens, block_size) 
    # return logs, prefill_logs, is_prefill_steps, infer_progress
    return cumsum_latencys, is_prefill_steps, full_infer_progress
















def get_finish_times(cumsum_latencys: List[float], infer_progress: List[List[int]]):
    """
        We use a list to store the continuous inference iteration ranges for each sequence.
            E.g., 
                [start1, end1, start2, end2, ...] --> for iter_i with 
                    start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
                the seq attend the corresponding iteration steps.
    """
    # print(f"get_finish_times(): infer_progress: {infer_progress}")
    # print(f"len(cumsum_latencys): {len(cumsum_latencys)}, len(infer_progress): {len(infer_progress)}")
    last_iters = [rng[-1] for rng in infer_progress]
    finish_times = cumsum_latencys[last_iters]
    return finish_times



def get_finish_times_from_rng_infos(
        cumsum_latencys: List[float], 
        cum_rng_nums: List[int], rng_ends: List[int],
        ):
    """
        We use a list to store the continuous inference iteration ranges for each sequence.
            E.g., 
                [start1, end1, start2, end2, ...] --> for iter_i with 
                    start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
                the seq attend the corresponding iteration steps.
    """
    last_iters = rng_ends[cum_rng_nums[1:] - 1]
    finish_times = cumsum_latencys[last_iters]
    return finish_times





def fake_FCFS_schedule(
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    """
        This function calls ``_fake_FCFS_schedule_NO_continuous_model_level_pipeline`` or
        ``_fake_FCFS_schedule_continuous_model_level_pipeline`` depending on whether arrive_times is empty.

        Input:
            cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}
        
        Output: 
            cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times

        NOTE: finish_times: the finish times of each request.
    """

    if len(inp_lens) == 0:
        return [], [0], [], [], [], []

    if len(arrive_times) == 0:

        if sort_input:
            # we need to sort the input requests
            inp_lens = np.asarray(inp_lens)
            out_lens = np.asarray(out_lens)
            order = np.argsort(-inp_lens, kind='stable')
            inp_lens = inp_lens[order]
            out_lens = out_lens[order]

        decode_logs, prefill_logs, is_prefill_steps, infer_progress =  \
            _fake_FCFS_schedule_NO_continuous_model_level_pipeline(
                inp_lens=inp_lens, out_lens=out_lens,
                max_seq_num=max_seq_num, max_block_num=max_block_num, max_num_batched_tokens=max_num_batched_tokens,
                block_size=block_size)
        
        # estimate total latency
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=prefill_logs, decode_logs=decode_logs, **cost_estimate_args)

        # get cumulative latencys
        (cumsum_latencys, cum_rng_nums, rng_starts, rng_ends) = \
            get_cumLatency_inferRng_info(
                    decode_latencys, prefill_latencys, 
                    is_prefill_steps, infer_progress)

        # get the finish time of each request in order
        finish_times = get_finish_times(cumsum_latencys, infer_progress)

        return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times
    else:

        print(f"Has input exec plans in this stage")
        # print(f"inp_lens: {inp_lens}")
        # print(f"out_lens: {out_lens}")
        # print(f"arrive_times: {arrive_times}")

        cumsum_latencys, is_prefill_steps, infer_progress = _fake_FCFS_schedule_continuous_model_level_pipeline(
            inp_lens=inp_lens,out_lens=out_lens, arrive_times=arrive_times, check_gap=check_gap,
            max_seq_num=max_seq_num, max_block_num=max_block_num, max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size, sort_input=sort_input, cost_estimate_args=cost_estimate_args)

        # print(f"infer_progress: {infer_progress}")

        # compute cum_rng_nums, rng_starts, rng_ends
        cum_rng_nums, rng_starts, rng_ends = _get_inferRng_info(infer_progress)

        # get the finish time of each request in order
        finish_times = get_finish_times(cumsum_latencys, infer_progress)

        return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times










def fake_FCFS_schedule_vertical_fuse(
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], ref_seq_ids: List[int],
        # 
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],        
        # 
        check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    """
        This function calls ``_fake_FCFS_schedule_NO_continuous_model_level_pipeline`` or
        ``_fake_FCFS_schedule_continuous_model_level_pipeline`` depending on whether arrive_times is empty.

        Input:
            cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}
        
        Output: 
            cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times

        NOTE: 
            1. finish_times: the finish times of each request.
            2. support the vertical fusion of models.
    """

    # print(f"in fake_FCFS_schedule_vertical_fuse")
    # print(f"inp_lens: {inp_lens}")
    # print(f"out_lens: {out_lens}")
    # print(f"arrive_times: {arrive_times}")

    # print(f"inp_lens_list: {inp_lens_list}")
    # print(f"out_lens_list: {out_lens_list}")
    # print(f"arrive_times_list: {arrive_times_list}")

    # NOTE: here the condition should be considering all inp len list available
    # if len(inp_lens) == 0:
    #    return [], [0], [], [], [], []
    if (len(inp_lens) + sum([len(_) for _ in inp_lens_list])) == 0:
        model_num = 1+len(inp_lens_list)
        return [], [[0] for _ in range(model_num)], [[] for _ in range(model_num)], [[] for _ in range(model_num)], [], [[] for _ in range(model_num)]

    if len(arrive_times) == 0:
        arrive_times = np.asarray([-1]*len(inp_lens))
    
    for i in range(len(arrive_times_list)):
        if len(arrive_times_list[i]) == 0:
            arrive_times_list[i] = np.asarray([-1]*len(inp_lens_list[i]))

    # print(f"Has input exec plans in this stage")
    # print(f"inp_lens: {inp_lens}")
    # print(f"out_lens: {out_lens}")
    # print(f"arrive_times: {arrive_times}")

    cumsum_latencys, is_prefill_steps, full_infer_progress = \
        _fake_FCFS_schedule_continuous_model_level_pipeline_vertical_fuse(
            # info of seqs whose arrive_times are known
            inp_lens, out_lens, arrive_times, ref_seq_ids,
            # 
            ref_seq_ids_list,
            inp_lens_list,
            out_lens_list,
            arrive_times_list,      
            # 
            check_gap,
            max_seq_num, max_block_num, max_num_batched_tokens,
            block_size,
            sort_input,
            cost_estimate_args,
            )

    # print(f"full_infer_progress: {full_infer_progress}")

    cum_rng_nums_list, rng_starts_list, rng_ends_list, finish_times_list = list(), list(), list(), list()
    for infer_progress in full_infer_progress:
        if len(infer_progress) == 0:
            cum_rng_nums_list.append(np.asarray([0], dtype=np.int64))
            rng_starts_list.append(np.asarray([], dtype=np.int64))
            rng_ends_list.append(np.asarray([], dtype=np.int64))
            finish_times_list.append(np.asarray([]))
            continue

        # compute cum_rng_nums, rng_starts, rng_ends
        cum_rng_nums, rng_starts, rng_ends = _get_inferRng_info(infer_progress)
        cum_rng_nums_list.append(cum_rng_nums)
        rng_starts_list.append(rng_starts)
        rng_ends_list.append(rng_ends)

        # get the finish time of each request in order
        finish_times = get_finish_times(cumsum_latencys, infer_progress)
        finish_times_list.append(finish_times)

    return cumsum_latencys, cum_rng_nums_list, rng_starts_list, rng_ends_list, is_prefill_steps, finish_times_list













def _update_fake_FCFS_schedule_metadata(
        old_inp_lens: List[int], # new_inp_lens: List[int],
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int],
        is_prefill_steps: List[bool],
        max_num_batched_tokens: int, stop_iter_i: int,
        cost_table: CostTable, 
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Compute the fake FCFS scheduling metadata restart from the iter ``stop_iter_i+1'' based on the given metadata.
        NOTE: we restart all running seqs at iter ``stop_iter_i'' and not finished after iter ``stop_iter_i'' ends.
        We use prefill steps to recover their seq lens after iter ``stop_iter_i'' ends.
            (1) We do not consider the ``watermark'' constraint in the prefill stage.
        NOTE:
            this function assumes there are running seqs after iter ``stop_iter_i'' ends.
    '''

    # 1. get the seqs which have not been finished or killed after stop_iter_i
    # print(stop_iter_i, rng_starts.tolist())
    # print(rng_ends.tolist())
    # print(f"cum_rng_nums: {cum_rng_nums.tolist()}")
    # print(f"~"*50)
    # print(f"stop_iter_i: {stop_iter_i}, len(cumsum_latencys): {len(cumsum_latencys)}")

    # get the alive ranges information
    new_rng_starts = rng_starts - (stop_iter_i + 1)
    new_rng_ends = rng_ends - (stop_iter_i + 1)
    # delete the ranges finished before (stop_iter_i+1)
    cum_alive_rng_nums = np.cumsum(np.concatenate(([0], (new_rng_ends >= 0))))
    alive_rng_nums = (cum_alive_rng_nums[cum_rng_nums[1:]] - cum_alive_rng_nums[cum_rng_nums[:-1]])
    # alive_rng_nums = alive_rng_nums[alive_rng_nums > 0]
    alive_cum_rng_nums = np.cumsum(np.concatenate(([0], alive_rng_nums)))
    
    new_rng_starts = np.maximum(new_rng_starts[new_rng_ends >= 0], 0)
    new_rng_ends = new_rng_ends[new_rng_ends >= 0]

    # print(f"new_rng_starts: {new_rng_starts}")
    # print(f"new_rng_ends: {new_rng_ends}")
    # print(f"alive_rng_nums: {alive_rng_nums}")
    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums}")
    # print(f"is_prefill_steps: {is_prefill_steps}")

    # we need to recover the seqs whose first range start is a decoding one originally
    start_from_prefillstep = np.asarray(is_prefill_steps)[new_rng_starts[alive_cum_rng_nums[:-1][alive_rng_nums>0]] + (stop_iter_i+1)]
    
    # print(f"alive_cum_rng_nums[:-1][alive_rng_nums>0]:{alive_cum_rng_nums[:-1][alive_rng_nums>0]}")
    # print(f"new_rng_starts[alive_cum_rng_nums[:-1][alive_rng_nums>0]]: {new_rng_starts[alive_cum_rng_nums[:-1][alive_rng_nums>0]]}")
    # print(f"start_from_prefillstep: {start_from_prefillstep}")
    
    running_seq_ids = np.nonzero(alive_rng_nums>0)[0][ np.nonzero(start_from_prefillstep==False)[0] ]

    # print(f"cumsum_latencys: {cumsum_latencys.tolist()}")
    # cum_in_rngs = np.cumsum(np.concatenate(([0], ((stop_iter_i>=rng_starts) * (stop_iter_i < rng_ends)))))
    # running_seq_ids = np.nonzero((cum_in_rngs[cum_rng_nums[1:]] - cum_in_rngs[cum_rng_nums[:-1]]) > 0)[0]

    # print(f"running_seq_ids: {running_seq_ids}")


    prefill_logs = list() # each item is (seqnum, tot_token_num, attention_sum, max_seqlen)    
    # store the inference progress of each sequence, for each seq, we store its continuous infer iter ranges
    infer_progress = list([] for _ in range(len(old_inp_lens)))
    # stores whether each step is a prefill step or not
    tot_iter_num: int = 0 # the current total iteration number, == len(logs) + len(prefill_logs)


    # 2. now we comp the prefill steps to recover these running seqs
    # make use of the FCFS policy
    finished_lens = np.cumsum(np.concatenate(
        ([0], ((stop_iter_i >= rng_starts) * (np.minimum(stop_iter_i, rng_ends) - rng_starts + 1)))
        ))
    finished_lens = finished_lens[cum_rng_nums[1:]] - finished_lens[cum_rng_nums[:-1]]

    # print(f"finished_lens: {finished_lens}")
    # print(f"old_inp_lens: {old_inp_lens}")
    # print(f"new_inp_lens: {new_inp_lens}")

    # new_prompt_lens = np.asarray(old_inp_lens)[running_seq_ids] + finished_lens[running_seq_ids] - 1
    # NOTE: we directly start from the new inp lens of the running seqs here
    new_prompt_lens = np.asarray(old_inp_lens)[running_seq_ids] + finished_lens[running_seq_ids]
    # new_prompt_ids = np.asarray(range(len(running_seq_ids)), dtype=int)
    new_prompt_ids = running_seq_ids
    # assert (np.asarray(new_inp_lens)[:len(running_seq_ids)] == (new_prompt_lens + 1)).all(), (new_inp_lens[:len(running_seq_ids)], (new_prompt_lens + 1))
    # TODO: change the computation method of new_prompt_lens
    # print(f"in _update_fake_FCFS_schedule_metadata")
    _, _, tot_iter_num = update_prefill_logs(prefill_logs, new_prompt_lens, max_num_batched_tokens,
                        new_prompt_ids, infer_progress, tot_iter_num,
                        need_query_available_requests=False, check_gap=1,
                        last_iter_seqs=list(), last_iter_seq_ids=list(),
                        must_record_first_step=False)

    tmp_is_prefill_steps = np.concatenate(([True]*len(prefill_logs), is_prefill_steps[stop_iter_i+1:]))
    # print(f"tot_iter_num: {tot_iter_num}, infer_progress: {infer_progress}, prefill_logs: {prefill_logs}")

    # 3. generate the new infer metadata
    # we do not need to update ``logs, prefill_logs, is_prefill_steps, infer_progress'' 
    # as they are only used to generate other cached infer metadata
    
    # 3.1 get new latencys
    prefill_latencys = estimate_cost_from_predicted_logs(
        prefill_logs, cost_table=cost_table, is_prompt=True,
        model_name=model_name, exec_plan=exec_plan, sample_config=sample_config, 
        trust_remote_code=trust_remote_code, revision=revision)
    tmp_cum_latencys = cumsum_latencys[stop_iter_i+1:] - cumsum_latencys[stop_iter_i]
    cum_prefill_latencys = np.cumsum(prefill_latencys)
    tmp_cum_latencys = np.concatenate((cum_prefill_latencys, (sum(prefill_latencys)+tmp_cum_latencys)))

    # print(f"_update_fake_FCFS_schedule_metadata: tmp_is_prefill_steps: {tmp_is_prefill_steps}")
    # print(f"_update_fake_FCFS_schedule_metadata: tmp_cum_latencys: {tmp_cum_latencys}")

    # 3.2 get new range infor
    # new_rng_starts = rng_starts - (stop_iter_i + 1)
    # new_rng_ends = rng_ends - (stop_iter_i + 1)
    # # delete the ranges finished before (stop_iter_i+1)
    # cum_alive_rng_nums = np.cumsum(np.concatenate(([0], (new_rng_ends >= 0))))
    # alive_rng_nums = (cum_alive_rng_nums[cum_rng_nums[1:]] - cum_alive_rng_nums[cum_rng_nums[:-1]])
    # # alive_rng_nums = alive_rng_nums[alive_rng_nums > 0]
    # alive_cum_rng_nums = np.cumsum(np.concatenate(([0], alive_rng_nums)))
    
    # new_rng_starts = np.maximum(new_rng_starts[new_rng_ends >= 0], 0)
    # new_rng_ends = new_rng_ends[new_rng_ends >= 0]

    # delete the running seqs from iter ``stop_iter_i+1''
    # NOTE: we did not delete the related latency of iter ``stop_iter_i+1'' in tmp_cum_latencys (approximation here)
    # NOTE: -> delete running seqs from the first decode step after ``stop_iter_i'', may not be ``stop_iter_i+1''
    # running_seq_first_rng_ids = alive_cum_rng_nums[:len(running_seq_ids)]
    running_seq_first_rng_ids = alive_cum_rng_nums[running_seq_ids]
    first_decode_iter_is = new_rng_starts[running_seq_first_rng_ids].copy()
    new_rng_starts[running_seq_first_rng_ids] += 1
    # reduce_cum_rng_nums = \
    #     np.cumsum((new_rng_starts[running_seq_first_rng_ids] > new_rng_ends[running_seq_first_rng_ids]))
    # alive_cum_rng_nums[1:1+len(running_seq_ids)] -= reduce_cum_rng_nums
    invalid_cum_rng_nums = np.cumsum(np.concatenate(([0], new_rng_starts > new_rng_ends)))
    reduce_rng_nums = invalid_cum_rng_nums[alive_cum_rng_nums[1:]] - invalid_cum_rng_nums[alive_cum_rng_nums[:-1]]
    alive_cum_rng_nums[1:] -= np.cumsum(reduce_rng_nums)

    keep_rng_ids = (new_rng_starts <= new_rng_ends)
    new_rng_starts = new_rng_starts[keep_rng_ids]
    new_rng_ends = new_rng_ends[keep_rng_ids]

    # also need to update new_prompt_ids as some running seqs may be finished after the recovered prefill steps
    # new_prompt_ids = new_prompt_ids[np.diff(alive_cum_rng_nums)[new_prompt_ids] > 0]


    assert len(new_rng_starts) == alive_cum_rng_nums[-1]

    # print(f"after delete running seqs from first decode iter")
    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums}, new_prompt_ids: {new_prompt_ids, new_prompt_ids.dtype}")
    # print(f"new_rng_starts: {new_rng_starts}")
    # print(f"new_rng_ends: {new_rng_ends}")
    # print(f"alive_cum_rng_nums[new_prompt_ids]: {alive_cum_rng_nums[new_prompt_ids]}")
    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums}")


    # TODO: if no seqs run on the first decode iter after  ``stop_iter_i'', delete this iter
    # first_decode_iter_is = first_decode_iter_is.reshape((-1, 1))
    # cum_seq_nums_for_each_first_decode_iter_is = np.cumsum(np.concatenate((np.zeros_like(first_decode_iter_is), 
    #                  (first_decode_iter_is>=new_rng_starts)*(first_decode_iter_is<=new_rng_ends)), axis=1), axis=1)
    first_decode_iter_is = np.unique(first_decode_iter_is)
    # print(f"first_decode_iter_is: {first_decode_iter_is}")
    # assert len(first_decode_iter_is) <= 1
    for first_decode_iter_i in first_decode_iter_is:
        # first_decode_iter_i=first_decode_iter_is[0]
        if not ((first_decode_iter_i>=new_rng_starts)*(first_decode_iter_i<=new_rng_ends)).any():
            # no seqs running in this iter

            # print(f"first_decode_iter_i: {first_decode_iter_i}")
            # print(f"first_decode_iter_i+len(prefill_latencys): {first_decode_iter_i+len(prefill_latencys)}")

            new_rng_starts[new_rng_starts>=first_decode_iter_i] -= 1
            new_rng_ends[new_rng_ends>=first_decode_iter_i] -= 1
            
            tmp_cum_latencys[first_decode_iter_i+len(prefill_latencys):-1] = \
                tmp_cum_latencys[first_decode_iter_i+len(prefill_latencys)+1:]
            tmp_cum_latencys = tmp_cum_latencys[:-1]
            
            tmp_is_prefill_steps[first_decode_iter_i+len(prefill_latencys):-1] = \
                tmp_is_prefill_steps[first_decode_iter_i+len(prefill_latencys)+1:]
            tmp_is_prefill_steps = tmp_is_prefill_steps[:-1]

            # print(f"tmp_is_prefill_steps: {tmp_is_prefill_steps}")


    # print(f"after remove empty iter")
    # # print(f"update rng starts: {new_rng_starts}  rng ends: {new_rng_ends}")
    # # print(f"alive_rng_nums: {alive_rng_nums}")
    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums}, new_prompt_ids: {new_prompt_ids, new_prompt_ids.dtype}")
    # print(f"alive_cum_rng_nums[new_prompt_ids]: {alive_cum_rng_nums[new_prompt_ids]}")
    # print(f"new_rng_starts: {new_rng_starts}")
    # print(f"new_rng_ends: {new_rng_ends}")
    
    # concate the new prefill ranges
    new_rng_starts += len(prefill_logs)
    new_rng_ends += len(prefill_logs)

    # print(f"len(prefill_logs): {len(prefill_logs)}, add offset rng starts: {new_rng_starts}  rng ends: {new_rng_ends}")

    
    need_add_rng = np.full((len(old_inp_lens)), False)
    need_add_rng[new_prompt_ids] = True

    alive_prompt_ids = new_prompt_ids[np.diff(alive_cum_rng_nums)[new_prompt_ids]>0]
    alive_prompt_first_rng_ids = alive_cum_rng_nums[new_prompt_ids][np.diff(alive_cum_rng_nums)[new_prompt_ids]>0]
    prefill_rng_ends = np.asarray([infer_progress[seq_i][1] for seq_i in alive_prompt_ids])
    
    # print(f"after add offset")
    # print(f"new_rng_starts: {new_rng_starts}")
    # print(f"new_rng_ends: {new_rng_ends}")
    # print(f"prefill_rng_ends: {prefill_rng_ends}")
    # print(f"alive_prompt_first_rng_ids: {alive_prompt_first_rng_ids}")


    need_add_rng[alive_prompt_ids] = new_rng_starts[alive_prompt_first_rng_ids] > (prefill_rng_ends + 1)
    old_alive_cum_rng_nums = alive_cum_rng_nums.copy()
    # alive_cum_rng_nums[1:1+len(new_prompt_ids)] += np.cumsum(need_add_rng)
    alive_cum_rng_nums[1:] += np.cumsum(need_add_rng)

    tmp_rng_starts = np.empty([len(new_rng_starts)+sum(need_add_rng)], dtype=new_rng_starts.dtype)
    tmp_rng_ends = np.empty_like(tmp_rng_starts)
    # i = alive_cum_rng_nums[len(running_seq_ids)]
    # old_i = old_alive_cum_rng_nums[len(running_seq_ids)]
    # tmp_rng_starts[i:] = new_rng_starts[old_i:]
    # tmp_rng_ends[i:] = new_rng_ends[old_i:]

    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums}")

    for seq_i, add_rng in zip(range(len(old_inp_lens)), need_add_rng):
        i = alive_cum_rng_nums[seq_i]
        j = alive_cum_rng_nums[seq_i+1]
        if i==j:
            # the corresponding seq is finished
            continue
        old_i = old_alive_cum_rng_nums[seq_i]
        old_j = old_alive_cum_rng_nums[seq_i+1]

        # print(f"i, j, old_i, old_j: {i, j, old_i, old_j}")
        # print(f"tmp_rng_starts: {tmp_rng_starts}")
        # print(f"tmp_rng_ends: {tmp_rng_ends}")

        if add_rng:
            start, end = infer_progress[seq_i]
            tmp_rng_starts[i+1:j] = new_rng_starts[old_i:old_j]
            tmp_rng_ends[i+1:j] = new_rng_ends[old_i:old_j]
            tmp_rng_starts[i] = start
            tmp_rng_ends[i] = end
        else:
            # merge new rng or no new rng
            tmp_rng_starts[i:j] = new_rng_starts[old_i:old_j]
            tmp_rng_ends[i:j] = new_rng_ends[old_i:old_j]
            if len(infer_progress[seq_i])>0:
                start, end = infer_progress[seq_i]
                tmp_rng_starts[i] = start




    # remove finished seqs from alive_cum_rng_nums
    # alive_cum_rng_nums = np.unique(alive_cum_rng_nums)
    # get the alive seq ids
    alive_cum_rng_nums, valid_indices = np.unique(alive_cum_rng_nums, return_index=True)
    # the indices of alive seqs
    valid_indices = valid_indices[1:] - 1

    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums}")

    # 
    # print(f"new metadata---------------------------")
    # print(f"need_add_rng: {need_add_rng.tolist()}")
    # print(f"alive_cum_rng_nums: {alive_cum_rng_nums.tolist()}")
    # print(f"tmp_rng_starts: {tmp_rng_starts.tolist()}")
    # print(f"tmp_rng_ends: {tmp_rng_ends.tolist()}")
    # print(f"tmp_is_prefill_steps: {tmp_is_prefill_steps.tolist()}")
    assert len(tmp_rng_starts) == len(tmp_rng_ends)
    assert len(tmp_rng_starts) == len(tmp_rng_ends)
    assert len(tmp_cum_latencys) == len(tmp_is_prefill_steps)
    assert max(tmp_rng_ends) == len(tmp_is_prefill_steps)-1
    assert alive_cum_rng_nums[-1] == len(tmp_rng_starts)
    return tmp_cum_latencys, alive_cum_rng_nums, tmp_rng_starts, tmp_rng_ends, tmp_is_prefill_steps, valid_indices



def update_fake_FCFS_schedule_metadata(
        old_inp_lens: List[int], # new_inp_lens: List[int],
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int],
        is_prefill_steps: List[bool],
        max_num_batched_tokens: int, stop_iter_i: int,
        cost_table: CostTable, 
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Compute the fake FCFS scheduling metadata restart from the iter ``stop_iter_i+1'' based on the given metadata.
        NOTE: we restart all running seqs at iter ``stop_iter_i'' and not finished after iter ``stop_iter_i'' ends.
        We use prefill steps to recover their seq lens after iter ``stop_iter_i'' ends.
            (1) We do not consider the ``watermark'' constraint in the prefill stage.
        NOTE:
            this function assumes there are running seqs after iter ``stop_iter_i'' ends.
    '''
    
    if (len(cumsum_latencys) == 0) or (stop_iter_i == (len(cumsum_latencys)-1)):
        # this dp worker has finished!
        return np.asarray([]), np.asarray([0]), np.asarray([]), np.asarray([]), np.asarray([]), \
            np.asarray([]), np.asarray([])


    cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, valid_indices = \
        _update_fake_FCFS_schedule_metadata(
            old_inp_lens, # new_inp_lens,
            cumsum_latencys, cum_rng_nums, rng_starts, rng_ends,
            is_prefill_steps,
            max_num_batched_tokens, stop_iter_i,
            cost_table, 
            model_name, exec_plan, sample_config, trust_remote_code, revision)
    
    # get the finish time of each request in order
    finish_times = get_finish_times_from_rng_infos(cumsum_latencys, cum_rng_nums, rng_ends)
    return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times, valid_indices













def plot_seq_curve(logs, tag:str):
    import matplotlib.pyplot as plt
    seq_nums = [i[0] for i in logs]
    fig, ax = plt.subplots()
    ax.plot(range(len(seq_nums)), seq_nums)
    ax.set(xlabel='iter', ylabel='seq_num',)
        #    title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig(f"./test_sampler/seq_nums{'Llama-2-7b-hf'}_tp{1}_{tag}.png")
    plt.show()
    plt.close(fig)

def plot_cum_seqnum_curve(logs, tag:str):
    import matplotlib.pyplot as plt
    seq_nums = np.cumsum([i[0] for i in logs])
    fig, ax = plt.subplots()
    ax.plot(range(len(seq_nums)), seq_nums)
    ax.set(xlabel='iter', ylabel='seq_num',)
        #    title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig(f"./test_sampler/cum_seq_nums{'Llama-2-7b-hf'}_tp{1}_{tag}.png")
    plt.show()
    plt.close(fig)



def plot_latency_curve(latencys, tag: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(len(latencys)), latencys)
    ax.set(xlabel='iter', ylabel='latency (s)',)
        #    title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig(f"./test_sampler/latency_{'Llama-2-7b-hf'}_{tag}.png")
    plt.show()
    plt.close(fig)





def estimate_cost_from_predicted_logs(
        logs: List[Tuple[int, int]], cost_table: CostTable, is_prompt: bool,
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Estimate the costs of the iterations in the logs.
    '''
    seqnums = np.asarray([i[0] for i in logs])
    context_tot_lens = np.asarray([i[1] for i in logs])

    s = np.asarray([i[-1] for i in logs])
    # s = None
    # if is_prompt:
    #     # assert False, "We currently do not support prompt stage estimation"
    #     s = np.asarray([i[-1] for i in logs])
    # else:
    #     s = [1]

    latencys = cost_table.estimate_cost(
        seqnums, s, context_tot_lens, is_prompt, 
        model_name, exec_plan, sample_config, trust_remote_code, revision)
    return latencys



def _estimate_prefill_and_decode_cost_from_predicted_logs(
        prefill_logs: List[Tuple[int, int]], decode_logs: List[Tuple[int, int]],
        cost_table: CostTable,
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None
        ) -> Tuple[float, List[float], List[float]]:
    prefill_latencys = estimate_cost_from_predicted_logs(
        prefill_logs, cost_table=cost_table, is_prompt=True,
        model_name=model_name, exec_plan=exec_plan, sample_config=sample_config, 
        trust_remote_code=trust_remote_code, revision=revision)
    decode_latencys = estimate_cost_from_predicted_logs(
        decode_logs, cost_table=cost_table, is_prompt=False,
        model_name=model_name, exec_plan=exec_plan, sample_config=sample_config, 
        trust_remote_code=trust_remote_code, revision=revision)

    return sum(prefill_latencys)+sum(decode_latencys), prefill_latencys, decode_latencys




'''
We use a list to store the continuous inference iteration ranges for each sequence.
    E.g., 
        [start1, end1, start2, end2, ...] --> for iter_i with 
            start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
        the seq attend the corresponding iteration steps.
'''
def get_info_at_stop_time_slowVersion(
        decode_latencys: List[float], prefill_latencys: List[float], 
        is_prefill_steps: List[bool], infer_progress: List[List[int]], 
        stop_time: float):
    '''
        Get the seq statuses at the given stop time.
        Output:
            1. finished seq lengths;
            2. remaining seq lengths.
    '''    
    is_prefill_steps = np.asarray(is_prefill_steps)


    import time
    time1 = time.perf_counter()

    # 1. compute the finished iter num at the stop_time.
    latencys = np.empty(len(decode_latencys)+len(prefill_latencys))
    latencys[is_prefill_steps==True] = prefill_latencys
    latencys[is_prefill_steps==False] = decode_latencys
    cumsum_latencys = np.cumsum(latencys)
    # the stop_iter_i-th iter will be finished
    stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')

    # 2. get finished seq lengths and remaining seq lengths
    finished_lens = [0 for i in range(len(infer_progress))]
    remaining_lens = [0 for i in range(len(infer_progress))]
    for seq_i, rngs in enumerate(infer_progress):
        # find the position to insert stop_iter_i
        # rngs[i-1] < stop_iter_i <= rngs[i]
        i = np.searchsorted(rngs, stop_iter_i, side='left')
        tmp_rngs = np.asarray(rngs).reshape((-1, 2))
        rng_lens = tmp_rngs[:,1]-tmp_rngs[:,0]+1
        
        if i%2 == 1:
            # the stop iter is in an inference range of this seq
            finished_lens[seq_i] = sum(rng_lens[:i//2]) + stop_iter_i - rngs[i-1] + 1
        else:
            # the stop iter is between two ranges or at the start of a range
            finished_lens[seq_i] = sum(rng_lens[:i//2])
            if (i < len(rngs)) and (stop_iter_i == rngs[i]):
                finished_lens[seq_i] = finished_lens[seq_i] + 1
        
        remaining_lens[seq_i] = sum(rng_lens) - finished_lens[seq_i]


    time2 = time.perf_counter()
    print(f"time: {time2 - time1}")

    # check
    another_res = get_info_at_stop_time(
        decode_latencys, prefill_latencys, 
        is_prefill_steps, infer_progress, 
        stop_time)
    assert ((finished_lens==another_res[0]).all()) and ((remaining_lens==another_res[1]).all()), (finished_lens, remaining_lens, another_res)


    return finished_lens, remaining_lens



def _get_inferRng_info(infer_progress: List[List[int]]):
    '''
        Get the infer ranges information for each seq in the inference process.
    '''  
    # 2. process infer range information
    rng_nums = np.asarray([len(rngs) for rngs in infer_progress])//2
    cum_rng_nums = np.cumsum(np.concatenate(([0], rng_nums)))
    concat_infer_progress = np.concatenate(infer_progress).reshape((-1, 2))
    rng_starts = concat_infer_progress[:,0]
    rng_ends = concat_infer_progress[:,1]
    
    return cum_rng_nums, rng_starts, rng_ends





def get_cumLatency_inferRng_info(
        decode_latencys: List[float], prefill_latencys: List[float], 
        is_prefill_steps: List[bool], infer_progress: List[List[int]]):
    '''
        Get the cumulative latencys and the infer ranges information for each seq in the inference process.
    '''  
    # 1. prepare cum latency
    is_prefill_steps = np.asarray(is_prefill_steps)
    latencys = np.empty(len(decode_latencys)+len(prefill_latencys))
    latencys[is_prefill_steps==True] = prefill_latencys
    latencys[is_prefill_steps==False] = decode_latencys
    cumsum_latencys = np.cumsum(latencys)

    # 2. process infer range information
    # rng_nums = np.asarray([len(rngs) for rngs in infer_progress])//2
    # cum_rng_nums = np.cumsum(np.concatenate(([0], rng_nums)))
    # concat_infer_progress = np.concatenate(infer_progress).reshape((-1, 2))
    # rng_starts = concat_infer_progress[:,0]
    # rng_ends = concat_infer_progress[:,1]
    cum_rng_nums, rng_starts, rng_ends = _get_inferRng_info(infer_progress)
    return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends




'''
We use a list to store the continuous inference iteration ranges for each sequence.
    E.g., 
        [start1, end1, start2, end2, ...] --> for iter_i with 
            start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
        the seq attend the corresponding iteration steps.
'''
def get_info_at_stop_time( 
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int], 
        stop_time: float, stop_iter_i: int):
    '''
        Get the seq statuses at the given stop time.
        Output:
            1. finished seq lengths;
            2. remaining seq lengths.
        NOTE: this is the fast version.
    '''    

    # 1. compute the finished iter num at the stop_time.
    # the stop_iter_i-th iter will be finished
    # stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')

    # 2. get finished seq lengths and remaining seq lengths
    # rng_nums = np.asarray([len(rngs) for rngs in infer_progress])//2
    # cum_rng_nums = np.cumsum(np.concatenate(([0], rng_nums)))
    # concat_infer_progress = np.concatenate(infer_progress).reshape((-1, 2))
    finished_lens = (stop_iter_i>=rng_starts) * (np.minimum(stop_iter_i, rng_ends)-rng_starts+1)
    finished_lens = np.cumsum(np.concatenate(([0], finished_lens)))
    finished_lens = finished_lens[cum_rng_nums[1:]] - finished_lens[cum_rng_nums[:-1]]
    
    # tot_out_lens = concat_infer_progress[:,1] - concat_infer_progress[:,0] + 1
    # tot_out_lens = np.cumsum(np.concatenate(([0], tot_out_lens)))
    # tot_out_lens = tot_out_lens[cum_rng_nums[1:]]-tot_out_lens[cum_rng_nums[:-1]]
    # remaining_lens = tot_out_lens - finished_lens


    return finished_lens












def comp_flops_from_seqlens(
        inp_lens: List[int], out_lens: List[int], only_decode, cost_table: CostTable, 
        model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    if only_decode:
        B_array = np.asarray([sum(out_lens)])
        s_array = np.asarray([1])
        inp_lens = np.asarray(inp_lens)
        out_lens = np.asarray(out_lens)
        context_tot_len_array = np.asarray([sum((2*inp_lens+out_lens-1)*out_lens/2)])
        tp_size=1
        flops = cost_table.comp_flops(
            tp_size,
            B_array, s_array, context_tot_len_array, is_prompt=False,
            model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
        return flops
    else:
        # include both prefill and decode
        inp_lens = np.asarray(inp_lens)
        out_lens = np.asarray(out_lens)
        B_array = np.asarray([sum(inp_lens+out_lens-1)])
        s_array = np.asarray([1])
        context_tot_len_array = np.asarray([sum((inp_lens+out_lens)*(inp_lens+out_lens-1)/2)])
        tp_size=1
        flops = cost_table.comp_flops(
            tp_size,
            B_array, s_array, context_tot_len_array, is_prompt=False,
            model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
        return flops


def comp_valid_throughput_at_stop_time(
        inp_lens: List[int],
        finished_lens: List[int],
        stop_time: float, cost_table: CostTable,
        model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Valid flops means we only consider the necessary flops in model computation 
        (not including 
            1. prepare input tensor and sampling, 
            2. as well as the waste flops due to padding or recomputation after kill).
            3. as well as the waste flops that may be due to tensor parallelism (e.g., kv_head_num < tp_size).
        Valid throughput is computed based on valid flops.
    '''

    # # 1. first get the inference informatio at the stop_time.
    # finished_lens, remaining_lens = get_info_at_stop_time(
    #     decode_latencys, prefill_latencys, 
    #     is_prefill_steps, infer_progress, 
    #     stop_time)
    
    # 2. comp the VALID flops according to the finished lens. 
    # NOTE: we do not directly sum up the flops of each iteration
    #   because there may be some seqs that are killed and recomputed, whose flops should not be counted.
    # NOTE: for the same reason, we do not consider the flops brought by padding as well.
    inp_lens_array = np.asarray(inp_lens)
    finished_lens_array = np.asarray(finished_lens)
    
    # the total length of the seqs (has been computed) before the stop iter
    tot_lens_array = (inp_lens_array + finished_lens_array - 1) * (finished_lens_array>0)
    
    # B_array = np.asarray([sum(tot_lens_array)])
    # s_array = np.asarray([1])
    # # TODO: 这个地方好像写错了，context tot len要求和，好奇怪啊，那为啥这里的throughput看起来这么正常
    # context_tot_len_array = np.asarray([sum(((1+tot_lens_array)*tot_lens_array)/2)])
    # tp_size=1
    # flops = cost_table.comp_flops(
    #     tp_size,
    #     B_array, s_array, context_tot_len_array, is_prompt=False,
    #     model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
    


    flops = comp_flops_from_seqlens(
        inp_lens=tot_lens_array, out_lens=[1], only_decode=False, cost_table=cost_table, 
        model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)


    # 3. comp the valid throughput 
    throughput = flops / stop_time
    print(f"flops: {flops}, stop_time: {stop_time}, throughput: {throughput}")

    return throughput







# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# below is some test code


def test():
    # read outlen distributions from experiment log file.
    import json

    # filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_2.log'
    # surfix = 'eos_'
    filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_3.log'
    surfix = ''

    def get_lens(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'output_lens =' in line:
                    pos = len('output_lens =')
                    values = json.loads(line[pos:])
                    return values



    lens = get_lens(filename)
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens] # actual outputs
    max_seq_num = 512
    max_cache_slot_num = 7339*16
    max_num_batched_tokens = 4096
    max_block_num = 7339
    block_size = 16
    fake_FCFS_schedule(
        inp_lens=inps, out_lens=outs, 
        max_seq_num=max_seq_num, max_block_num=max_block_num, max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size)



# def test(model: str, max_seq_num: int, max_cache_slot_num: int):
def test_sampler():
    import output_length_sampler
    # get inp_lens of SharedGPT
    # read outlen distributions from experiment log file.

    model = 'Llama-2-7b-hf'
    max_seq_num = 512
    max_cache_slot_num = 7339*16
    max_num_batched_tokens = 4096
    max_block_num = 7339
    block_size = 1*16

    import json

    # filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_6.log' # ignore eos
    # surfix = 'eos_'
    filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_7.log' # not ignore eos
    surfix = ''

    def get_lens(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'output_lens =' in line:
                    pos = len('output_lens =')
                    values = json.loads(line[pos:])
                    return values



    lens = get_lens(filename)
    inps = [i[0] for i in lens]
    outs = output_length_sampler.sample_out_len_for_given_model(model=model, inp_lens=inps)
    # outs = output_length_sampler.get_expectation_out_len_for_given_model(model=model, inp_lens=inps)
    logs, prefill_logs, is_prefill_steps, infer_progress = fake_FCFS_schedule(
        inp_lens=inps, out_lens=outs, 
        max_seq_num=max_seq_num, max_block_num=max_block_num, max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size)
        
    # plot_seq_curve(logs)
    # plot_seq_curve(prefill_logs, tag='prefill')
    # plot_cum_seqnum_curve(prefill_logs, tag='prefill')



    # get cost table and estimate exec latencys
    # each item is (model_name, exec_plan, filename)
    # each exec_plan is (tp, gpu_ratio, wldeg, cache_gpus)
    model_path = 'NousResearch/Llama-2-7b-hf'
    exec_plan = (1, 0.9, 2, 0) # (tp, gpu_ratio, wldeg, cache_gpu_num)
    sample_config = (1, 1, -1, 0) #(temp, top_p, top_k, min_p)
    
    # logfiles = [(model_path, (1, 0.9, 2, 0), './Cost_Model_per_iter/Formal_Llama-2-7b-hf_0430_tp1_temp1.0_wldeg2.log')]
    # each tuple in logfiles: (model_name, exec_plan, sample_config, filename)
    logfiles = [
        (model_path, exec_plan, sample_config, './Cost_Model_per_iter/Formal_Llama-2-7b-hf_0503_tp1_temp1.0_wldeg2_samp_prepInp.log', 5, 1e-3, True,None),
        ]
    
    # each tuple in prepInp_decode_logfiles: (model_name, exec_plan, filename)
    prepInp_decode_logfiles = [(model_path, exec_plan, './Cost_Model_per_iter/Formal_Llama-2-7b-hf_0503_tp1_temp1.0_wldeg2_DecodePrepInp.log')]

    # each item is (model_name, trust_remote_code, revision)
    model_infos = [(model_path, True, None)]

    # init cost_table
    cost_table = CostTable(logfiles, prepInp_decode_logfiles, model_infos)

    decode_latencys = estimate_cost_from_predicted_logs(
        logs=logs, cost_table=cost_table, is_prompt=False, 
        model_name=model_path, exec_plan=exec_plan, sample_config=sample_config, trust_remote_code=True, revision=None)
    prefill_latencys = estimate_cost_from_predicted_logs(
        logs=prefill_logs, cost_table=cost_table, is_prompt=True, 
        model_name=model_path, exec_plan=exec_plan, sample_config=sample_config, trust_remote_code=True, revision=None)

    # def get_per_iter_records(filename):
    #     ret = list()
    #     with open(filename, 'r') as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             if 'TFLOPs Time:' in line:
    #                 items = line.split(' ')
    #                 # ['(4,', 4086,', '2084896)', '28.07862132736', 'TFLOPs', 'Time:', '0.2851552767679095', 's', '0.00032312609255313873', 's', '0.27753941575065255', 's', '0.004479971248656511', 's', 'Throughput:', '98.46783003848617', 'TFLOPs/s', '101.16985096122868', 'TFLOPs/s']
    #                 values = items[0][1:-1], items[1][:-1], items[2][:-1], items[3], items[6], items[8], items[10], items[12], items[15], items[17]
    #                 values = [float(i) for i in values]
    #                 ret.append(values)
    #     return ret
    
    def get_per_iter_records(filename):
        per_iter_records = list()
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if '[[' == line[:2]:
                    flop_metadata, flops, times, throughputs, tp_size = \
                        MyThroughputLogger.load_record(line=line)
                    per_iter_records.append((flop_metadata, flops, times, throughputs))
        return per_iter_records
    
    per_iter_records = get_per_iter_records(filename)
    real_decode_latencys = [times[0]-times[-1] \
        for (flop_metadata, _, times, _) in per_iter_records if not flop_metadata[-1]]
    real_prefill_latencys = [times[0]-times[-1] \
        for (flop_metadata, _, times, _) in per_iter_records if flop_metadata[-1]]

    # plot_seq_curve([[i[0]] for i in per_iter_records if i[0] != i[1]], tag='prefill_real')
    # plot_cum_seqnum_curve([[flop_metadata[0]] \
    #     for (flop_metadata, _, _, _) in per_iter_records if flop_metadata[-1]], tag='prefill_real')


    # compare the latencys
    pred = sum(decode_latencys)
    real = sum(real_decode_latencys)
    pred_prefill = sum(prefill_latencys)
    real_prefill = sum(real_prefill_latencys)
    print(f"total exec latency: pred: {pred} VS real: {real}")
    print(f"total exec latency: pred: {pred_prefill} VS real: {real_prefill}")

    # plot the latency curve
    # plot_latency_curve(decode_latencys, tag=f'{exec_plan}_pred')
    # plot_latency_curve(real_decode_latencys, tag=f'{exec_plan}_real')
    # plot_latency_curve(prefill_latencys, tag=f'{exec_plan}_pred_prefill')
    # plot_latency_curve(real_prefill_latencys, tag=f'{exec_plan}_real_prefill')
    return ((pred, real), (pred_prefill, real_prefill))


if __name__ == 'main':
    # read outlen distributions from experiment log file.
    # test()
    test_sampler()
    res = [test_sampler() for i in range(10)]
    print(f'Decode-pred:{np.mean([i[0][0] for i in res])} VS real:{np.mean([i[0][1] for i in res])}') 
    print(f'Prefill-pred:{np.mean([i[1][0] for i in res])} VS real:{np.mean([i[1][1] for i in res])}') 

