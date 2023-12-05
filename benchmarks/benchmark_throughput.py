"""Benchmark offline inference throughput."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['USE_OUR_METHOD'] = 'False'

os.environ['run_profile'] = 'False'
os.environ['step_start'] = '250'
os.environ['step_end'] = '350'


run_simple_test = False



import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


# <jingzhi> 
import numpy as np
import copy


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests






def my_sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # if (prompt_len < 1180 or prompt_len + output_len > 1260):
        #     continue
        # if prompt_len > 1024:
        #     continue
        # if (output_len > 128) or (output_len < 100):
        #     continue
        filtered_dataset.append((prompt, prompt_len, output_len))


    # for extreme case test or for cost model profiling
    # filtered_dataset = [(filtered_dataset[0][0],filtered_dataset[0][1], 2+16)]
    # filtered_dataset = [(filtered_dataset[0][0],filtered_dataset[0][1], 120)]
    # filtered_dataset = [(info[0],info[1], 120) for info in filtered_dataset]

    # Sample the requests.
    print(f"len(filtered_dataset): {len(filtered_dataset)}")
    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))
    print(f"len(sampled_requests): {len(sampled_requests)}")
    if len(sampled_requests) < 1000:
        sampled_requests = [_ for _ in sampled_requests for i in range((num_requests+len(sampled_requests)-1)//len(sampled_requests))]

    print(f"len(sampled_requests): {len(sampled_requests)}, total token num: {sum([info[1]+info[2] for info in sampled_requests])}")

    if run_simple_test:
        sampled_requests = [(None, 5, 6), (None, 8, 3)]


    sampled_requests = sorted(sampled_requests, key=lambda info: info[2]) #, reverse=True)
    print(f"sampled_requests: {[info[1:] for info in sampled_requests]}")

    return sampled_requests









def dry_run_by_outlen(requests:List[Tuple[str, int, int]]):
    # 运行一个backtracking search来找最优的request scheduling 策略。
    # 暂时先只考虑按照给定的request考虑把新的request加进GPU中。
    
    def get_blk_num(seqlen:int)->int:
        block_size = 16
        return (seqlen+block_size-1)//block_size

    max_seq_num = 256
    schedules = list()
    finished_num = 0
    tot_num = len(requests)

    curr_on_card = list()
    curr_released = list()
    
    curr_blk_num = 0
    tot_blk_num = 1350

    data_movement = 0
    iter_num = 0

    while finished_num < tot_num:
        iter_num+=1
        iter_schedule = list()
        print(f"iternum: {iter_num}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}")

        if len(curr_released)==0:
            print("1")
            # first consider add new requests
            quota = max_seq_num-len(curr_on_card)
            new_requests = list()
            for req in requests[:quota]:
                if get_blk_num(req[1]) <= tot_blk_num - curr_blk_num:
                    iter_schedule.append((req[1],req[2]))
                    curr_blk_num = curr_blk_num + get_blk_num(req[1])
                else:
                    new_requests.append(req)
            curr_on_card = curr_on_card + iter_schedule
            requests = new_requests + requests[quota:]
            # print(f"requests: {[_[1:] for _ in requests]}, curr_on_card: {curr_on_card}")
            schedules.append(iter_schedule)
        if len(iter_schedule) == 0:
            print("2")
            # check whether the space for on card requests is enough
            required_blk_num = sum([get_blk_num(info[0]) for info in curr_on_card])
            if required_blk_num <= tot_blk_num:
                print("2.1")
                iter_schedule = curr_on_card.copy()
                if len(curr_released) > 0:
                    print("2.1.1")
                    # consider add in more requests  based on backtracking search?
                    # 但是不能直接用cost来当指标，因为我们希望cost最小，这样的话，就会自动选择到每个iteration只运行1个request。
                    # 但是目前并没有一个metric帮助选择request。先无脑选output length更短的吗？（但是这样就变成greedy了）之后再设计更好的算法。
                    curr_released = sorted(curr_released, key=lambda info: info[1])
                    selected_ids = list()
                    for i, info in enumerate(curr_released):
                        if len(curr_on_card) == max_seq_num:
                            break
                        if get_blk_num(info[0]) <= tot_blk_num - curr_blk_num:
                            # print(f"blk num infor: {get_blk_num(info[0]), tot_blk_num, curr_blk_num}")
                            iter_schedule.append(info)
                            curr_blk_num = curr_blk_num + get_blk_num(info[0])
                            selected_ids.append(i)
                            curr_on_card.append(info)
                            data_movement = data_movement + get_blk_num(info[0])
                    curr_released = [curr_released[i] for i in range(len(curr_released)) if i not in selected_ids]
                schedules.append(iter_schedule)
            else:
                print("2.2")
                # need to swap out requests
                curr_on_card = sorted(curr_on_card, key=lambda info: info[1])
                iter_schedule = list()
                curr_blk_num = 0
                for info in curr_on_card:
                    if get_blk_num(info[0]) <= tot_blk_num - curr_blk_num:
                        # print(f"blk num infor: {get_blk_num(info[0]), tot_blk_num, curr_blk_num}")
                        iter_schedule.append(info)
                        curr_blk_num = curr_blk_num + get_blk_num(info[0])
                        # print(f"iter_schedule: {iter_schedule}")
                    else:
                        curr_released.append(info)
                        data_movement = data_movement + get_blk_num(info[0])
                schedules.append(iter_schedule)
                curr_on_card = iter_schedule
                # print(f"curr_on_card: {curr_on_card}")
        # update req status after 1 step inference
        new_curr_on_card = list()
        for info in curr_on_card:
            if info[1] == 1:
                finished_num += 1
            else:
                new_curr_on_card.append((info[0] + 1, info[1] - 1))
        curr_on_card = new_curr_on_card
        curr_blk_num = sum([get_blk_num(info[0]-1) for info in curr_on_card])
        print(f"iternum: {iter_num}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}")


    # finish the dry run, now report the metadata
    print(f"Total data movement: {data_movement}")
    print(f"Total iteration number: {iter_num}")








# 这个写法计算效率太低了
# 怎样加快速度？需要pruned掉一些东西？
# 
def DP_select_requests_to_release_slow(curr_on_card, tot_blk_num, to_release, best = [[], float('inf')]):
    '''
    to_release: a list of already selected requests
    best: [the best to_release set, the future #block demand of the best solution]
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


        if run_simple_test:
            print(f"--In DP--: curr_on_card: {curr_on_card}, to_release: {to_release}, future_peak_demand: {future_peak_demand}, best: {best}, demand: {demand}, tot_blk_num: {tot_blk_num}")
        else:
            print(f"--In DP--: to_release: {to_release}, future_peak_demand: {future_peak_demand}, best: {best}, demand: {demand}, tot_blk_num: {tot_blk_num}")

        if future_peak_demand < best[1]:
            best[0] = copy.deepcopy(to_release)
            best[1] = future_peak_demand
        return

    # 
    for info_i, info in enumerate(curr_on_card):
        to_release = to_release + [info]
        DP_select_requests_to_release(curr_on_card[:info_i]+curr_on_card[info_i+1:], tot_blk_num, to_release, best)
        to_release = to_release[:-1]











def DP_select_requests_to_release(curr_on_card, candidates, tot_blk_num, to_release, best = [[], [], float('inf')]):
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


        if run_simple_test:
            print(f"--In DP--: curr_on_card: {curr_on_card}, to_release: {to_release}, future_peak_demand: {future_peak_demand}, best: {best}, demand: {demand}, tot_blk_num: {tot_blk_num}")
        else:
            print(f"--In DP--: to_release: {to_release}, future_peak_demand: {future_peak_demand}, best: {best[0],best[2]}, demand: {demand}, tot_blk_num: {tot_blk_num}")

        if future_peak_demand < best[2]:
            best[0] = copy.deepcopy(to_release)
            best[1] = copy.deepcopy(curr_on_card)
            best[2] = future_peak_demand
        return True

    # 
    assert curr_on_card[-len(candidates):] == candidates


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








# we estimate the future block requirement pressure and release/reload requests based on this
def dry_run(requests:List[Tuple[str, int, int]]):
    # 运行一个backtracking search来找最优的request scheduling 策略。
    # 暂时先只考虑按照给定的request考虑把新的request加进GPU中。
    
    def get_blk_num(seqlen:int)->int:
        block_size = 1 if run_simple_test else 16 # 16 # simple test
        return (seqlen+block_size-1)//block_size

    max_seq_num = 256
    schedules = list()
    finished_num = 0
    tot_num = len(requests)

    curr_on_card = list()
    curr_released = list()
    
    curr_blk_num = 0
    tot_blk_num = 13 if run_simple_test else 1350 # 1350 # simple test

    data_movement = 0
    iter_num = 0
    blk_num_to_release = 0

    while finished_num < tot_num:
        iter_num+=1
        iter_schedule = list()
        print(f"iternum: {iter_num}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}")

        if len(curr_released)==0:
            print("1")
            # first consider add new requests
            quota = max_seq_num-len(curr_on_card)
            new_requests = list()
            for req in requests[:quota]:
                if get_blk_num(req[1]) <= tot_blk_num - curr_blk_num:
                    iter_schedule.append((req[1],req[2]))
                    curr_blk_num = curr_blk_num + get_blk_num(req[1])
                else:
                    new_requests.append(req)
            curr_on_card = curr_on_card + iter_schedule
            requests = new_requests + requests[quota:]
            # print(f"requests: {[_[1:] for _ in requests]}, curr_on_card: {curr_on_card}")
            schedules.append(iter_schedule)
        if len(iter_schedule) == 0:
            print("2")
            # check whether the space for on card requests is enough
            required_blk_num = sum([get_blk_num(info[0]) for info in curr_on_card])
            if required_blk_num <= tot_blk_num:
                print("2.1")
                iter_schedule = curr_on_card.copy()

                # deal with the page-swap-out case
                if blk_num_to_release + required_blk_num > tot_blk_num:
                    print("2.1.0")
                    # 需要接着挪出一部分之前没挪完的block
                    to_release_this_iter = sum([get_blk_num(info[0]) for info in curr_on_card]) - \
                        sum([get_blk_num(info[0] - 1) for info in curr_on_card])
                    to_release_this_iter = to_release_this_iter - \
                        (tot_blk_num - sum([get_blk_num(info[0] - 1) for info in curr_on_card]) - blk_num_to_release)
                    assert to_release_this_iter >= 0

                    data_movement = data_movement + to_release_this_iter

                    blk_num_to_release = blk_num_to_release - to_release_this_iter
                    assert blk_num_to_release >= 0

                    schedules.append(iter_schedule)
                    print(f"page-swap-in: {to_release_this_iter}, data_movement: {data_movement}")
                else:
                    # 不需要挪出之前没挪完的部分依然有足够的空间，可以考虑挪回之前被release的request
                    curr_blk_num = required_blk_num
                    if len(curr_released) > 0:
                        print("2.1.1")
                        # consider add in more requests  based on backtracking search?
                        # 但是不能直接用cost来当指标，因为我们希望cost最小，这样的话，就会自动选择到每个iteration只运行1个request。
                        # 但是目前并没有一个metric帮助选择request。先无脑选output length更短的吗？（但是这样就变成greedy了）之后再设计更好的算法。
                        # curr_released = sorted(curr_released, key=lambda info: info[1])
                        selected_ids = list()
                        for i, info in enumerate(curr_released[::-1]):
                            if len(curr_on_card) == max_seq_num:
                                break
                            if get_blk_num(info[0]) <= tot_blk_num - curr_blk_num:
                                # print(f"blk num infor: {get_blk_num(info[0]), tot_blk_num, curr_blk_num}")
                                iter_schedule.append(info)
                                curr_blk_num = curr_blk_num + get_blk_num(info[0])
                                selected_ids.append(i)
                                curr_on_card.append(info)
                                data_movement = data_movement + get_blk_num(info[0]-1)
                        curr_released = [curr_released[i] for i in range(len(curr_released)) if i not in selected_ids]
                        if len(selected_ids) >= 1:
                            # 有一部分block已经在GPU上了，无需挪动。这个地方好像有点问题，因为这个条件并不等于partially released的request就被swap in了。
                            data_movement = data_movement - blk_num_to_release
                            blk_num_to_release = 0
                    schedules.append(iter_schedule)
                    if run_simple_test:
                        print(f"after reloading: {iter_schedule}, data_movement: {data_movement}")
            else:
                print("2.2")
                # need to swap out requests

                if iter_num == 134:
                    print(f"curr_on_card: {curr_on_card}")


                curr_on_card = sorted(curr_on_card, key=lambda info: info[0]) # sort by increasing in_lens

                # 不是单纯分析output token的最终总量，而是on card的request的峰值extra token总量，只能greedy地分析？
                # 这里是不是可以用DP算法来求？但是dp算法求出来也需要一个顺序，应该怎么给这些request排序呢？可以不用排序，重新load的时候也搞一个dp？
                # 暂时还是按greedy来，按照对峰值extra token的贡献程度来排序。First in Last out。
                best_solution = [[], [], float('inf')]
                DP_select_requests_to_release(curr_on_card, curr_on_card, tot_blk_num, list(), best = best_solution)
                print(f"release selection by DP: {best_solution}")

                iter_schedule = best_solution[1]
                curr_released = curr_released + best_solution[0]
                schedules.append(iter_schedule)
                curr_on_card = iter_schedule

                # compute the number blocks that can be released in the future without release more requests
                to_release_this_iter = sum([get_blk_num(info[0]) for info in iter_schedule]) - \
                    sum([get_blk_num(info[0] - 1) for info in iter_schedule])
                to_release_this_iter = max(0, \
                    to_release_this_iter - (tot_blk_num - blk_num_to_release - sum([get_blk_num(info[0] - 1) for info in iter_schedule+best_solution[0]])))
                # 必定得把之前剩余的block全都release掉   为啥这里一定会把剩余的block全部消耗掉？不一定？实验里面出现了这个assertion为否的情况。理论上确实如此？
                # 因为否则的话，我们可以保留更多的request把blk_num_to_release 消耗完
                assert (to_release_this_iter >= blk_num_to_release), f"best_solution: {best_solution}, blk_num_to_release: {blk_num_to_release}, to_release_this_iter: {to_release_this_iter}, iter_schedule: {iter_schedule}, required_blk_num: {required_blk_num}"

                blk_num_to_release = blk_num_to_release + sum([get_blk_num(info[0]-1) for info in best_solution[0]])
                blk_num_to_release = blk_num_to_release - to_release_this_iter

                data_movement = data_movement + to_release_this_iter

                print(f"page-swap-out: {to_release_this_iter}, data_movement: {data_movement}")

        # update req status after 1 step inference
        new_curr_on_card = list()
        for info in curr_on_card:
            if info[1] == 1:
                finished_num += 1
            else:
                new_curr_on_card.append((info[0] + 1, info[1] - 1))
        curr_on_card = new_curr_on_card
        curr_blk_num = sum([get_blk_num(info[0]) for info in curr_on_card]) # 在下一个iter 当前这些on card 的request需要的总block数
        if run_simple_test:
            print(f"iternum: {iter_num}, curr_on_card: {curr_on_card}, curr_released: {curr_released}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}, blk_num_to_release: {blk_num_to_release}")
        else:
            print(f"iternum: {iter_num}, curr_released_num: {len(curr_released)}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}, blk_num_to_release: {blk_num_to_release}")

    # finish the dry run, now report the metadata
    print(f"Total data movement: {data_movement}")
    print(f"Total iteration number: {iter_num}")














def dry_run_vllm(requests:List[Tuple[str, int, int]]):
    # 运行一个backtracking search来找最优的request scheduling 策略。
    # 暂时先只考虑按照给定的request考虑把新的request加进GPU中。
    
    def get_blk_num(seqlen:int)->int:
        block_size = 1 if run_simple_test else 16 # 16 # simple test
        return (seqlen+block_size-1)//block_size

    max_seq_num = 256
    schedules = list()
    finished_num = 0
    tot_num = len(requests)

    curr_on_card = list()
    curr_released = list()
    
    curr_blk_num = 0
    tot_blk_num = 13 if run_simple_test else 1350 # 1350 # simple test
    watermark = int(0.01*tot_blk_num)

    data_movement = 0
    iter_num = 0

    while finished_num < tot_num:
        iter_num+=1
        iter_schedule = list()
        print(f"iternum: {iter_num}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}")

        if len(curr_released)==0:
            print("1")
            # first consider add new requests
            quota = max_seq_num-len(curr_on_card)
            new_requests = list()
            for req in requests[:quota]:
                if get_blk_num(req[1]) <= tot_blk_num-watermark - curr_blk_num:
                    iter_schedule.append((req[1],req[2]))
                    curr_blk_num = curr_blk_num + get_blk_num(req[1])
                else:
                    new_requests.append(req)
            curr_on_card = curr_on_card + iter_schedule
            requests = new_requests + requests[quota:]
            schedules.append(iter_schedule)
        if len(iter_schedule) == 0:
            print("2")
            # check whether the space for on card requests is enough
            required_blk_num = sum([get_blk_num(info[0]) for info in curr_on_card])
            if required_blk_num <= tot_blk_num:
                print("2.1")
                iter_schedule = curr_on_card.copy()
                if len(curr_released) > 0:
                    print("2.1.1")
                    # consider add in more requests  based on backtracking search?
                    # 但是不能直接用cost来当指标，因为我们希望cost最小，这样的话，就会自动选择到每个iteration只运行1个request。
                    # 但是目前并没有一个metric帮助选择request。先无脑选output length更短的吗？（但是这样就变成greedy了）之后再设计更好的算法。
                    selected_ids = list()
                    for i, info in enumerate(curr_released):
                        if len(curr_on_card) == max_seq_num:
                            break
                        if get_blk_num(info[0]) <= tot_blk_num - curr_blk_num:
                            iter_schedule.append(info)
                            curr_blk_num = curr_blk_num + get_blk_num(info[0])
                            selected_ids.append(i)
                            curr_on_card.append(info)
                            data_movement = data_movement + get_blk_num(info[0]-1)
                        else:
                            break
                    curr_released = [curr_released[i] for i in range(len(curr_released)) if i not in selected_ids]
                schedules.append(iter_schedule)
            else:
                print("2.2")
                # need to swap out requests
                iter_schedule = list()
                curr_blk_num = 0
                for info in curr_on_card:
                    if get_blk_num(info[0]) <= tot_blk_num-watermark - curr_blk_num:
                        iter_schedule.append(info)
                        curr_blk_num = curr_blk_num + get_blk_num(info[0])
                    else:
                        curr_released.append(info)
                        data_movement = data_movement + get_blk_num(info[0]-1)
                schedules.append(iter_schedule)
                curr_on_card = iter_schedule
        # update req status after 1 step inference
        new_curr_on_card = list()
        for info in curr_on_card:
            if info[1] == 1:
                finished_num += 1
            else:
                new_curr_on_card.append((info[0] + 1, info[1] - 1))
        curr_on_card = new_curr_on_card
        curr_blk_num = sum([get_blk_num(info[0]) for info in curr_on_card]) # the number of blocks in the next iteration required by on card requests
        print(f"iternum: {iter_num}, on_card_num: {len(curr_on_card)}, gpu_blk_num: {sum([get_blk_num(info[0]) for info in curr_on_card])}, data_movement: {data_movement}")


    # finish the dry run, now report the metadata
    print(f"Total data movement: {data_movement}")
    print(f"Total iteration number: {iter_num}")
    










def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,

        # <jingzhi> parameter setting
        gpu_memory_utilization= 0.9, # 0.4, #  0.1801,
        swap_space=58,

    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do use internal method.
    llm._run_engine(use_tqdm=True, run_profile=(os.environ['run_profile'] == 'True'), 
        step_start=int(os.environ['step_start']), step_end=int(os.environ['step_end']))
    end = time.perf_counter()

    print(f"total time: {end - start}, DP time: {llm.llm_engine.scheduler.my_scheduler_config.DP_time}")
    return end - start


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    requests = my_sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "vllm":
        elapsed_time = run_vllm(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")




def dry_run_main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    requests = my_sample_requests(args.dataset, args.num_prompts, tokenizer)


    dry_run(requests)
    dry_run_vllm(requests)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "dryrun"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    args = parser.parse_args()

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    if args.tokenizer is None:
        args.tokenizer = args.model


    if args.backend == "dryrun":
        dry_run_main(args)
    else:
        main(args)


    # main(args)
