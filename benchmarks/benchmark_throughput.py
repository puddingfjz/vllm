"""Benchmark offline inference throughput."""





import os
# os.environ['CUDA_VISIBLE_DEVICES']='3,1,2,0' # '2,3' # '3,0,1,2' # should be set before initialize cuda in torch
os.environ['USE_VLLM']='True'
# os.environ['TOT_GPU_NUM'] = '4' # should be consistent with os.environ['CUDA_VISIBLE_DEVICES']
# os.environ['WEIGHT_LOAD_DEGREE'] = '16' # now will set it in command
# os.environ['CHANGE_KV_LAYOUT'] = 'True' # whether the KV layout is changed
os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False' # whether we will dynamically increase the on-card layer weights


os.environ['RUN_MULTI_MODEL'] = 'False' # whether this model is running in a multi-model environment
os.environ['SOFT_RESCHEDULE'] = 'False' # whether to reinitialize LLMs directly or update the current LLM (i.e., soft reschedule)
os.environ['NO_PREEMPT'] = 'True' # allow model preemption or not
# about scheduling
os.environ['SORT_REQS'] = 'True' # whether to sort the requests according to their output lengths, default is False
os.environ['COLLECT_TIME_LOG'] = 'False' # whether to sort the requests according to their output lengths, default is False



def environs_are_correct():
    if os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] == 'True':
        assert (os.environ['USE_VLLM'] == 'False')

# we first check the os environ variables are correct
environs_are_correct()
    

'''
Command: 
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager > layerBylayer1.log
/ssddata/jingzhi/Nsight_Systems_2023_2_1/target-linux-x64/nsys profile -w true -t cuda,nvtx,osrt -s cpu  --cudabacktrace=true -x true -o ./nsys_profile/my_profile1 python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 100 --enforce-eager > DEBUG.log

# with record range
/ssddata/jingzhi/Nsight_Systems_2023_2_1/target-linux-x64/nsys profile -w true -t cuda,nvtx,osrt -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop-shutdown --kill=sigkill --cudabacktrace=true -x true -o ./nsys_profile/my_profile3 python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 100 --enforce-eager > DEBUG.lpg


try llama2
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 100 --enforce-eager > layerBylayer_llama2_1.log

NousResearch/Llama-2-70b-hf
NousResearch/Llama-2-7b-hf 
models--NousResearch--Llama-2-7b-chat-hf

use this line to occupy memory
import torch
c = torch.empty(70*1024*1024*1024//4, device=torch.device('cuda:2'))
d = torch.empty(70*1024*1024*1024//4, device=torch.device('cuda:3'))


llama2:
0.538 gpu: 5223 blocks PD=10
0.537 gpu: 5210 blocks PD=10
0.5372 gpu: 5213 blocks PD=10
'''










import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from tqdm import tqdm


# <jingzhi>
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
# to support data parallel
from vllm.engine.ray_utils import ray
# from my_llm_infer_worker import LLM_INFER_WORKER, REMOTE_LLM_INFER_WORKER, REMOTE_LLM_INFER_WORKER_MESSAGE_PASSER
import my_llm_infer_worker_multiprocessing
from collections import deque

print(f'executing benchmark_throughput.py')





def get_dataset(dataset_path: str):
    if dataset_path == 'ShareGPT_V3_unfiltered_cleaned_split.json':
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data["conversations"][0]["value"],
                    data["conversations"][1]["value"]) for data in dataset]
        return dataset
    elif dataset_path == 'no_robot.parquet':
        # deal with other dataset
        import pyarrow.parquet as pq
        dataset = list()
        for fname in ['no_robot_train.parquet', 'no_robot_test.parquet']:
            a = pq.read_table(fname)
            a = a.to_pylist()
            dataset.extend([(data['messages'][0]['content'],
                             data['messages'][1]['content']) for data in a])
        return dataset
    elif dataset_path == 'train-00000-of-00001-b334c773bce22cb2.parquet':
        # NOTE: for this dataset, there is no answer text, we only have the prompt part
        # deal with other dataset
        import pyarrow.parquet as pq
        dataset = list()
        for fname in ['train-00000-of-00001-b334c773bce22cb2.parquet']:
            a = pq.read_table(fname)
            a = a.to_pylist()
            dataset.extend([data['text'] for data in a])
        return dataset
          





def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    # <jingzhi>
    random_seed: int=0,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    # with open(dataset_path) as f:
    #     dataset = json.load(f)
    # # Filter out the conversations with less than 2 turns.
    # dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # # Only keep the first two turns of each conversation.
    # dataset = [(data["conversations"][0]["value"],
    #             data["conversations"][1]["value"]) for data in dataset]

    # <jingzhi>
    dataset = get_dataset(dataset_path)

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
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
    # <jingzhi> make sample size be ``min(num_requests, len(filtered_dataset))''
    random.seed(random_seed)
    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))

    if os.environ['SORT_REQS'] == 'True':
        sampled_requests = sorted(sampled_requests, key=lambda x: x[1], reverse=True)


    print(f"tot_tokens: {sum([x[1]+x[2] for x in sampled_requests])}, tot_context_lens: {sum([(x[1]+x[2]-1)*(x[1]+x[2])/2 for x in sampled_requests])}")

    return sampled_requests



# the original version of run_vllm
def run_vllm_ori(
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
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    
    # <jingzhi>
    gpu_memory_utilization: float,
    temperature: float,
    ignore_eos: bool,
    fixed_output_len: int,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        # <jingzhi>
        # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
        # max_num_seqs=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=512,
        max_paddings=512,
    )

    print(f"finish init LLM engine")

    # <jingzhi>
    print(f"max_model_len: {llm.llm_engine.model_config.max_model_len}")
    print(f"temperature: {temperature}")
    print(f"ignore_eos: {ignore_eos}")

    # Add the requests to the engine.
    # for prompt, _, output_len in requests:
    #     sampling_params = SamplingParams(
    #         n=n,
    #         # <jingzhi> change to greedy sampling to check correctness.
    #         temperature=0.0 if use_beam_search else 1e-6, #1.0
    #         top_p=1.0,
    #         use_beam_search=use_beam_search,
    #         ignore_eos=True,
    #         max_tokens=output_len,
    #     )
    #     # FIXME(woosuk): Do not use internal method.
    #     llm._add_request(
    #         prompt=prompt,
    #         prompt_token_ids=None,
    #         sampling_params=sampling_params,
    #     )
    
    # we need to apply chat template if possible
    tokenizer_obj = AutoTokenizer.from_pretrained(
        tokenizer, trust_remote_code=trust_remote_code)
    print(f"tokenizer.chat_template: {tokenizer.chat_template}, tokenizer.chat_template!=None: {tokenizer.chat_template!=None}", flush=True)
    
    for prompt, _, output_len in requests:
        # print(f"in len: {_}, out len: {output_len} vs {4096-_}")
        sampling_params = SamplingParams(
            n=n,
            # <jingzhi> change to greedy sampling to check correctness.
            temperature=0.0 if use_beam_search else temperature, # 0 or 1e-6 (greedy), #1.0
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=ignore_eos, # False, # True (original),
            max_tokens=output_len if ignore_eos else (llm.llm_engine.model_config.max_model_len-_),
            # max_tokens=llm.llm_engine.model_config.max_model_len-_ # 4096-_  # output_len, #TODO(jingzhi) test when using max tokens
        )

        # we need to apply chat template if possible
        if tokenizer_obj.chat_template != None:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)


        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )
        print(f"output_len {output_len, (llm.llm_engine.model_config.max_model_len-_)}")

    print(f"finish adding requests")

    start = time.perf_counter()
    # FIXME(woosuk): Do not use internal method.
    outputs = llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    
    print(f"outputs:\n")
    print(f"this execution plan running time: {end - start}s ---abs {end}")
    # print(f"output_lens = {[[len(req_output.prompt_token_ids), len(completion_output.token_ids), output_len] for req_output, (_, _, output_len) in zip(outputs, requests) for completion_output in req_output.outputs]}")
    print(f"output_lens = {[[len(req_output.prompt_token_ids), len(completion_output.token_ids), -1] for req_output in outputs for completion_output in req_output.outputs]}")
    print(f"tot_inp_lens = {sum([len(req_output.prompt_token_ids) for req_output in outputs])}")
    print(f"tot_out_len = {sum([len(completion_output.token_ids) for req_output in outputs for completion_output in req_output.outputs])}")


    # for req_output in outputs:
    #     for completion_output in req_output.outputs:
    #         print(req_output.request_id, req_output.prompt_token_ids[:10], completion_output.token_ids)

    return end - start











# <jingzhi> run_vllm_with_preemption
def run_vllm_v0826_no_data_parallel(
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
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    
    # <jingzhi>
    gpu_memory_utilization: float,
    temperature: float,
    ignore_eos: bool,
) -> float:
    
    # <jingzhi> test shared array
    from vllm.core.multimodel_scheduler import SHARED_CONTECT
    # SHARED_CONTECT.test_and_print()
    # SHARED_CONTECT.test_task()
    # set the status to inform the multi-model scheduler that the prepartion is finished
    SHARED_CONTECT.set_finish_preparation_before_init_LLM()
    # wait for the model to be started
    SHARED_CONTECT.wait_to_be_started()
     

    from vllm import LLM, SamplingParams

    start = None
    rescheduled_iter_num = -1 # how many times this model has been rescheduled on the machine
    llm = None
    while (not SHARED_CONTECT.is_finished()):

        rescheduled_iter_num += 1

        # <jingzhi> For Profiling
        start_before_prepare_model = time.perf_counter()


        print(f"waiting----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)

        if rescheduled_iter_num > 0:
            # need wait for the signal to start model loading
            # SHARED_CONTECT.sync_before_loading_model()
            SHARED_CONTECT.wait_to_be_started()

        print(f"finish waiting----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)
        
        # update the execution plan
        tensor_parallel_size, gpu_memory_utilization = SHARED_CONTECT.update_execution_plan(
            tensor_parallel_size, gpu_memory_utilization)


        print(f"loading LLM  tensor_parallel_size {tensor_parallel_size} gpu_memory_utilization {gpu_memory_utilization}----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)


        # <jingzhi> For Profiling
        start_prepare_model = time.perf_counter()
        print(f"total time before preparing model: {start_prepare_model-start_before_prepare_model}s ---abs {start_prepare_model}")

        if (rescheduled_iter_num == 0) or (os.environ['SOFT_RESCHEDULE'] == 'False'):
            start_time_load_LLM = time.perf_counter()
            llm = LLM(
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                kv_cache_dtype=kv_cache_dtype,
                device=device,
                # <jingzhi>
                # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
                # max_num_seqs=2048,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=512,
                max_paddings=512,
            )
            end_time_load_LLM = time.perf_counter()
            print(f"total time to load LLM: {end_time_load_LLM - start_time_load_LLM}")
        else:
            # update llm instead of initialize a new one
            start_time_load_LLM = time.perf_counter()
            llm.update_llm_engine(
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                kv_cache_dtype=kv_cache_dtype,
                device=device,
                # <jingzhi>
                # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
                # max_num_seqs=2048,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=512,
                max_paddings=512,
            )
            end_time_load_LLM = time.perf_counter()
            print(f"total time to update LLM: {end_time_load_LLM - start_time_load_LLM}")


        if rescheduled_iter_num == 0:

            # <jingzhi>
            print(f"max_model_len: {llm.llm_engine.model_config.max_model_len}")
            print(f"temperature: {temperature}")
            print(f"ignore_eos: {ignore_eos}")

            # Add the requests to the engine.
            for prompt, _, output_len in requests:
                # print(f"in len: {_}, out len: {output_len} vs {4096-_}")
                sampling_params = SamplingParams(
                    n=n,
                    # <jingzhi> change to greedy sampling to check correctness.
                    temperature=0.0 if use_beam_search else temperature, # 0 or 1e-6 (greedy), #1.0
                    top_p=1.0,
                    use_beam_search=use_beam_search,
                    ignore_eos=ignore_eos, # False, # True (original),
                    max_tokens=output_len if ignore_eos else (llm.llm_engine.model_config.max_model_len-_),
                    # max_tokens=llm.llm_engine.model_config.max_model_len-_ # 4096-_  # output_len, #TODO(jingzhi) test when using max tokens
                )
                # FIXME(woosuk): Do not use internal method.
                llm._add_request(
                    prompt=prompt,
                    prompt_token_ids=None,
                    sampling_params=sampling_params,
                )
        else:
            # directly set the remaining requests to the scheduler waiting list
            print(f"directly using unfinished requests!")
            llm.llm_engine.scheduler.waiting = SHARED_CONTECT.remaining_requests


        # <jingzhi> For Profiling
        end_prepare_model = time.perf_counter()
        print(f"total time to prepare model: {end_prepare_model-start_prepare_model}s ---abs {end_prepare_model}", flush=True)


        
        # TODO (jingzhi) because we allow model preemption here, we may adjust this later
        if start == None:
            start = time.perf_counter()

        tmp_start = time.perf_counter()

        # FIXME(woosuk): Do not use internal method.
        outputs = llm._run_engine(use_tqdm=True)
        end = time.perf_counter()
        

        print(f"this execution plan running time: {end - tmp_start}s ---abs {end}")
        print(f"outputs:\n")
        print(f"output_lens = {[[len(req_output.prompt_token_ids), len(completion_output.token_ids), output_len] for req_output, (_, _, output_len) in zip(outputs, requests) for completion_output in req_output.outputs]}")
        print(f"tot_inp_lens = {sum([len(req_output.prompt_token_ids) for req_output in outputs])}")
        print(f"tot_out_len = {sum([len(completion_output.token_ids) for req_output in outputs for completion_output in req_output.outputs])}")
        # for req_output, (_, _, output_len) in zip(outputs, requests):
        #     for completion_output in req_output.outputs:
        #         print(req_output.request_id, len(req_output.prompt_token_ids), len(completion_output.token_ids), len(req_output.prompt_token_ids)+len(completion_output.token_ids), len(req_output.prompt_token_ids)+output_len)
        #         print('Q---------------------------------------')
        #         print(req_output.prompt)
        #         print('A---------------------------------------')
        #         print(completion_output.text)



        # TODO (jingzhi) release the resources of the current execution plan
        print(f"deleting LLM-----------------", flush=True)
        # destroy_model_parallel()
        # del llm
        # gc.collect()
        # torch.cuda.empty_cache()

        print(f"SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)
        print(f"{model} One round finished!!!!!!!!!!")


        # print status we collect
        # my_throughput_logger = llm.llm_engine.driver_worker.model_runner.my_throughput_logger
        # my_throughput_logger.cal_throughput()
        # my_throughput_logger.print_by_record()
    
    print(f"{model} Finally finished!!!!!!!!!!")

    return end - start







# <jingzhi> run_vllm_with_preemption
# <jingzhi> support data parallel:
#    if the degree of data parallel is k, then
#    the main process processes one group of data
#    the other k-1 worker processes process one group of data each
# NOTE: 1. using ray works for data parallel, but does not work for multi-level model system 
#       as ray actor cannot be passed to a multiprocessing subprocess.
# 
# def run_vllm_using_ray(
#     requests: List[Tuple[str, int, int]],
#     model: str,
#     tokenizer: str,
#     quantization: Optional[str],
#     tensor_parallel_size: int,
#     seed: int,
#     n: int,
#     use_beam_search: bool,
#     trust_remote_code: bool,
#     dtype: str,
#     max_model_len: Optional[int],
#     enforce_eager: bool,
#     kv_cache_dtype: str,
#     device: str,
    
#     # <jingzhi>
#     gpu_memory_utilization: float,
#     temperature: float,
#     ignore_eos: bool,
#     fixed_output_len: int,
# ) -> float:

#     # <jingzhi> test shared array
#     from vllm.core.multimodel_scheduler import SHARED_CONTECT
#     # SHARED_CONTECT.test_and_print()
#     # SHARED_CONTECT.test_task()
#     # set the status to inform the multi-model scheduler that the prepartion is finished
#     SHARED_CONTECT.set_finish_preparation_before_init_LLM()
#     # wait for the model to be started
#     SHARED_CONTECT.wait_to_be_started()

#     # use this to pass message to data parallel ray workers
#     message_passer = REMOTE_LLM_INFER_WORKER_MESSAGE_PASSER.remote()
     

#     from vllm import LLM, SamplingParams

#     start = None
#     rescheduled_iter_num = -1 # how many times this model has been rescheduled on the machine
#     llm = None
#     remaining_outputs = None
#     while (not SHARED_CONTECT.is_finished()):

#         rescheduled_iter_num += 1

#         # <jingzhi> For Profiling
#         start_before_prepare_model = time.perf_counter()


#         print(f"waiting----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)

#         if rescheduled_iter_num > 0:
#             # need wait for the signal to start model loading
#             # SHARED_CONTECT.sync_before_loading_model()
#             SHARED_CONTECT.wait_to_be_started()

#         print(f"finish waiting----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)
        

#         # TODO (jingzhi) because we allow model preemption here, we may adjust this later
#         if start == None:
#             start = time.perf_counter()
        
#         # <jingzhi> For Profiling
#         start_prepare_model = time.perf_counter()
#         print(f"total time before preparing model: {start_prepare_model-start_before_prepare_model}s ---abs {start_prepare_model}")


#         # run the inference after receiving the start signal from the main process
#         # update the execution plan
#         tensor_parallel_size, gpu_memory_utilization = SHARED_CONTECT.update_execution_plan(
#             tensor_parallel_size, gpu_memory_utilization)


#         print(f"loading LLM  tensor_parallel_size {tensor_parallel_size} gpu_memory_utilization {gpu_memory_utilization}----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)
        
#         infer_args = [
#             # the parameters of LLM engine--------------
#             model,
#             tokenizer,
#             quantization,
#             tensor_parallel_size,
#             seed,
#             n,
#             use_beam_search,
#             trust_remote_code,
#             dtype,
#             max_model_len,
#             enforce_eager,
#             kv_cache_dtype,
#             device,

#             # <jingzhi>
#             gpu_memory_utilization,
#             temperature,
#             ignore_eos,
#             fixed_output_len
#         ]

#         # ==================================================================================
#         # ==================================================================================
#         # ==================================================================================
#         # ==================================================================================
#         # DEAL WITH DATA PARALLELISM

#         # prepare requests (reorder them for dp workers if necessary)
#         dp_size = SHARED_CONTECT.get_dp_size()
#         print(f"dp_size: {dp_size}", flush=True)
#         if rescheduled_iter_num > 0:
#             # except the first round, always make requests pointing to remaining_requests
#             if dp_size == 1:
#                 requests = [deque(remaining_outputs)]
#             else:
#                 requests = list(remaining_outputs)
#                 tmp_requests = list()
#                 for dp_i in range(dp_size):
#                     tmp_requests.append(deque(requests[dp_i::dp_size]))
#                 requests = tmp_requests
#         else:
#             if dp_size > 1:
#                 tmp_requests = list()
#                 for dp_i in range(dp_size):
#                     tmp_requests.append(requests[dp_i::dp_size])
#                 requests = tmp_requests
#             else:
#                 requests = [requests]


#         # initialize inference workers
#         ray_dp_workers: List[REMOTE_LLM_INFER_WORKER] = list()
#         main_dp_worker = LLM_INFER_WORKER(0, requests[0], message_passer)
#         if dp_size > 1:
#             ray_dp_workers = [REMOTE_LLM_INFER_WORKER.options(
#                                 # TODO <jingzhi> support changing env variables as we cannot init ray twice where the env variables in actors are initialized
#                                 runtime_env={"env_vars": dict(os.environ)},
#                             ).remote(
#                                 SHARED_CONTECT.shared_id, worker_i,
#                                 requests[worker_i], message_passer,
#                                 os.environ["CUDA_VISIBLE_DEVICES"],
#                                 SHARED_CONTECT.communicator,
#                             ) 
#                             for worker_i in range(1, dp_size)]
        

#         # reset the state about the model's complete inp pool in the system communicator
#         SHARED_CONTECT.communicator.reset_state_for_model(SHARED_CONTECT.shared_id, dp_size)

#         # make each worker run the inference
#         ray_dp_worker_outputs = [worker.execute_method.remote("do_inference", *infer_args) for worker in ray_dp_workers]
#         main_output = main_dp_worker.do_inference(*infer_args)
#         if dp_size > 1:
#             ray_dp_worker_outputs = ray.get(ray_dp_worker_outputs)
#             # now we can mark the finish status in SHARED_CONTECT.prepare_for_reschedule
#             # first check whether this model is finished
#             is_finished = (len(main_output[1]) \
#                            + sum([len(worker_output[1]) for worker_output in ray_dp_worker_outputs])) == 0
#             SHARED_CONTECT.set_finished(is_finished)
#             # then mark the preparation_for_reschedule process as finished
#             SHARED_CONTECT.set_finish_preparation_for_reschedule()
#             # clear the dp message passer status
#             message_passer.clear.remote()


#         print(f"obtain all outputs in this round in main dp actor  ---abs {time.perf_counter()}")

#         # reorganize all generated outputs
#         gened_outputs = main_output[0]
#         for worker_output in ray_dp_worker_outputs:
#             gened_outputs.extend(worker_output[0])
#         # resort the remaining requests
#         remaining_outputs = main_output[1]
#         for worker_output in ray_dp_worker_outputs:
#             remaining_outputs.extend(worker_output[1])
        
#         # <jingzhi> NOTE: in model-level pipeline, we sort the remaining outputs by their req ids
#         remaining_outputs = sorted(remaining_outputs, key=lambda seq_group: seq_group.request_id)
#         # if dp_size > 1:
#         #     remaining_outputs = sorted(remaining_outputs, key=lambda seq_group: len(seq_group.prompt_token_ids))
        
#         print(f"reorganize all remaining requests ---abs {time.perf_counter()}")

#         # kill remote llm inference workers if any
#         for worker in ray_dp_workers:
#             ray.kill(worker)


#         print(f"kill all dp actors if any ---abs {time.perf_counter()}")
        
#         # ray worker应该返回什么request序列呢？返回他们的finished的request和remaining request，
#         # 先暂时naive地每次都一定返回，之后再根据是否dp_size发生了改变决定是否当前轮次要返回。
#         # 还需要kill ray worker，也可以留到判断下一个dp_size和当前dp_size是否相同之后再决定要不要kill


#     end = time.perf_counter()
#     print(f"{model} Finally finished!!!!!!!!!!")

#     return end - start











# <jingzhi> run_vllm_with_preemption
# <jingzhi> support data parallel:
#    if the degree of data parallel is k, then
#    the main process processes one group of data
#    the other k-1 worker processes process one group of data each
# <jingzhi> support multi-level model system
# NOTE: we use multiprocessing to launch subprocess for dp workers
# 
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
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    
    # <jingzhi>
    gpu_memory_utilization: float,
    temperature: float,
    ignore_eos: bool,
    fixed_output_len: int,
) -> float:


    # <jingzhi> test shared array
    from vllm.core.multimodel_scheduler import SHARED_CONTECT


    # SHARED_CONTECT.test_and_print()
    # SHARED_CONTECT.test_task()
    # set the status to inform the multi-model scheduler that the prepartion is finished
    SHARED_CONTECT.set_finish_preparation_before_init_LLM()
    # wait for the model to be started
    SHARED_CONTECT.wait_to_be_started()



    # <jingzhi> use multiprocessing
    from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED, wait

    # # use this to pass message to data parallel ray workers
    # message_passer = REMOTE_LLM_INFER_WORKER_MESSAGE_PASSER.remote()
     

    from vllm import LLM, SamplingParams

    start = None
    rescheduled_iter_num = -1 # how many times this model has been rescheduled on the machine
    llm = None
    remaining_outputs = None
    while (not SHARED_CONTECT.is_finished()):

        rescheduled_iter_num += 1

        # <jingzhi> For Profiling
        start_before_prepare_model = time.perf_counter()


        print(f"waiting-----------------: shared_id {SHARED_CONTECT.shared_id} rescheduled_iter_num: {rescheduled_iter_num}", flush=True)

        if rescheduled_iter_num > 0:
            # need wait for the signal to start model loading
            # SHARED_CONTECT.sync_before_loading_model()
            SHARED_CONTECT.wait_to_be_started()

        print(f"finish waiting-----------------: shared_id {SHARED_CONTECT.shared_id}  rescheduled_iter_num: {rescheduled_iter_num}", flush=True)

        # TODO (jingzhi) because we allow model preemption here, we may adjust this later
        if start == None:
            start = time.perf_counter()
        
        # <jingzhi> For Profiling
        start_prepare_model = time.perf_counter()
        print(f"total time before preparing model: shared_id {SHARED_CONTECT.shared_id}: {start_prepare_model-start_before_prepare_model}s ---abs {start_prepare_model}")



        # run the inference after receiving the start signal from the main process
        # update the execution plan
        tensor_parallel_size, gpu_memory_utilization = SHARED_CONTECT.update_execution_plan(
            tensor_parallel_size, gpu_memory_utilization)



        print(f"loading LLM: shared_id {SHARED_CONTECT.shared_id}  tensor_parallel_size {tensor_parallel_size} gpu_memory_utilization {gpu_memory_utilization}----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)
        
        infer_args = [
            # the parameters of LLM engine--------------
            model,
            tokenizer,
            quantization,
            tensor_parallel_size,
            seed,
            n,
            use_beam_search,
            trust_remote_code,
            dtype,
            max_model_len,
            enforce_eager,
            kv_cache_dtype,
            device,

            # <jingzhi>
            gpu_memory_utilization,
            temperature,
            ignore_eos,
            fixed_output_len
        ]

        # ==================================================================================
        # ==================================================================================
        # ==================================================================================
        # ==================================================================================
        # DEAL WITH DATA PARALLELISM


        # prepare requests (reorder them for dp workers if necessary)
        dp_size = SHARED_CONTECT.get_dp_size()
        print(f"shared_id {SHARED_CONTECT.shared_id} dp_size: {dp_size}", flush=True)
        if rescheduled_iter_num > 0:
            # except the first round, always make requests pointing to remaining_requests
            if dp_size == 1:
                requests = [deque(remaining_outputs)]
            else:
                requests = list(remaining_outputs)
                tmp_requests = list()
                for dp_i in range(dp_size):
                    tmp_requests.append(deque(requests[dp_i::dp_size]))
                requests = tmp_requests
        else:
            if dp_size > 1:
                tmp_requests = list()
                for dp_i in range(dp_size):
                    tmp_requests.append(requests[dp_i::dp_size])
                requests = tmp_requests
            else:
                requests = [requests]


        # reset the state about the model's complete inp pool in the system communicator
        SHARED_CONTECT.communicator.reset_state_for_model(SHARED_CONTECT.shared_id, dp_size)

        # launch multiprocessing subprocess for dp workers------------------------------------------------------------
        print(f"start doing inference: shared_id {SHARED_CONTECT.shared_id}-------------\n")
        
        
        executor = None
        futures = None
        ray_dp_worker_outputs = list()
        # if dp_size > 1:
        #     executor = ProcessPoolExecutor(max_workers=dp_size-1)
        #     futures = [executor.submit(my_llm_infer_worker_multiprocessing.do_inference, \
        #                             worker_i, requests[worker_i], *infer_args) \
        #                                 for worker_i in range(1, dp_size)]
        
        #     print(f"finish launching dp processes-------------\n")

        # if dp_size > 1:
        # NOTE: we will launch the subprocesses for each dp worker, even when there is only 1 dp worker, 
        # to avoid re-init cuda environment
        executor = ProcessPoolExecutor(max_workers=dp_size)
        futures = [executor.submit(my_llm_infer_worker_multiprocessing.do_inference, \
                                worker_i, requests[worker_i], *infer_args) \
                                    for worker_i in range(dp_size)]
    
        print(f"finish launching dp processes: shared_id {SHARED_CONTECT.shared_id} -------------\n")
        
        # run the inference of the main dp worker
        # main_output = my_llm_infer_worker_multiprocessing.do_inference(0, requests[0], *infer_args)
        # print(f"finish main dp processes-------------\n")
        # if dp_size > 1:
        #     done, not_done = wait(futures, return_when=ALL_COMPLETED)
        #     ray_dp_worker_outputs = [future.result() for future in done]

        #     print(f"finish fetching dp worker results-------------\n")

        #     # now we can mark the finish status in SHARED_CONTECT.prepare_for_reschedule
        #     # first check whether this model is finished
        #     # is_finished = (len(main_output[1]) \
        #     #                + sum([len(worker_output[1]) for worker_output in ray_dp_worker_outputs])) == 0
        #     gened_output_num = (main_output[0] + \
        #                    sum([worker_output[0] for worker_output in ray_dp_worker_outputs]))
            
        #     print(f"gened_output_num: {gened_output_num}-------------\n")

        #     SHARED_CONTECT.set_finished(gened_output_num, set_event_state_anyway=True)
        #     # # then mark the preparation_for_reschedule process as finished ==> mark it when calling ``set_finished``
        #     # SHARED_CONTECT.set_finish_preparation_for_reschedule()


        # if dp_size > 1:
        done, not_done = wait(futures, return_when=ALL_COMPLETED)
        ray_dp_worker_outputs = [future.result() for future in done]

        print(f"finish fetching dp worker results: shared_id {SHARED_CONTECT.shared_id} -------------\n")

        # now we can mark the finish status in SHARED_CONTECT.prepare_for_reschedule
        # first check whether this model is finished
        # is_finished = (len(main_output[1]) \
        #                + sum([len(worker_output[1]) for worker_output in ray_dp_worker_outputs])) == 0
        gened_output_num = sum([worker_output[0] for worker_output in ray_dp_worker_outputs])
        
        print(f"gened_output_num: shared_id {SHARED_CONTECT.shared_id}: {gened_output_num} -------------\n")

        # must update the remaining req num here in the main process, 
        # s.t., the next round dp workers can have correct reamining req num to use
        SHARED_CONTECT.set_finished(gened_output_num)#, set_event_state_anyway=True)
        # # then mark the preparation_for_reschedule process as finished ==> mark it when calling ``set_finished``
        SHARED_CONTECT.set_finish_preparation_for_reschedule()
        

        print(f"obtain all outputs in this round in main dp actor: shared_id {SHARED_CONTECT.shared_id}  ---abs {time.perf_counter()}")
        
        
        # reorganize all generated outputs
        # gened_outputs = main_output[0]
        # for worker_output in ray_dp_worker_outputs:
        #     gened_outputs.extend(worker_output[0])
        # resort the remaining requests
        # remaining_outputs = main_output[1]
        remaining_outputs = list()
        for worker_output in ray_dp_worker_outputs:
            remaining_outputs.extend(worker_output[1])
        
        # <jingzhi> NOTE: in model-level pipeline, we sort the remaining outputs by their req ids
        remaining_outputs = sorted(remaining_outputs, key=lambda seq_group: seq_group.request_id)
        # if dp_size > 1:
        #     remaining_outputs = sorted(remaining_outputs, key=lambda seq_group: len(seq_group.prompt_token_ids))
        
        print(f"reorganize all remaining requests: shared_id {SHARED_CONTECT.shared_id} ---abs {time.perf_counter()}")

        # kill remote llm inference workers if any
        if dp_size > 1:
            executor.shutdown()


        print(f"kill all dp actors if any: shared_id {SHARED_CONTECT.shared_id} ---abs {time.perf_counter()}")
        
        # ray worker应该返回什么request序列呢？返回他们的finished的request和remaining request，
        # 先暂时naive地每次都一定返回，之后再根据是否dp_size发生了改变决定是否当前轮次要返回。
        # 还需要kill ray worker，也可以留到判断下一个dp_size和当前dp_size是否相同之后再决定要不要kill



    end = time.perf_counter()
    print(f"{model} Finally finished!!!!!!!!!!")

    return end - start



















# <jingzhi> run_vllm_with_preemption
# TODO (jingzhi): this function is used for DEBUG
def run_vllm_for_debug(
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
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    
    # <jingzhi>
    gpu_memory_utilization: float,
) -> float:
    
    # <jingzhi> test shared array
    from vllm.core.multimodel_scheduler import SHARED_CONTECT
    # SHARED_CONTECT.test_and_print()
    # SHARED_CONTECT.test_task()
     

    from vllm import LLM, SamplingParams

    start = None
    rescheduled_iter_num = -1 # how many times this model has been rescheduled on the machine
    llm = None
    while (tensor_parallel_size < 4):

        rescheduled_iter_num += 1

        # <jingzhi> For Profiling
        start_before_prepare_model = time.perf_counter()


        print(f"waiting----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)

        # if rescheduled_iter_num > 0:
        #     # need wait for the signal to start model loading
        #     SHARED_CONTECT.sync_before_loading_model()

        print(f"finish waiting----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)
        
        # update the execution plan
        # tensor_parallel_size, gpu_memory_utilization = SHARED_CONTECT.update_execution_plan(
        #     tensor_parallel_size, gpu_memory_utilization)

        if rescheduled_iter_num > 0:
            tensor_parallel_size, gpu_memory_utilization = 4, 0.9
            os.environ['WEIGHT_LOAD_DEGREE'] = '2'


        print(f"loading LLM  tensor_parallel_size {tensor_parallel_size} gpu_memory_utilization {gpu_memory_utilization}----------------- rescheduled_iter_num: {rescheduled_iter_num}", flush=True)


        # <jingzhi> For Profiling
        start_prepare_model = time.perf_counter()
        print(f"total time before preparing model: {start_prepare_model-start_before_prepare_model}s ---abs {start_prepare_model}")

        if (rescheduled_iter_num == 0) or (os.environ['SOFT_RESCHEDULE'] == 'False'):
            llm = LLM(
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                kv_cache_dtype=kv_cache_dtype,
                device=device,
                # <jingzhi>
                # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
                # max_num_seqs=2048,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=512,
                max_paddings=512,
            )
        else:
            # update llm instead of initialize a new one
            llm.update_llm_engine(
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                kv_cache_dtype=kv_cache_dtype,
                device=device,
                # <jingzhi>
                # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
                # max_num_seqs=2048,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=512,
                max_paddings=512,
            )


        if rescheduled_iter_num == 0:
            # Add the requests to the engine.
            for prompt, _, output_len in requests:
                sampling_params = SamplingParams(
                    n=n,
                    # <jingzhi> change to greedy sampling to check correctness.
                    temperature=0.0 if use_beam_search else 1e-6, #1.0
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
        else:
            # directly set the remaining requests to the scheduler waiting list
            print(f"directly using unfinished requests!")
            llm.llm_engine.scheduler.waiting = SHARED_CONTECT.remaining_requests


        # <jingzhi> For Profiling
        end_prepare_model = time.perf_counter()
        print(f"total time to prepare model: {end_prepare_model-start_prepare_model}s ---abs {end_prepare_model}", flush=True)


        
        # TODO (jingzhi) because we allow model preemption here, we may adjust this later
        if start == None:
            start = time.perf_counter()

        tmp_start = time.perf_counter()

        # FIXME(woosuk): Do not use internal method.
        outputs = llm._run_engine(use_tqdm=True)
        end = time.perf_counter()
        

        print(f"this execution plan running time: {end - tmp_start}s ---abs {end}")
        print(f"outputs:\n")
        # for req_output in outputs:
        #     for completion_output in req_output.outputs:
        #         print(req_output.request_id, req_output.prompt_token_ids[:10], completion_output.token_ids)



        # TODO (jingzhi) release the resources of the current execution plan
        print(f"deleting LLM-----------------", flush=True)
        # destroy_model_parallel()
        # del llm
        # gc.collect()
        # torch.cuda.empty_cache()

        print(f"SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)


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
        # <jingzhi>
        print(f"do sample: {not use_beam_search}")
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample= False, #not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        gened_strs = tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)

        # <jingzhi>
        print(f"output_lens: {[(prompt_len, output_len, req_output.shape, prompt_len+output_len) for req_output, gend_str in zip (llm_outputs, gened_strs)]}") 
        for str1, str2 in zip(batch, gened_strs):
            print('Q---------------------------------------')
            print(str1)
            print('A---------------------------------------')
            print(str2[len(str1):])        

        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import pipeline
    llm = pipeline(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    print(args)

    # <jingzhi> For Profiling
    start_main = time.perf_counter()

    print(f"\nTIMESTAMP start_main: {start_main}\n")

    # <jingzhi> deal with extra parameters
    os.environ['WEIGHT_LOAD_DEGREE'] = args.weight_load_degree



    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    # <jingzhi> For Profiling
    print(f"finish get tokenizer ---abs: {time.perf_counter()}")

    # <jingzhi> support multi-level model system
    if os.getenv("GET_INP_FROM_COMMUNICATOR", "False") == 'True':
        requests = []
    else:
        if args.dataset is None:
            # Synthesize a prompt with the given input length.
            prompt = "hi" * (args.input_len - 1)
            requests = [(prompt, args.input_len, args.output_len)
                        for _ in range(args.num_prompts)]
        else:
            requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                    args.output_len)


    # <jingzhi> For Profiling
    print(f"finish request sampling ---abs: {time.perf_counter()}")


    if args.backend == "vllm":
        elapsed_time = run_vllm(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype,
                                args.max_model_len, args.enforce_eager,
                                args.kv_cache_dtype, args.device,
                                # <jingzhi> add more control
                                args.gpu_use_ratio,
                                args.temperature,
                                args.ignore_eos,
                                # <jingzhi> support multi-level model system
                                args.output_len,
                                )
    elif args.backend == 'vllm_ori':
        elapsed_time = run_vllm_ori(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype,
                                args.max_model_len, args.enforce_eager,
                                args.kv_cache_dtype, args.device,
                                # <jingzhi> add more control
                                args.gpu_use_ratio,
                                args.temperature,
                                args.ignore_eos,
                                # <jingzhi> support multi-level model system
                                args.output_len,
                                )        
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s",
        # <jingzhi> flush print
          flush=True)
    
    # <jingzhi> For Profiling
    end_main = time.perf_counter()
    print(f"TOT TIME TO RUN MAIN(): {end_main - start_main}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii", "vllm_ori"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
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
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    


    # <jingzhi> deal with extra parameters
    parser.add_argument(
        "--weight-load-degree", "-wldegree", 
        type=str,
        default="16",
        help='weight load degree when cache model weights on other gpus.')


    parser.add_argument(
        "--gpu-use-ratio", "-gpuratio", 
        type=float,
        default="0.9",
        help='gpu utilization ratio.')    

    parser.add_argument(
        "--temperature", 
        type=float,
        default="1.0",
        help='temperature.')    




    parser.add_argument(
        "--ignore-eos", 
        action='store_true',
        help='whether to ignore eos token or not during inference.')




    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend in ["vllm", "vllm_ori"]:
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
