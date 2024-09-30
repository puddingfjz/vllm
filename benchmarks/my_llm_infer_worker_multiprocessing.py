""" 
This file is about llm inference workers to support data parallelism.
Each llm inference worker deals with a part of the requests. 
We support data parallelism + tensor parallelism.
Each worker process is the driver process is it further does tensor parallelism.
"""

from typing import List, Optional, Tuple
from collections import deque
# from vllm.engine.ray_utils import ray
import time
import os
from vllm.core.multimodel_scheduler import SHARED_CONTECT, LLM_COMMUNICATOR


   
def do_inference(
        worker_i: int, 
        remaining_requests, # input requests or remaining seqgroups

        # 
        # the parameters of LLM engine--------------
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
        fixed_output_len: int
    ):

    from vllm import LLM, SamplingParams

    # ------------------------------------------------------------------------------------------------- 
    rescheduled_iter_num = 0
    
    # store the message passer in SHARED_CONTECT
    # SHARED_CONTECT.message_passer_for_dp = message_passer
    SHARED_CONTECT.dp_id = worker_i

    os.environ['DP_WORKER_I'] = str(worker_i)
    # -------------------------------------------------------------------------------------------------

    # we need modify os.environ['TOT_ORDERED_GPUS']
    gpus = os.environ['TOT_ORDERED_GPUS'].split(',')
    gpus = gpus[worker_i*tensor_parallel_size:] + gpus[:worker_i*tensor_parallel_size]
    gpus = ','.join([str(i) for i in gpus])
    os.environ['TOT_ORDERED_GPUS'] = gpus

    # print(f"os.environ['TOT_ORDERED_GPUS']: {os.environ['TOT_ORDERED_GPUS']}")

    # <jingzhi> For Profiling
    start_prepare_model = time.perf_counter()
    # print(f"total time before preparing model: {start_prepare_model-start_before_prepare_model}s ---abs {start_prepare_model}")
    print(f"start do_inference: ---abs {start_prepare_model}")

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


    # if rescheduled_iter_num == 0:
    # support multi-level model system:
    # 1. when the model is started for the first time, it fetches input from the model communicator directly;
    # 2. it is possible that when the model is restarted there is no remaining requests.
    if (len(remaining_requests)>0) and isinstance(remaining_requests[0], tuple):
        # we have not transform the input requests to seqgroups

        # <jingzhi>
        print(f"max_model_len: {llm.llm_engine.model_config.max_model_len}")
        print(f"temperature: {temperature}")
        print(f"ignore_eos: {ignore_eos}")

        # Add the requests to the engine.
        for prompt, _, output_len in remaining_requests:
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
        # llm.llm_engine.scheduler.waiting = SHARED_CONTECT.remaining_requests
        llm.llm_engine.scheduler.waiting = remaining_requests


    # prepare sampling parameters
    sampling_parameters = {                    
        "n":n,
        # <jingzhi> change to greedy sampling to check correctness.
        "temperature":0.0 if use_beam_search else temperature, # 0 or 1e-6 (greedy), #1.0
        "top_p":1.0,
        "use_beam_search":use_beam_search,
        "ignore_eos":ignore_eos, # False, # True (original),
        "max_tokens":fixed_output_len if ignore_eos else (llm.llm_engine.model_config.max_model_len)}


    # <jingzhi> For Profiling
    end_prepare_model = time.perf_counter()
    print(f"total time to prepare model: {end_prepare_model-start_prepare_model}s ---abs {end_prepare_model}", flush=True)


    
    # TODO (jingzhi) because we allow model preemption here, we may adjust this later
    # if start == None:
    #     start = time.perf_counter()

    tmp_start = time.perf_counter()

    # FIXME(woosuk): Do not use internal method.
    outputs = llm._run_engine(use_tqdm=True, sampling_parameters=sampling_parameters)
    end = time.perf_counter()
    

    print(f"this execution plan running time: {end - tmp_start}s ---abs {end}")
    print(f"outputs:\n")
    # print(f"output_lens = {[[len(req_output.prompt_token_ids), len(completion_output.token_ids), output_len] for req_output, (_, _, output_len) in zip(outputs, requests) for completion_output in req_output.outputs]}")
    print(f"output_lens = {[[len(req_output.prompt_token_ids), len(completion_output.token_ids), -1] for req_output in outputs for completion_output in req_output.outputs]}")
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

    # return request status
    print(f"worker i: {worker_i}, len of outputs and remainings before return: {len(SHARED_CONTECT.gened_outputs), len(SHARED_CONTECT.remaining_requests)}")
    # return (SHARED_CONTECT.gened_outputs, SHARED_CONTECT.remaining_requests)
    return len(SHARED_CONTECT.gened_outputs), SHARED_CONTECT.remaining_requests
    

