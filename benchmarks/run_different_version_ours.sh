# for pd in 40 20 16
# do
#     python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 1000 --enforce-eager -tp 2 -wldegree $pd -gpuratio 0.9 > ours_0226_70b_1_tp2_pd${pd}_gpu0.9.log
# done


# for gpur in 0.5 0.6 0.7 0.8 0.9
# do
#     for tp in 1 2
#     do
#         python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp $tp -wldegree 20 -gpuratio $gpur > ours_0226_13b_1_tp${tp}_pd20_gpu${gpur}.log
#     done
# done



# for gpur in 0.5 0.6 0.7 0.8 0.9
# do
#     for tp in 1 2
#     do
#         python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager -tp $tp -wldegree 16 -gpuratio $gpur > ours_0226_7b_1_tp${tp}_pd16_gpu${gpur}.log
#     done
# done




# for tp in 1 2 4
# do
#     python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager -tp $tp > vllm_0226_7b_1_tp${tp}.log
# done



# python3 my_bench_multimodel_throughput.py > ours_multimodel_0312_13b70b_10kreq_fast_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 10000 --enforce-eager -tp 4 > vllm_70b_tp4_0312_10kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 10000 --enforce-eager -tp 2 > vllm_70b_tp2_0312_10kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 10000 --enforce-eager -tp 4 > vllm_13b_tp4_0312_10kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 10000 --enforce-eager -tp 2 > vllm_13b_tp2_0312_10kreq_1.log



# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager > vllm_7b_tp1_0315_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager -tp 2 > vllm_7b_tp2_0315_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager -tp 4 > vllm_7b_tp4_0315_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 1000 --enforce-eager -tp 4 > vllm_70b_tp4_0315_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 4 > vllm_13b_tp4_0315_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 2 > vllm_13b_tp2_0315_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 1 > vllm_13b_tp1_0315_1kreq_1.log


# python3 my_bench_multimodel_throughput.py > ours_multimodel_0315_4x7b70b_2kreq_fast_soft_nopreempt_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 2000 --enforce-eager > vllm_7b_tp1_0315_2kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 2000 --enforce-eager -tp 2 > vllm_7b_tp2_0315_2kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 2000 --enforce-eager -tp 4 > vllm_7b_tp4_0315_2kreq_1.log


# python3 my_bench_multimodel_throughput.py > ours_multimodel_0315_4x7b70b_1kreq_fast_soft_nopreempt_2.log

# python3 my_bench_multimodel_throughput.py > ours_multimodel_0315_4x7b70b_2kreq_fast_soft_nopreempt_2.log



# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 2 > collect_output_lengths/vllm_7b_tp2_0328_1kreq_3.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 2 > collect_output_lengths/vllm_13b_tp2_0328_1kreq_2.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 1000 --enforce-eager -tp 2 > collect_output_lengths/vllm_70b_tp2_0328_1kreq_2.log

# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 30 --enforce-eager --backend hf --hf-max-batch-size 64 > collect_output_lengths/vllm_7b_tp1_0328_1kreq_1.log




# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model BAAI/Aquila-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Aquila-7B_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model BAAI/AquilaChat-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/AquilaChat-7B_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Baichuan2-13B-Chat_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Baichuan-7B_tp4_0328_1kreq_1.log




# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model THUDM/chatglm3-6b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/chatglm3-6b_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model CohereForAI/c4ai-command-r-v01 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/c4ai-command-r-v01_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model google/gemma-2b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/gemma-2b_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model google/gemma-7b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/gemma-7b_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model bigcode/starcoder --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/starcoder_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-j-6b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/gpt-j-6b_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-neox-20b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/gpt-neox-20b_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mistral-7B-v0.1 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Mistral-7B-v0.1_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mistral-7B-Instruct-v0.1 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Mistral-7B-Instruct-v0.1_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Mixtral-8x7B-v0.1_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model microsoft/phi-2 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/phi-2_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model Qwen/Qwen2-beta-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Qwen2-beta-7B_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model Qwen/Qwen2-beta-7B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/Qwen2-beta-7B-Chat_tp4_0328_1kreq_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model stabilityai/stablelm-base-alpha-7b-v2 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/stablelm-base-alpha-7b-v2_tp4_0328_1kreq_1.log




# python3 benchmark_throughput.py --dataset no_robot.parquet --model THUDM/chatglm3-6b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/chatglm3-6b_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model CohereForAI/c4ai-command-r-v01 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/c4ai-command-r-v01_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model google/gemma-2b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gemma-2b_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model google/gemma-7b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gemma-7b_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model bigcode/starcoder --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/starcoder_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model EleutherAI/gpt-j-6b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gpt-j-6b_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model EleutherAI/gpt-neox-20b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gpt-neox-20b_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model mistralai/Mistral-7B-v0.1 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Mistral-7B-v0.1_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model mistralai/Mistral-7B-Instruct-v0.1 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Mistral-7B-Instruct-v0.1_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Mixtral-8x7B-v0.1_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model microsoft/phi-2 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/phi-2_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model Qwen/Qwen2-beta-7B --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Qwen2-beta-7B_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model stabilityai/stablelm-base-alpha-7b-v2 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/stablelm-base-alpha-7b-v2_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model Qwen/Qwen2-beta-7B-Chat --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Qwen2-beta-7B-Chat_tp4_0331_10kreq_1.log


# python3 benchmark_throughput.py --dataset no_robot.parquet --model BAAI/Aquila-7B --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Aquila-7B_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model BAAI/AquilaChat-7B --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/AquilaChat-7B_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Baichuan2-13B-Chat_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan-7B --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Baichuan-7B_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/llama2_7b_tp4_0331_10kreq_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-chat-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/llama2_7b_chat_tp4_0331_10kreq_1.log




# change temperature
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.2> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_10kreq_temp0.2_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.4> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_10kreq_temp0.4_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.6> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_10kreq_temp0.6_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.8> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_10kreq_temp0.8_1.log

# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.2> collect_output_lengths/no_robot/Llama-2-7b-chat-hf_tp4_0331_1kreq_temp0.2_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.4> collect_output_lengths/no_robot/Llama-2-7b-chat-hf_tp4_0331_1kreq_temp0.4_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.6> collect_output_lengths/no_robot/Llama-2-7b-chat-hf_tp4_0331_1kreq_temp0.6_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.8> collect_output_lengths/no_robot/Llama-2-7b-chat-hf_tp4_0331_1kreq_temp0.8_1.log

# python3 benchmark_throughput.py --dataset no_robot.parquet --model BAAI/AquilaChat-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.2> collect_output_lengths/no_robot/AquilaChat-7B_tp4_0331_1kreq_temp0.2_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model BAAI/AquilaChat-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.4> collect_output_lengths/no_robot/AquilaChat-7B_tp4_0331_1kreq_temp0.4_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model BAAI/AquilaChat-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.6> collect_output_lengths/no_robot/AquilaChat-7B_tp4_0331_1kreq_temp0.6_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model BAAI/AquilaChat-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.8> collect_output_lengths/no_robot/AquilaChat-7B_tp4_0331_1kreq_temp0.8_1.log


# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.2> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_1kreq_temp0.2_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.4> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_1kreq_temp0.4_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.6> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_1kreq_temp0.6_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.8> collect_output_lengths/no_robot/Llama-2-7b-hf_tp4_0331_1kreq_temp0.8_1.log


# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.2> collect_output_lengths/no_robot/Baichuan2-13B-Chat_tp4_0331_1kreq_temp0.2_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.4> collect_output_lengths/no_robot/Baichuan2-13B-Chat_tp4_0331_1kreq_temp0.4_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.6> collect_output_lengths/no_robot/Baichuan2-13B-Chat_tp4_0331_1kreq_temp0.6_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.8> collect_output_lengths/no_robot/Baichuan2-13B-Chat_tp4_0331_1kreq_temp0.8_1.log


# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.2> collect_output_lengths/no_robot/Baichuan-7B_tp4_0331_1kreq_temp0.2_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.4> collect_output_lengths/no_robot/Baichuan-7B_tp4_0331_1kreq_temp0.4_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.6> collect_output_lengths/no_robot/Baichuan-7B_tp4_0331_1kreq_temp0.6_1.log
# python3 benchmark_throughput.py --dataset no_robot.parquet --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code --temperature 0.8> collect_output_lengths/no_robot/Baichuan-7B_tp4_0331_1kreq_temp0.8_1.log



# get cost model data by profiling
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp1_0416_temp0.2_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_2.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_6.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_0429_tp1_temp1.0_wldeg2.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_0430_tp1_temp1.0_wldeg2.log
# get cost model data by profiling for sampling part
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_sample1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_sample2.log
# get cost model data by profiling for prepare input part
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_prepInp1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_prepInp2.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_prepInp3.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_0503_tp1_temp1.0_wldeg2_samp_prepInp.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_0503_tp1_temp1.0_wldeg2_DecodePrepInp.log
# get cost model data for different exec plans by profiling [collect exec, sample, and prepInp cost together]
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp2_temp1.0_wldeg2_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp4_temp1.0_wldeg2_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 8 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp1_temp1.0_wldeg8_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 16 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp1_temp1.0_wldeg16_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 8 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp2_temp1.0_wldeg8_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 16 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_tp2_temp1.0_wldeg16_0504_1.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-13b-hf_tp1_temp1.0_wldeg2_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-13b-hf_tp2_temp1.0_wldeg2_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-13b-hf_tp4_temp1.0_wldeg2_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 20 --backend ours> Cost_Model_per_iter/Llama-2-13b-hf_tp1_temp1.0_wldeg20_0504_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 20 --backend ours> Cost_Model_per_iter/Llama-2-13b-hf_tp2_temp1.0_wldeg20_0504_1.log
# formal collection for more exec plans~~~~
python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-13b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code -wldegree 2 > collect_output_lengths/no_robot/Llama-2-13b-hf_tp4_0505_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model NousResearch/Llama-2-70b-hf --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code -wldegree 2 > collect_output_lengths/no_robot/Llama-2-70b-hf_tp4_0505_10kreq_1.log

python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-70b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp2_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-70b-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp4_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-70b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 40 --backend ours> Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp1_temp1.0_wldeg40_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-70b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 16 --backend ours> Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp2_temp1.0_wldeg16_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-70b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 20 --backend ours> Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp2_temp1.0_wldeg20_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-70b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 40 --backend ours> Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp2_temp1.0_wldeg40_0504_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_tp2_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_tp4_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 8 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_tp1_temp1.0_wldeg8_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 16 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_tp1_temp1.0_wldeg16_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 8 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_tp2_temp1.0_wldeg8_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 16 --backend ours> Cost_Model_per_iter/Formal_Llama-2-7b-hf_tp2_temp1.0_wldeg16_0504_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp1_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp2_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp4_temp1.0_wldeg2_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 10 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp1_temp1.0_wldeg10_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 20 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp1_temp1.0_wldeg20_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 10 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp2_temp1.0_wldeg10_0504_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-13b-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 20 --backend ours> Cost_Model_per_iter/Formal_Llama-2-13b-hf_tp2_temp1.0_wldeg20_0504_4.log
# 
# more models formal cost data collection ~
python3 construct_cost_model.py --input-len 16 --output-len 16 --model THUDM/chatglm3-6b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_chatglm3-6b_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model THUDM/chatglm3-6b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_chatglm3-6b_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model THUDM/chatglm3-6b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_chatglm3-6b_tp4_temp1.0_wldeg2_0518_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model EleutherAI/gpt-j-6b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_gpt-j-6b_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model EleutherAI/gpt-j-6b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_gpt-j-6b_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model EleutherAI/gpt-j-6b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_gpt-j-6b_tp4_temp1.0_wldeg2_0518_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model EleutherAI/gpt-neox-20b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_gpt-neox-20b_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model EleutherAI/gpt-neox-20b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_gpt-neox-20b_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model EleutherAI/gpt-neox-20b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_gpt-neox-20b_tp4_temp1.0_wldeg2_0518_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Llama-2-7b-chat-hf_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Llama-2-7b-chat-hf_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Llama-2-7b-chat-hf_tp4_temp1.0_wldeg2_0518_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Baichuan2-13B-Chat_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Baichuan2-13B-Chat_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Baichuan2-13B-Chat_tp4_temp1.0_wldeg2_0518_4.log
# 
python3 construct_cost_model.py --input-len 16 --output-len 16 --model baichuan-inc/Baichuan-7B --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Baichuan-7B_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model baichuan-inc/Baichuan-7B --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Baichuan-7B_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model baichuan-inc/Baichuan-7B --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Baichuan-7B_tp4_temp1.0_wldeg2_0518_4.log
# 
# python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Mixtral-8x7B-v0.1_tp1_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Mixtral-8x7B-v0.1_tp2_temp1.0_wldeg2_0518_4.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/Formal_Mixtral-8x7B-v0.1_tp4_temp1.0_wldeg2_0518_4.log




# profile with nsight system
/ssddata/jingzhi/Nsight_Systems_2023_2_1/target-linux-x64/nsys profile -w true -t cuda,nvtx,osrt -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop --cudabacktrace=true -x true -o ./nsys_profile/profile_per_iter_cost python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_3.log
/ssddata/jingzhi/Nsight_Systems_2023_2_1/target-linux-x64/nsys profile -w true -t cuda,nvtx,osrt -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop --cudabacktrace=true -x true -o ./nsys_profile/profile_per_iter_cost_prefill python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_4.log
/ssddata/jingzhi/Nsight_Systems_2023_2_1/target-linux-x64/nsys profile -w true -t cuda,nvtx,osrt -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop --cudabacktrace=true -x true -o ./nsys_profile/profile_per_iter_cost_prefill2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_5.log

# test sampling speed acceleration
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/testSample_Llama-2-7b-hf_0430_tp1_temp1.0_wldeg2.log
# check prepare input break down cost
python3 construct_cost_model.py --input-len 16 --output-len 16 --model NousResearch/Llama-2-7b-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/testPrepInp_Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2.log



# is ignore-eos false here wrong?
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> Cost_Model_per_iter/baseline_tp1_llama2_7b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> Cost_Model_per_iter/baseline_tp1_llama2_7b_2.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_llama2_7b_3.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> Cost_Model_per_iter/baseline_tp1_llama2_7b_4.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_llama2_7b_5.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> Cost_Model_per_iter/baseline_tp1_llama2_7b_6.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_llama2_7b_7.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model THUDM/chatglm3-6b --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_chatglm3-6b_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model THUDM/chatglm3-6b --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_chatglm3-6b_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model THUDM/chatglm3-6b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_chatglm3-6b_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-j-6b --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_gpt-j-6b_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-j-6b --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_gpt-j-6b_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-j-6b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_gpt-j-6b_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-neox-20b --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_gpt-neox-20b_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-neox-20b --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_gpt-neox-20b_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model EleutherAI/gpt-neox-20b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_gpt-neox-20b_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Llama-2-7b-chat-hf_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Llama-2-7b-chat-hf_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-chat-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Llama-2-7b-chat-hf_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Baichuan2-13B-Chat_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Baichuan2-13B-Chat_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan2-13B-Chat --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Baichuan2-13B-Chat_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Baichuan-7B_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Baichuan-7B_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model baichuan-inc/Baichuan-7B --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Baichuan-7B_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Mixtral-8x7B-v0.1_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Mixtral-8x7B-v0.1_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Mixtral-8x7B-v0.1_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Llama-2-70b-hf_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Llama-2-70b-hf_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-70b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Llama-2-70b-hf_1.log
# 
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Llama-2-7b-hf_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Llama-2-7b-hf_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Llama-2-7b-hf_1.log
# 
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp1_Llama-2-13b-hf_1.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp2_Llama-2-13b-hf_1.log
# python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-13b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 > Cost_Model_per_iter/baseline_tp4_Llama-2-13b-hf_1.log




# test search algorithms~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 search_exec_plans.py > test_search/testcase_5_llama2_7b_1_llama2_70b.log



# test end2end schedule performance~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 schedule_multi_model.py > test_end2end_schedule/test_1.log









# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# new round experiments
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model NousResearch/Llama-2-7b-hf --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_llama2_7b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model lmsys/vicuna-13b-v1.5 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_vicuna_13b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_OpenAssistant_12b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model chavinlo/alpaca-13b --num-prompts 1000 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_alpaca_13b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_baize_v2_13b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_koala_13B_HF.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_dolly_v2_12b.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_mpt_7b_chat.log
python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model THUDM/chatglm3-6b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp4_chatglm3_6b.log




# 构建新模型的cost model (1) 收集各模型 size和flops coefficient的信息 AND 收集各个模型的initialization cost的信息
python3 comp_model_size.py > Cost_Model_per_iter/NEWROUND_get_model_info.log 2> Cost_Model_per_iter/NEWROUND_get_model_info.err
python3 comp_model_size.py >> Cost_Model_per_iter/NEWROUND_get_model_info_init_cost.log 2>> Cost_Model_per_iter/NEWROUND_get_model_info_init_cost.err



CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model lmsys/vicuna-13b-v1.5 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_vicuna-13b-v1.5_tp1_0728_temp1.0_wldeg2_1.log 2> Cost_Model_per_iter/NEWROUND_vicuna-13b-v1.5_tp1_0728_temp1.0_wldeg2_1.err
python3 construct_cost_model.py --input-len 16 --output-len 16 --model lmsys/vicuna-13b-v1.5 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/NEWROUND_vicuna-13b-v1.5_tp2_0728_temp1.0_wldeg2_1.log 2> Cost_Model_per_iter/NEWROUND_vicuna-13b-v1.5_tp2_0728_temp1.0_wldeg2_1.err
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model lmsys/vicuna-13b-v1.5 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_vicuna-13b-v1.5_tp4_0728_temp1.0_wldeg2_1.log 2> Cost_Model_per_iter/NEWROUND_vicuna-13b-v1.5_tp4_0728_temp1.0_wldeg2_1.err


CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp1_0728_temp1.0_wldeg2_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp2_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp4_0728_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model chavinlo/alpaca-13b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_alpaca-13b_tp1_0728_temp1.0_wldeg2_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model chavinlo/alpaca-13b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/NEWROUND_alpaca-13b_tp2_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model chavinlo/alpaca-13b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_alpaca-13b_tp4_0728_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_baize-v2-13b_tp1_0728_temp1.0_wldeg2_1.log
python3 construct_cost_model.py --input-len 16 --output-len 16 --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend ours> Cost_Model_per_iter/NEWROUND_baize-v2-13b_tp2_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_baize-v2-13b_tp4_0728_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_koala-13B-HF_tp1_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_koala-13B-HF_tp2_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_koala-13B-HF_tp4_0728_temp1.0_wldeg2_1.log



CUDA_VISIBLE_DEVICES=3,2,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_dolly-v2-12b_tp1_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_dolly-v2-12b_tp2_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_dolly-v2-12b_tp4_0728_temp1.0_wldeg2_1.log



CUDA_VISIBLE_DEVICES=1,2,3,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_mpt-7b-chat_tp1_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_mpt-7b-chat_tp2_0728_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_mpt-7b-chat_tp4_0728_temp1.0_wldeg2_1.log






# 收集output Length distribution
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model lmsys/vicuna-13b-v1.5 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_vicuna-13b-v1.5_tp2_0730_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp2_0730_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model chavinlo/alpaca-13b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_alpaca-13b_tp2_0730_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model project-baize/baize-v2-13b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_baize-v2-13b_tp2_0730_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model TheBloke/koala-13B-HF --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_koala-13B-HF_tp2_0730_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model databricks/dolly-v2-12b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_dolly-v2-12b_tp2_0730_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 benchmark_throughput.py --dataset no_robot.parquet --model mosaicml/mpt-7b-chat --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_mpt-7b-chat_tp2_0730_10kreq_1.log

# after collecting output length distribution, get the pdf from logs.
# run scripts in collect_output_lengths/analyse_script.py

# test search algorithms~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 search_exec_plans.py > test_search/testcase_llmblender_0808_1.log
python3 search_exec_plans.py >> test_search/testcase_data_parallel_chatglm_0814_1.log
python3 search_exec_plans.py >> test_search/test_dp_pipeline_2Llama_0917_1.log

# test end2end schedule performance~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA_VISIBLE_DEVICES=0,3 python3 schedule_multi_model.py > test_end2end_schedule/test_llmblender_0808_2gpu_1.log
CUDA_VISIBLE_DEVICES=0,3,1,2 python3 schedule_multi_model.py > test_end2end_schedule/test_llmblender_0811_4gpu_1.log



# test end2end schedule performance with data parallelism~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 schedule_multi_model.py > test_end2end_schedule/test_data_parallel_0903_4gpu_1.log




# test end2end schedule performance with multi-level model system (data parallel + model-level pipeline)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 schedule_multi_model.py > test_end2end_schedule/test_multi-level_system_0929_4gpu_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 schedule_multi_model.py > test_end2end_schedule/test_multi-level_system_1021_2gpu_1-chainSummary-1.log
CUDA_VISIBLE_DEVICES=1,2 python3 schedule_multi_model.py > test_end2end_schedule/test_multi-level_system_1021_2gpu_1-chainSummary-3.log
CUDA_VISIBLE_DEVICES=4,7 python3 schedule_multi_model.py > test_end2end_schedule/test_multi-level_system_1021_2gpu_1-mapreduce-4_horizontal_fusion_3.log
CUDA_VISIBLE_DEVICES=4,7 python3 schedule_multi_model.py > test_end2end_schedule/test_multi-level_system_1021_2gpu_1-chain_summary_naiveSearchSpace_1.log
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 schedule_multi_model.py > test_end2end_schedule/test_multi-level_system_1031_4gpu-general_ours_1.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_0.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 1 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 2 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 3 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_3.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 4 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_4.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 5 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_5.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 6 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_1_6.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_0.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 1 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 2 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 3 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_3.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 4 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_4.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 5 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_5.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 6 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_ours_10kreq_2_6.log



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_0.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 1 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 2 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 3 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_3.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 4 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_4.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 5 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_5.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 6 --ratio-set 1  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_1_6.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_0.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 1 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 2 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 3 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_3.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 4 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_4.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 5 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_5.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 6 --ratio-set 2  > test_end2end_schedule/test_11118gpu-router_naiveSearchSpace_V2_10kreq_2_6.log



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_ours_10kreq_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_naiveSearchSpace_V2_10kreq_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_ours_10kreq_2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_naiveSearchSpace_V2_10kreq_2.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_ours_40G_10kreq_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_naiveSearchSpace_V2_40G_10kreq_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-chain-summary_ours_40G_10kreq_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-chain-summary_naiveSearchSpace_V2_40G_10kreq_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-chain-summary_ours_40G_1kreq_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-chain-summary_naiveSearchSpace_V2_40G_1kreq_1.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_ours_1kreq_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_11118gpu-general_naiveSearchSpace_V2_1kreq_1.log




python3 search_exec_plans.py >> test_search/test_multi-level_system_1003_2gpu_1-chainSummary-1.log
python3 search_exec_plans.py >> test_search/test_multi-level_system_1009_2gpu_1-chainSummary-1.log









































# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# new round experiments for RouterBench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model meta-llama/Llama-2-70b-chat-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp2_Llama-2-70b-chat-hf.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mixtral-8x7B-Instruct-v0.1  --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp2_Mixtral-8x7B-Instruct-v0.1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model 01-ai/Yi-34B-Chat  --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp2_Yi-34B-Chat.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp2_WizardLM-13B-V1.2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp2_CodeLlama-34b-Instruct-hf.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model mistralai/Mistral-7B-Instruct-v0.1 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code -gpuratio 0.9 -wldegree 2 --ignore-eos> NEWROUND_end2end_test/baseline_tp1_Mistral-7B-Instruct-v0.1.log



huggingface-cli download meta-llama/Llama-2-70b-chat-hf
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1
huggingface-cli download 01-ai/Yi-34B-Chat
huggingface-cli download WizardLMTeam/WizardLM-13B-V1.2
huggingface-cli download meta-llama/CodeLlama-34b-Instruct-hf
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2


huggingface-cli download lmsys/vicuna-13b-v1.5
huggingface-cli download OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
huggingface-cli download chavinlo/alpaca-13b
huggingface-cli download project-baize/baize-v2-13b
huggingface-cli download TheBloke/koala-13B-HF
huggingface-cli download databricks/dolly-v2-12b
huggingface-cli download mosaicml/mpt-7b-chat
huggingface-cli download THUDM/chatglm3-6b
huggingface-cli download stabilityai/stablelm-tuned-alpha-7b





# collect output lengths distribution data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/Llama-2-70b-chat-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Llama-2-70b-chat-hf_tp2_1202_10kreq_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mixtral-8x7B-Instruct-v0.1 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mixtral-8x7B-Instruct-v0.1_tp2_1202_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model 01-ai/Yi-34B-Chat --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Yi-34B-Chat_tp2_1202_10kreq_1.log

CUDA_VISIBLE_DEVICES=0,1 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_WizardLM-13B-V1.2_tp2_1202_10kreq_1.log
CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_CodeLlama-34b-Instruct-hf_tp2_1202_10kreq_1.log
CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mistral-7B-Instruct-v0.2_tp2_1202_10kreq_1.log


# apply chat template and re-run 
CUDA_VISIBLE_DEVICES=0,1 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/Llama-2-70b-chat-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Llama-2-70b-chat-hf_tp2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=2,3 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mixtral-8x7B-Instruct-v0.1 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mixtral-8x7B-Instruct-v0.1_tp2_1205_10kreq_1.log

CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_WizardLM-13B-V1.2_tp2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_CodeLlama-34b-Instruct-hf_tp2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mistral-7B-Instruct-v0.2_tp2_1205_10kreq_1.log


CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model THUDM/chatglm3-6b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_chatglm3-6b_tp2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model stabilityai/stablelm-tuned-alpha-7b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_stablelm-tuned-alpha-7b_tp2_1205_10kreq_1.log



# 重新跑不同temprature下的加了chat template的版本的output 长度分布；所有模型都跑吗？还是只跑LLM-Blender下的。先跑Router版本的吧。
CUDA_VISIBLE_DEVICES=0,1 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/Llama-2-70b-chat-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 0.2 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Llama-2-70b-chat-hf_tp2_temp0.2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mixtral-8x7B-Instruct-v0.1 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 0.2 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mixtral-8x7B-Instruct-v0.1_tp2_temp0.2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 0.2 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_WizardLM-13B-V1.2_tp2_temp0.2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 0.2 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_CodeLlama-34b-Instruct-hf_tp2_temp0.2_1205_10kreq_1.log
CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 0.2 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mistral-7B-Instruct-v0.2_tp2_temp0.2_1205_10kreq_1.log

for temp in 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=0,1 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/Llama-2-70b-chat-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Llama-2-70b-chat-hf_tp2_temp${temp}_1205_10kreq_1.log
done


for temp in 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mixtral-8x7B-Instruct-v0.1 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mixtral-8x7B-Instruct-v0.1_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_WizardLM-13B-V1.2_tp2_temp${temp}_1205_10kreq_1.log
done

for temp in 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_CodeLlama-34b-Instruct-hf_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Mistral-7B-Instruct-v0.2_tp2_temp${temp}_1205_10kreq_1.log
done


# 重新跑不同temprature下的LLM-Blender里的模型，加了chat template
for temp in 0.7 0.5 # 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=0,1 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model lmsys/vicuna-13b-v1.5 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_vicuna-13b-v1.5_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=0,1 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp2_temp${temp}_1205_10kreq_1.log
done

for temp in 0.7 0.5 # 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=2,3 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model chavinlo/alpaca-13b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_alpaca-13b_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=2,3 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model project-baize/baize-v2-13b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_baize-v2-13b_tp2_temp${temp}_1205_10kreq_1.log
done

for temp in 0.7 0.5 # 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model TheBloke/koala-13B-HF --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_koala-13B-HF_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model databricks/dolly-v2-12b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_dolly-v2-12b_tp2_temp${temp}_1205_10kreq_1.log
done


for temp in 0.7 0.5 # 0.2 0.4 0.6 0.8
do
    CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mosaicml/mpt-7b-chat --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_mpt-7b-chat_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model THUDM/chatglm3-6b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_chatglm3-6b_tp2_temp${temp}_1205_10kreq_1.log
    CUDA_VISIBLE_DEVICES=6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model stabilityai/stablelm-tuned-alpha-7b --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature $temp -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_stablelm-tuned-alpha-7b_tp2_temp${temp}_1205_10kreq_1.log
done



# forget to get max output lengths





# collect per iter latency doata
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model meta-llama/Llama-2-70b-chat-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_Llama-2-70b-chat-hf_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mixtral-8x7B-Instruct-v0.1 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_Mixtral-8x7B-Instruct-v0.1_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_WizardLM-13B-V1.2_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_CodeLlama-34b-Instruct-hf_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_Mistral-7B-Instruct-v0.2_tp2_1202_temp1.0_wldeg2_1.log

CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_WizardLM-13B-V1.2_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_CodeLlama-34b-Instruct-hf_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_Mistral-7B-Instruct-v0.2_tp1_1202_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model meta-llama/Llama-2-70b-chat-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_Llama-2-70b-chat-hf_tp4_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mixtral-8x7B-Instruct-v0.1 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_Mixtral-8x7B-Instruct-v0.1_tp4_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model WizardLMTeam/WizardLM-13B-V1.2 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_WizardLM-13B-V1.2_tp4_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model meta-llama/CodeLlama-34b-Instruct-hf --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_CodeLlama-34b-Instruct-hf_tp4_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mistralai/Mistral-7B-Instruct-v0.2 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_Mistral-7B-Instruct-v0.2_tp4_1202_temp1.0_wldeg2_1.log


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 重新跑一下zxcpu上的其他model的per-iter cost data


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 comp_model_size.py



CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model lmsys/vicuna-13b-v1.5 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_vicuna-13b-v1.5_tp1_1202_temp1.0_wldeg2_1.log 
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model lmsys/vicuna-13b-v1.5 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_vicuna-13b-v1.5_tp2_1202_temp1.0_wldeg2_1.log 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model lmsys/vicuna-13b-v1.5 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_vicuna-13b-v1.5_tp4_1202_temp1.0_wldeg2_1.log 


CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_oasst-sft-4-pythia-12b-epoch-3.5_tp4_1202_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model chavinlo/alpaca-13b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_alpaca-13b_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model chavinlo/alpaca-13b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_alpaca-13b_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model chavinlo/alpaca-13b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_alpaca-13b_tp4_1202_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_baize-v2-13b_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_baize-v2-13b_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model project-baize/baize-v2-13b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_baize-v2-13b_tp4_1202_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_koala-13B-HF_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_koala-13B-HF_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model TheBloke/koala-13B-HF --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_koala-13B-HF_tp4_1202_temp1.0_wldeg2_1.log



CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_dolly-v2-12b_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_dolly-v2-12b_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model databricks/dolly-v2-12b --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_dolly-v2-12b_tp4_1202_temp1.0_wldeg2_1.log



CUDA_VISIBLE_DEVICES=2 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_mpt-7b-chat_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_mpt-7b-chat_tp2_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 4 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_mpt-7b-chat_tp4_1202_temp1.0_wldeg2_1.log


CUDA_VISIBLE_DEVICES=4 python3 construct_cost_model.py --input-len 16 --output-len 16 --model THUDM/chatglm3-6b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_chatglm3-6b_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4,5 python3 construct_cost_model.py --input-len 16 --output-len 16 --model THUDM/chatglm3-6b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_chatglm3-6b_tp2_1202_temp1.0_wldeg2_1.log

CUDA_VISIBLE_DEVICES=4 python3 construct_cost_model.py --input-len 16 --output-len 16 --model stabilityai/stablelm-tuned-alpha-7b --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_stablelm-tuned-alpha-7b_tp1_1202_temp1.0_wldeg2_1.log
CUDA_VISIBLE_DEVICES=4,5 python3 construct_cost_model.py --input-len 16 --output-len 16 --model stabilityai/stablelm-tuned-alpha-7b --num-prompts 1 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter_zxcpu/NEWROUND_stablelm-tuned-alpha-7b_tp2_1202_temp1.0_wldeg2_1.log





# test end2end schedule performance with routerbench dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_maxlen_8192_2.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_maxlen_8192_2.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_singlechoice_3.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_singlechoice_3.log


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_singlechoice_3.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_singlechoice_3.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_maxlen_8192_2.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_maxlen_8192_2.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_maxlen_4096_2.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_maxlen_4096_2.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_maxlen_4096_3.log

CUDA_VISIBLE_DEVICES=2,3,4,5 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_test_chat_template_1.log



CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_Not_MCQ_2.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_Not_MCQ_2.log



CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_ours_Not_MCQ_maxlen_4096_3.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1203_8gpu-router_naiveSearchSpace_V2_Not_MCQ_maxlen_4096_3.log



CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-router_naiveSearchSpace_V2_Not_MCQ_set_outlen_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-router_ours_Not_MCQ_set_outlen_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_8gpu-router_naiveSearchSpace_V2_Not_MCQ_set_outlen_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_8gpu-router_ours_Not_MCQ_set_outlen_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_8gpu-router_ours_Not_MCQ_set_outlen_topk1_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_8gpu-router_naiveSearchSpace_V2_MCQ_set_outlen_1.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_8gpu-router_ours_MCQ_set_outlen_topk1_quota1_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-router_naiveSearchSpace_V2_MCQ_set_outlen_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case router --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-router_ours_MCQ_set_outlen_topk1_quota1_1.log


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-blender_naiveSearchSpace_V2_maxlen_512_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-blender_ours_maxlen_512_1.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-blender_naiveSearchSpace_V2_maxlen_512_10k_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case general --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1208_4gpu-blender_ours_maxlen_512_10k_1.log



CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_4gpu-booookscore_naiveSearchSpace_V2_maxlen_900_1k_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_4gpu-booookscore_ours_maxlen_900_1k_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_4gpu-booookscore_naiveSearchSpace_V2_maxlen_900_100_1.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_4gpu-booookscore_ours_maxlen_900_100_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_4gpu-booookscore_naiveSearchSpace_V2_maxlen_900_500_3.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_4gpu-booookscore_ours_maxlen_900_500_3.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_maxlen_900_500_4.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_maxlen_900_500_4.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_maxlen_900_500_8.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_maxlen_900_500_8.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_maxlen_900_500_9.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_maxlen_900_500_9.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_17.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_17.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_18.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_18.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_19.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_19.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_20.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_20.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_21.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_21.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_22.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_22.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_23.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_23.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_24.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_24.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_25.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_25.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_26.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_26.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_27.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_27.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline naive --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_naiveSearchSpace_V2_2eval_maxlen_900_500_28.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1211_8gpu-booookscore_ours_2eval_maxlen_900_500_28.log




# 看到底为什么vertically fused model的cost estimation很慢
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline ours --test-case chain-summary --ratio-seed 0 --ratio-set 1  > test_end2end_schedule/test_1228_8gpu-booookscore_ours_2eval_maxlen_900_500_1.log




























# 正式用命令行测试所有实验结果


gpu_name=A100-80G
byte_per_gpu=85899345920
max_group_seq_num=1
top_k=20
similar_threshold=0.2
fully_connected_gpu_unit=2
machine_name=zxcpu


gen_execplans_baseline=ours
# gen_execplans_baseline=naive
specify_outlen=
# specify_outlen=--specify_outlen

# chain-summary
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case chain-summary --ratio-seed 0 --ratio-set 1 --reqnum 100 --max_token_num 900 --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num 8 --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num 3 --summarize_model 'lmsys/vicuna-13b-v1.5' --evaluator_model 'meta-llama/Llama-2-70b-chat-hf' > test_end2end_schedule/test_0104_8gpu-booookscore_${gen_execplans_baseline}_5eval_maxlen_900_500_1.log


# ensemble
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case general --ratio-seed 0 --ratio-set 1 --reqnum 10000 --max_token_num 512 $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num 8 --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name > test_end2end_schedule/test_1231_8gpu-llm-blender_${gen_execplans_baseline}_maxlen_512_10k_1.log


# router
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case router --ratio-seed 0 --ratio-set 1 --reqnum 10000 --router_question_version 'not_multiple_choice_question' --max_token_num 4096  $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num 8 --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name > test_end2end_schedule/test_1231_8gpu-router_${gen_execplans_baseline}_maxlen_4096_10k_1.log





# 运行实验需要的各种setting

# for gpu_name in A100-40G A100-80G
for gpu_name in A100-80G
do
    byte_per_gpu=85899345920
    if [ $gpu_name = A100-40G ]; then
        byte_per_gpu=42949672960
    fi
    # for tot_gpu_num in 8 4
    for tot_gpu_num in 8 4
    do
        gpu_ids=0,1,2,3,4,5,6,7
        if [ $tot_gpu_num -eq 4 ]; then
            gpu_ids=0,1,2,3
        fi
        for max_group_seq_num in 1 20
        do 
            top_k=20
            similar_threshold=0.2
            fully_connected_gpu_unit=2
            machine_name=zxcpu

            specify_outlen=
            # chain-summary
            # for summarize_model in lmsys/vicuna-13b-v1.5 mistralai/Mixtral-8x7B-Instruct-v0.1
            for summarize_model in lmsys/vicuna-13b-v1.5 
            do
                summarize_model_setting=vicuna-13b-v1.5
                if [ $summarize_model = mistralai/Mixtral-8x7B-Instruct-v0.1 ]; then
                    summarize_model_setting=Mixtral-8x7B-Instruct-v0.1
                fi
                # for reqnum in 100 300 500 700
                for reqnum in 100 300 500
                do
                    # for evaluator_num in 1 2 3 4 5 6 7
                    for evaluator_num in 2 4 6 8
                    do
                        for gen_execplans_baseline in ours naive
                        do
                            if [ $max_group_seq_num -eq 20 ] && [ $gen_execplans_baseline = naive ]; then
                                continue
                            fi

                            if [ -a test_end2end_schedule/test_1231_${tot_gpu_num}gpu-booookscore_${gen_execplans_baseline}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_900_${reqnum}_${max_group_seq_num}_1.log ]; then
                                echo skip test_1231_${tot_gpu_num}gpu-booookscore_${gen_execplans_baseline}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_900_${reqnum}_${max_group_seq_num}_1.log
                                continue
                            fi

                            CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case chain-summary --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num 900 --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf >> test_end2end_schedule/test_1231_${tot_gpu_num}gpu-booookscore_${gen_execplans_baseline}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_900_${reqnum}_${max_group_seq_num}_1.log
                        done
                    done
                done
            done
            specify_outlen=
            # router
            for use_specify_outlen in no yes
            do
                specify_outlen=
                outlen_file_name_setting=maxlen_4096
                if [ $use_specify_outlen = yes ]; then
                    specify_outlen=--specify_outlen
                    outlen_file_name_setting=setOutlen
                fi

                echo use_specify_outlen: $use_specify_outlen  specify_outlen: $specify_outlen

                for gen_execplans_baseline in ours naive
                do 
                    if [ $max_group_seq_num -eq 20 ] && [ $gen_execplans_baseline = naive ]; then
                        continue
                    fi

                    if [ -a test_end2end_schedule/test_1231_${tot_gpu_num}gpu-router_${gen_execplans_baseline}_${gpu_name}_${machine_name}_not_multiple_choice_question_${outlen_file_name_setting}_10000_${max_group_seq_num}_1.log ]; then
                        echo skip test_1231_${tot_gpu_num}gpu-router_${gen_execplans_baseline}_${gpu_name}_${machine_name}_not_multiple_choice_question_${outlen_file_name_setting}_10000_${max_group_seq_num}_1.log
                        continue
                    fi

                    CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case router --ratio-seed 0 --ratio-set 1 --reqnum 10000 --router_question_version 'not_multiple_choice_question' --max_token_num 4096  $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name >> test_end2end_schedule/test_1231_${tot_gpu_num}gpu-router_${gen_execplans_baseline}_${gpu_name}_${machine_name}_not_multiple_choice_question_${outlen_file_name_setting}_10000_${max_group_seq_num}_1.log
                done
            done
            specify_outlen=
            # ensemble
            for reqnum in 1000 5000 10000 
            do
                for max_token_num in 512 256
                do
                    for gen_execplans_baseline in ours naive
                    do 

                        if [ $max_group_seq_num -eq 20 ] && [ $gen_execplans_baseline = naive ]; then
                            continue
                        fi

                        if [ -a test_end2end_schedule/test_1231_${tot_gpu_num}gpu-llm-blender_${gen_execplans_baseline}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_1.log ]; then
                            echo skip test_1231_${tot_gpu_num}gpu-llm-blender_${gen_execplans_baseline}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_1.log
                            continue
                        fi

                        CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case general --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name >> test_end2end_schedule/test_1231_${tot_gpu_num}gpu-llm-blender_${gen_execplans_baseline}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_1.log
                    done
                done
            done
        done
    done
done





