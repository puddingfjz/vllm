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


