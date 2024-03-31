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




python3 benchmark_throughput.py --dataset no_robot.parquet --model THUDM/chatglm3-6b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/chatglm3-6b_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model CohereForAI/c4ai-command-r-v01 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/c4ai-command-r-v01_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model google/gemma-2b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gemma-2b_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model google/gemma-7b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gemma-7b_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model bigcode/starcoder --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/starcoder_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model EleutherAI/gpt-j-6b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gpt-j-6b_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model EleutherAI/gpt-neox-20b --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/gpt-neox-20b_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model mistralai/Mistral-7B-v0.1 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Mistral-7B-v0.1_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model mistralai/Mistral-7B-Instruct-v0.1 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Mistral-7B-Instruct-v0.1_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model mistralai/Mixtral-8x7B-v0.1 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Mixtral-8x7B-v0.1_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model microsoft/phi-2 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/phi-2_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model Qwen/Qwen2-beta-7B --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Qwen2-beta-7B_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model Qwen/Qwen2-beta-7B-Chat --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/Qwen2-beta-7B-Chat_tp4_0331_10kreq_1.log
python3 benchmark_throughput.py --dataset no_robot.parquet --model stabilityai/stablelm-base-alpha-7b-v2 --num-prompts 10000 --enforce-eager -tp 4 --trust-remote-code> collect_output_lengths/no_robot/stablelm-base-alpha-7b-v2_tp4_0331_10kreq_1.log
