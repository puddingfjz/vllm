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




for tp in 1 2 4
do
    python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts 1000 --enforce-eager -tp $tp > vllm_0226_7b_1_tp${tp}.log
done
