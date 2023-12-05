for i in {1..180}
do
   python3 benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model huggyllama/llama-7b --num-prompts $i >> CostModel_log_peakaware_recomp3.log
done
