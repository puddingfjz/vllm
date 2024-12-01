# This file include the scripts to add a new model in our system

1. run construct_cost_model.py to collect cost model data for per-iter latency

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_mpt-7b-chat_tp1_0728_temp1.0_wldeg2_1.log
```

2. run normal benchmark_throughut vllm inference function to collect output lengths distribution data

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model mosaicml/mpt-7b-chat --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm > collect_output_lengths/no_robot/NEWROUND_mpt-7b-chat_tp2_0730_10kreq_1.log
```

3. run collect_output_lengths/analyse_script.py to get output length distribution [need modify ``analyse_script.py``]

```shell
cd collect_output_lengths
python3 analyse_script.py
```

4. run my_per_iter_latency_estimator.get_cost_table to prepare cost model meta data for direct use later [need modify ``get_cost_table()``]

```python
from my_per_iter_latency_estimator import get_cost_table
cost_table = get_cost_table()
```

5. 构建新模型的cost model (1) 收集各模型 size和flops coefficient的信息 AND 收集各个模型的initialization cost的信息 [need modify ``comp_model_size.py``]

```python
python3 comp_model_size.py > Cost_Model_per_iter/NEWROUND_get_model_info.log 2> Cost_Model_per_iter/NEWROUND_get_model_info.err
python3 comp_model_size.py >> Cost_Model_per_iter/NEWROUND_get_model_info_init_cost.log 2>> Cost_Model_per_iter/NEWROUND_get_model_info_init_cost.err
```
