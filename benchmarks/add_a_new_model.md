# This file include the scripts to add a new model in our system

1. download model weights

```shell
huggingface-cli download mosaicml/mpt-7b-chat
```

2. get model size, flops coeffs, and init cost.
   构建新模型的cost model (1) 收集各模型 size和flops coefficient的信息 AND 收集各个模型的initialization cost的信息 [need modify ``comp_model_size.py``].  **这里更新的init_cost需要重新更新per-iter-latency cost model才能生效（i.e.，执行step 6）**

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 comp_model_size.py
```

3. run construct_cost_model.py to collect cost model data for per-iter latency

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 construct_cost_model.py --input-len 16 --output-len 16 --model mosaicml/mpt-7b-chat --num-prompts 1 --enforce-eager -tp 1 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 --backend vllm> Cost_Model_per_iter/NEWROUND_mpt-7b-chat_tp1_0728_temp1.0_wldeg2_1.log
```

4. run normal benchmark_throughut vllm inference function to collect output lengths distribution data

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_throughput.py --backend vllm_ori --dataset no_robot.parquet --model meta-llama/Llama-2-70b-chat-hf --num-prompts 10000 --enforce-eager -tp 2 --trust-remote-code --temperature 1.0 -gpuratio 0.9 -wldegree 2 > collect_output_lengths/no_robot/NEWROUND_Llama-2-70b-chat-hf_tp2_1202_10kreq_1.log
```

5. run collect_output_lengths/analyse_script.py to get output length distribution [need modify ``analyse_script.py``]

```shell
# cd collect_output_lengths
python3 collect_output_lengths/analyse_script.py
```

6. run my_per_iter_latency_estimator.get_cost_table to prepare cost model meta data for direct use later [need modify ``get_cost_table()``]

```python
from my_per_iter_latency_estimator import get_cost_table, get_cost_table_zxcpu
# cost_table = get_cost_table()
cost_table = get_cost_table_zxcpu()
```

`<!-- 7. 构建新模型的cost model (1) 收集各模型 size和flops coefficient的信息 AND 收集各个模型的initialization cost的信息 [need modify ``comp_model_size.py``] -->`

<!-- ```python
python3 comp_model_size.py > Cost_Model_per_iter/NEWROUND_get_model_info.log 2> Cost_Model_per_iter/NEWROUND_get_model_info.err
python3 comp_model_size.py >> Cost_Model_per_iter/NEWROUND_get_model_info_init_cost.log 2>> Cost_Model_per_iter/NEWROUND_get_model_info_init_cost.err
``` -->

7. 如果要从某个服务器传输模型参数到另一个服务器

```shell
scp -r ./models--meta-llama--Llama-2-70b-chat-hf jfangak@zxcpu1:/ssddata/jingzhi/.cache/huggingface/hub/
scp -r ./models--WizardLMTeam--WizardLM-13B-V1.2 jfangak@zxcpu1:/ssddata/jingzhi/.cache/huggingface/hub/
scp -r ./models--mistralai--Mixtral-8x7B-Instruct-v0.1 jfangak@zxcpu1:/ssddata/jingzhi/.cache/huggingface/hub/
scp -r ./models--mistralai--Mistral-7B-Instruct-v0.2 jfangak@zxcpu1:/ssddata/jingzhi/.cache/huggingface/hub/
scp -r ./models--meta-llama--CodeLlama-34b-Instruct-hf jfangak@zxcpu1:/ssddata/jingzhi/.cache/huggingface/hub/
```
