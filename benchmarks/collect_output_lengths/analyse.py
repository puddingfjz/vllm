import json

prefix = 'collect_output_lengths/'



dataset = {key:list() 
           for key in [
               'llama7b', 'llama13b', 'llama70b', 
               'Aquila-7B', 'AquilaChat-7B', 
               'Baichuan2-13B-Chat', 'Baichuan-7B', 
               'chatglm3-6b', 
               'gpt-j-6b', 'gpt-neox-20b', 
               'Mixtral-8x7B-v0.1', 'phi-2',
               'Qwen2-beta-7B', 'Qwen2-beta-7B-Chat']}




def update_dataset(dataset, key, filename):
    with open(prefix+filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'output_lens = ' in line:
                data = line[len('output_lens = '):]
                data = json.loads(data)
                dataset[key].append(data)




update_dataset(dataset, 'llama7b', 'vllm_7b_tp1_0328_1kreq_2.log')
update_dataset(dataset, 'llama13b', 'vllm_13b_tp1_0328_1kreq_1.log')
update_dataset(dataset, 'llama70b', 'vllm_70b_tp2_0328_1kreq_1.log')
update_dataset(dataset, 'llama7b', 'vllm_7b_tp2_0328_1kreq_3.log')
update_dataset(dataset, 'llama13b', 'vllm_13b_tp2_0328_1kreq_2.log')
update_dataset(dataset, 'llama70b', 'vllm_70b_tp2_0328_1kreq_2.log')
update_dataset(dataset, 'Aquila-7B', 'Aquila-7B_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'AquilaChat-7B', 'AquilaChat-7B_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'Baichuan2-13B-Chat', 'Baichuan2-13B-Chat_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'Baichuan-7B', 'Baichuan-7B_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'chatglm3-6b', 'chatglm3-6b_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'gpt-j-6b', 'gpt-j-6b_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'gpt-neox-20b', 'gpt-neox-20b_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'Mixtral-8x7B-v0.1', 'Mixtral-8x7B-v0.1_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'phi-2', 'phi-2_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'Qwen2-beta-7B', 'Qwen2-beta-7B_tp4_0328_1kreq_1.log')
update_dataset(dataset, 'Qwen2-beta-7B-Chat', 'Qwen2-beta-7B-Chat_tp4_0328_1kreq_1.log')








for k, vs in dataset.items():
    print(f"{k:<20}TOT_len: {str([sum([sum(i[:2]) for i in v]) for v in vs]):<20}AVG_out: {str([sum([i[1] for i in v]) / len(v) for v in vs]):<20}")


