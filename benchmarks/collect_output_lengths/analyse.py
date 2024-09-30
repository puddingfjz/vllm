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



# print the cumulative distribution curve for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 4))
for k, vs in dataset.items():   
    lens = vs[0]
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens]
    # 
    outs = np.asarray(outs)
    outs = outs[outs<=4096]
    elements, counts = np.unique(outs, return_counts=True)
    cum_counts = np.cumsum(counts)
    cum_counts = cum_counts/cum_counts[-1]
    ax.scatter(elements, cum_counts, s=2, label = k)
    ax.set(xlabel='out len', ylabel='cum num')
        #    title='About as simple as it gets, folks')
    if max(outs) > 4096:
        print(k)



ax.grid()
plt.legend()
fig.savefig(f"Cost_Model_per_iter/figures/outlen_cum_distribution_AllModels.pdf")
plt.show()



# print cumulative distribution curve for tot len
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 4))
for k, vs in dataset.items():   
    lens = vs[0]
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens]
    # 
    outs = np.asarray(outs)
    inps = np.asarray(inps)
    tots = outs + inps
    tots = tots[tots<=4096]
    elements, counts = np.unique(tots, return_counts=True)
    cum_counts = np.cumsum(counts)
    cum_counts = cum_counts/cum_counts[-1]
    ax.scatter(elements, cum_counts, s=2, label = k)
    ax.set(xlabel='tot len', ylabel='cum num')
        #    title='About as simple as it gets, folks')
    if max(tots) > 4096:
        print(k)



ax.grid()
plt.legend()
fig.savefig(f"Cost_Model_per_iter/figures/totlen_cum_distribution_AllModels.pdf")
plt.show()




# ========================================================
# ========================================================
# ========================================================
# analyse the data for ``no_robot'' dataset
import json

prefix = 'collect_output_lengths/no_robot/'


dataset = {key:list() 
           for key in [
               'Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 
               'Aquila-7B', 'AquilaChat-7B', 
               'Baichuan2-13B-Chat', 'Baichuan-7B', 
               'chatglm3-6b', 
               'gpt-j-6b', 'gpt-neox-20b', 
               'Mixtral-8x7B-v0.1', 'phi-2',
               'Qwen2-beta-7B', 'Qwen2-beta-7B-Chat']}


max_model_len_dict = {key:None 
           for key in [
               'Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 
               'Aquila-7B', 'AquilaChat-7B', 
               'Baichuan2-13B-Chat', 'Baichuan-7B', 
               'chatglm3-6b', 
               'gpt-j-6b', 'gpt-neox-20b', 
               'Mixtral-8x7B-v0.1', 'phi-2',
               'Qwen2-beta-7B', 'Qwen2-beta-7B-Chat']}


def update_dataset(dataset, key, filename):
    try:
        with open(prefix+filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'output_lens = ' in line:
                    data = line[len('output_lens = '):]
                    data = json.loads(data)
                    dataset[key].append(data)
    except:
        print(f"Failed: file {prefix+filename}")



def get_max_model_len(max_model_len_dict, key, filename):
    # max_model_len: 4096 for llama2 7b
    try:
        with open(prefix+filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'max_model_len:' in line:
                    data = line[len('max_model_len:'):]
                    data = json.loads(data)
                    max_model_len_dict[key] = data
    except:
        print(f"Failed: file {prefix+filename}")    




for model in dataset.keys():
    filename = f'{model}_tp4_0331_10kreq_1.log'
    update_dataset(dataset, model, filename)
    get_max_model_len(max_model_len_dict, model, filename)



for k, vs in dataset.items():
    print(f"{k:<20}TOT_len: {str([sum([sum(i[:2]) for i in v]) for v in vs]):<20}AVG_out: {str([sum([i[1] for i in v]) / len(v) for v in vs]):<20}")







# print the cumulative distribution curve for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 4))
for k, vs in dataset.items():   
    if len(vs) == 0:
        continue
    lens = vs[0]
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens]
    # 
    outs = np.asarray(outs)
    outs = outs[outs<=4096]
    if len(outs) == 0:
        continue
    elements, counts = np.unique(outs, return_counts=True)
    cum_counts = np.cumsum(counts)
    cum_counts = cum_counts/cum_counts[-1]
    ax.scatter(elements, cum_counts, s=2, label = k)
    ax.set(xlabel='out len', ylabel='cum num')
        #    title='About as simple as it gets, folks')
    if max(outs) > 4096:
        print(k)



ax.grid()
plt.legend()
fig.savefig(f"Cost_Model_per_iter/figures/outlen_cum_distribution_AllModels_norobot.pdf")
plt.show()


# print the cumulative tot len distribution curve for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 4))
for k, vs in dataset.items():   
    if len(vs) == 0:
        continue
    lens = vs[0]
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens]
    # 
    inps = np.asarray(inps)
    outs = np.asarray(outs)
    tots = inps + outs
    tots = tots[tots<=4096]
    if len(tots) == 0:
        continue
    elements, counts = np.unique(tots, return_counts=True)
    cum_counts = np.cumsum(counts)
    cum_counts = cum_counts/cum_counts[-1]
    ax.scatter(elements, cum_counts, s=2, label = k)
    ax.set(xlabel='tot len', ylabel='cum num')
        #    title='About as simple as it gets, folks')
    if max(tots) > 4096:
        print(k)



ax.grid()
plt.legend()
fig.savefig(f"Cost_Model_per_iter/figures/totlen_cum_distribution_AllModels_norobot.pdf")
plt.show()




# print out len vs inp len for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np

for k, vs in dataset.items():   
    if len(vs) == 0:
        continue
    lens = vs[0]
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens]
    # 
    inps = np.asarray(inps)
    outs = np.asarray(outs)
    tots = inps + outs
    inps = inps[tots<=4096]
    outs = outs[tots<=4096]
    if len(inps) == 0:
        continue
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(inps, outs, s=2)
    ax.set(xlabel='inp len', ylabel='out len')
        #    title='About as simple as it gets, folks')
    if max(tots) > 4096:
        print(k)
    # 
    ax.grid()
    # plt.legend()
    fig.savefig(f"Cost_Model_per_iter/figures/outlen_Vs_inplen_{k}_norobot.pdf")
    plt.show()





# print out len cumulative distribution curves per inp len interval for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np

for k, vs in dataset.items():   
    if len(vs) == 0:
        continue
    lens = vs[0]
    inps = [i[0] for i in lens]
    outs = [i[1] for i in lens]
    # 
    inps = np.asarray(inps)
    outs = np.asarray(outs)
    tots = inps + outs
    inps = inps[tots<=4096]
    outs = outs[tots<=4096]
    if len(inps) == 0:
        continue
    fig, ax = plt.subplots(figsize=(8, 4))
    interval = 100
    for i in range(max(inps)//100):
        start, end = (i*interval, (i+1)*interval)
        indices = (inps>start) * (inps<=end)
        tmp_outs = outs[indices]
        if len(tmp_outs) == 0:
            continue
        # compute cumsum
        elements, counts = np.unique(tmp_outs, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_counts = cum_counts/cum_counts[-1]
        # ax.scatter(elements, cum_counts, s=2, label = f'{start, end}')
        ax.plot(elements, cum_counts, marker='1', markersize=8, label = f'{start, end} #:{len(tmp_outs)}')
    ax.set(xlabel='out len', ylabel='cum num')
        #    title='About as simple as it gets, folks')
    # 
    ax.grid()
    plt.legend()
    fig.savefig(f"Cost_Model_per_iter/figures/outlen_per_inplen_{k}_norobot.pdf")
    plt.show()






# ********************************************************
# we try to obtain the outlen distribution for each model 
# (exculding the cases where reqs are stopped due to the max_seq_len constraint).
# ********************************************************
import matplotlib.pyplot as plt
import numpy as np
pdf_dict = dict()
for k, vs in dataset.items():   
    if len(vs) == 0:
        continue
    lens = vs[0]
    # the max seq len for this model 
    max_model_len = max_model_len_dict[k]
    event_nums = np.zeros(max_model_len, dtype=np.int32)
    sample_sizes = np.zeros(max_model_len, dtype=np.int32)
    # put the records into bins where each bin associated with a inp len
    inps = np.asarray([i[0] for i in lens])
    outs = np.asarray([i[1] for i in lens])
    uniq_inps = set([i[0] for i in lens])
    assert min(outs) >= 1
    # each bin -- inp_len: out_lens, counts
    bins = {inp: np.unique(outs[inps==inp], return_counts=True) for inp in uniq_inps}
    for inp in bins:
        sample_size = sum(bins[inp][1])
        # print(bins[inp])
        for out, count in zip(*(bins[inp])):
            if inp + out < max_model_len:
                # the seqs are not stopped due to max_model_len
                event_nums[out-1] += count
        sample_sizes[:max_model_len-inp-1] = sample_sizes[:max_model_len-inp-1] \
            + sample_size
    sample_sizes[event_nums==0] = 1 # to avoid invalid division
    pdf = event_nums / sample_sizes
    cdf = np.cumsum(pdf)
    # we need to store this pdf into file so that we can directly use it
    # outlen>=max_model_len is meaningless as inplen<=0 in this case, so we set pdf[-1] to 1-cdf[-2]
    assert (pdf[-1] == 0) and (cdf[-1] == cdf[-2])
    pdf[-1] = 1 - cdf[-2]
    assert sum(pdf) == 1
    pdf_dict[k] = pdf.tolist()
    # 
    # plot the cdf together with the cumulative outlen distribution per inp for this model
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, max_model_len+1), cdf, marker='.', markersize=10, label = f'cdf: max {cdf[-1]:.2f}', color='tan')
    # 
    # plot experimental cumulative distribution curves
    tots = inps + outs
    inps = inps[tots<=max_model_len]
    outs = outs[tots<=max_model_len]
    assert max(tots) <= max_model_len, f"{k} max tot > max_model_len {max_model_len}"
    interval = 100
    for i in range(max(inps)//100):
        start, end = (i*interval, (i+1)*interval)
        indices = (inps>start) * (inps<=end)
        tmp_outs = outs[indices]
        if len(tmp_outs) == 0:
            continue
        # compute cumsum
        elements, counts = np.unique(tmp_outs, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_counts = cum_counts/cum_counts[-1]
        # ax.scatter(elements, cum_counts, s=2, label = f'{start, end}')
        ax.plot(elements, cum_counts, marker='1', markersize=8, label = f'{start, end} #:{len(tmp_outs)}')
    ax.set(xlabel=f'out len (max_model_len:{max_model_len})', ylabel='cum prob')
    # 
    ax.grid()
    plt.legend()
    fig.savefig(f"Cost_Model_per_iter/figures/cdf_{k}_norobot_2.pdf")
    plt.show()    



# store the pdf dict to file so that we can directly use it
with open('./collect_output_lengths/no_robot/out_len_sampler_2.py', 'w') as file:
    file.write(f"pdf_dict = {json.dumps(pdf_dict)}")








# ---------------------------------------------------------------------------------------------
# analyse the output length according the request category
# first get the category information

def get_dataset(dataset_path: str):
    if dataset_path == 'ShareGPT_V3_unfiltered_cleaned_split.json':
        with open(f'{dataset_path}') as f:
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
                             data['messages'][1]['content'], 
                             data['category']) for data in a])
        return dataset
          



from typing import List, Optional, Tuple
import random
import os
from transformers import AutoTokenizer
os.environ['SORT_REQS'] = 'True'

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # 
    # Load the dataset.
    # <jingzhi>
    dataset = get_dataset(dataset_path)
    # 
    # Tokenize the prompts and completions.
    prompts = [data[0] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data[1] for data in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        if len(dataset[i]) > 2:
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len, dataset[i][2]))
        else:
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len, None))
    # 
    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len, category in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, category))
    #  
    # Sample the requests.
    # <jingzhi> make sample size be ``min(num_requests, len(filtered_dataset))''
    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))
    # 
    if os.environ['SORT_REQS'] == 'True':
        sampled_requests = sorted(sampled_requests, key=lambda x: x[1], reverse=True)
    # 
    print(f"tot_tokens: {sum([x[1]+x[2] for x in sampled_requests])}, tot_context_lens: {sum([(x[1]+x[2]-1)*(x[1]+x[2])/2 for x in sampled_requests])}")
    # 
    return sampled_requests



def get_sampled_data(model):
    seed = 0
    random.seed(seed)
    # 
    tokenizer = model
    # 
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer, trust_remote_code=True)
    # 
    dataset = 'no_robot.parquet'
    num_prompts = 10000
    # 
    requests = sample_requests(dataset, num_prompts, tokenizer, None)
    return requests



model_paths = ['THUDM/chatglm3-6b', 'CohereForAI/c4ai-command-r-v01', 
               'google/gemma-2b', 'google/gemma-7b', 
               'bigcode/starcoder', 
               'EleutherAI/gpt-j-6b', 'EleutherAI/gpt-neox-20b', 
               'mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-v0.1',
               'microsoft/phi-2', 'Qwen/Qwen2-beta-7B', 'Qwen/Qwen2-beta-7B-Chat', 
               'stabilityai/stablelm-base-alpha-7b-v2', 
               'BAAI/Aquila-7B', 'BAAI/AquilaChat-7B', 
               'baichuan-inc/Baichuan2-13B-Chat', 'baichuan-inc/Baichuan-7B', 
               'NousResearch/Llama-2-7b-hf', 'NousResearch/Llama-2-7b-chat-hf']





new_dataset = dict()
for model in model_paths:
    new_data = {}
    for k in dataset:
        pos = model.find('/')
        if k == model[pos+1:]:
            if len(dataset[k]) < 1:
                new_dataset[k] = new_data
                break
            requests = get_sampled_data(model)
            assert len(dataset[k]) == 1
            for req, data in zip(requests, dataset[k][0]):
                category = req[-1]
                assert (req[1], req[2]) == (data[0], data[2])
                if category not in new_data:
                    new_data[category] = list()
                new_data[category].append(data)
            new_dataset[k] = new_data




category_strs = list(list(new_dataset.values())[0].keys())
# compute the average response for each category
for count, (model, categorys) in enumerate(new_dataset.items()):
    if count == 0:
        print(f"{'':<20}{''.join([f'{i:<12}' for i in category_strs])}")
    if len(categorys.keys()) == 0:
        types = categorys.keys()
    else:
        types = category_strs
    vs = [categorys[t] for t in types]
    values = [(sum([i[0] for i in v]) / len(v), 
               sum([i[1] for i in v]) / len(v), 
               len(v), 
               np.std([i[1] for i in v])) for v in vs]
    # print(f"{model:<20}"+''.join([f"{(f'{value[0]:.0f}' + f', {value[1]:.0f}' + f', {value[2]:.0f}'): <12}" for value in values]))
    # print(f"{model:<20}"+''.join([f"{(f'{value[1]:.0f}' + f', {value[2]:.0f}'): <12}" for value in values]))
    print(f"{model:<20}"+''.join([f"{(f'{value[1]:.0f}' + f', {value[3]:.0f}'): <12}" for value in values]))
    



for k, vs in dataset.items():
    print(f"{k:<20}TOT_len: {str([sum([sum(i[:2]) for i in v]) for v in vs]):<20}AVG_out: {str([sum([i[1] for i in v]) / len(v) for v in vs]):<20}")




# draw lines to show the change between different categories for each model

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
for count, (model, categorys) in enumerate(new_dataset.items()):
    if len(categorys.keys()) == 0:
        continue
    else:
        types = category_strs
    vs = [categorys[t] for t in types]
    values = [(sum([i[0] for i in v]) / len(v), sum([i[1] for i in v]) / len(v)) for v in vs]
    ys = [v[1] for v in values]
    xs = range(len(types))
    ax.plot(xs, ys, label=model)


ax.set(xlabel='category', ylabel='avg output len',)
ax.grid()
plt.xticks(range(len(category_strs)), category_strs, rotation=20)
plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncols=3)
plt.tight_layout()
fig.savefig(f"./collect_output_lengths/no_robot/avg_outlens.png")
plt.show()





fig, ax = plt.subplots()
for count, (model, categorys) in enumerate(new_dataset.items()):
    if len(categorys.keys()) == 0:
        continue
    else:
        types = category_strs
    vs = [categorys[t] for t in types]
    values = [(sum([i[0] for i in v]) / len(v), 
               sum([i[1] for i in v]) / len(v), 
               np.std([i[1] for i in v])) for v in vs]
    ys = [v[2] for v in values]
    xs = range(len(types))
    ax.plot(xs, ys, label=model)


ax.set(xlabel='category', ylabel='output len std',)
ax.grid()
plt.xticks(range(len(category_strs)), category_strs, rotation=20)
plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncols=3)
plt.tight_layout()
fig.savefig(f"./collect_output_lengths/no_robot/std_outlens.png")
plt.show()



# print the cumulative distribution curve for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np


for count, (model, categorys) in enumerate(new_dataset.items()):
    if len(categorys.keys()) == 0:
        continue
    else:
        types = category_strs
    vs = [categorys[t] for t in types]
    fig, ax = plt.subplots()
    for t, lens in zip(types, vs):
        outs = [i[1] for i in lens]
        outs = np.asarray(outs)
        outs = outs[outs<=4096]
        if len(outs) == 0:
            continue
        elements, counts = np.unique(outs, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_counts = cum_counts/cum_counts[-1]
        ax.scatter(elements, cum_counts, s=2, label = f'{t} #:{len(outs)}')
        ax.set(xlabel='out len', ylabel='cum num')
            #    title='About as simple as it gets, folks')
        if max(outs) > 4096:
            print(k)
    ax.grid()
    plt.legend()
    fig.savefig(f"Cost_Model_per_iter/figures/outlen_cum_distribution_{model}_norobot.pdf")
    plt.show()




# print the cumulative totlen distribution curve for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np


for count, (model, categorys) in enumerate(new_dataset.items()):
    if len(categorys.keys()) == 0:
        continue
    else:
        types = category_strs
    vs = [categorys[t] for t in types]
    fig, ax = plt.subplots()
    for t, lens in zip(types, vs):
        tots = [i[0] + i[1] for i in lens]
        tots = np.asarray(tots)
        tots = tots[tots<=4096]
        if len(tots) == 0:
            continue
        elements, counts = np.unique(tots, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_counts = cum_counts/cum_counts[-1]
        ax.scatter(elements, cum_counts, s=2, label = t)
        ax.set(xlabel='tot len', ylabel='cum num')
            #    title='About as simple as it gets, folks')
        if max(tots) > 4096:
            print(k)
    ax.grid()
    plt.legend()
    fig.savefig(f"Cost_Model_per_iter/figures/totlen_cum_distribution_{model}_norobot.pdf")
    plt.show()












# print the outlen cumulative distribution curve per inplen for each model---------------------------
import matplotlib.pyplot as plt
import numpy as np


for count, (model, categorys) in enumerate(new_dataset.items()):
    if len(categorys.keys()) == 0:
        continue
    else:
        types = category_strs
    vs = [categorys[t] for t in types]
    for t, lens in zip(types, vs):
        inps = [i[0] for i in lens]
        outs = [i[1] for i in lens]
        tots = [i[0] + i[1] for i in lens]
        inps = np.asarray(inps)
        outs = np.asarray(outs)
        tots = np.asarray(tots)
        inps = inps[tots<=4096]
        outs = outs[tots<=4096]
        if len(inps) == 0:
            continue
        # 
        fig, ax = plt.subplots()
        interval = 100
        for i in range(max(inps)//100):
            start, end = (i*interval, (i+1)*interval)
            indices = (inps>start) * (inps<=end)
            tmp_outs = outs[indices]
            if len(tmp_outs) == 0:
                continue
            # compute cumsum
            elements, counts = np.unique(tmp_outs, return_counts=True)
            cum_counts = np.cumsum(counts)
            cum_counts = cum_counts/cum_counts[-1]
            # ax.scatter(elements, cum_counts, s=2, label = f'{start, end}')
            ax.plot(elements, cum_counts, marker='1', markersize=8, label = f'{start, end} #:{sum(counts)}')
        ax.set(xlabel=f'out len, #:{len(inps)}', ylabel='cum num')
            #    title='About as simple as it gets, folks')
        ax.grid()
        plt.legend()
        fig.savefig(f"Cost_Model_per_iter/figures/outlen_per_inplen_{model}_{t}_norobot.pdf")
        plt.show()









