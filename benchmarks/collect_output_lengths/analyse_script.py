# ========================================================
# ========================================================
# ========================================================
# analyse the data for ``no_robot'' dataset
import json

prefix = 'collect_output_lengths/no_robot/'


model_names = [
               'Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 
               'Aquila-7B', 'AquilaChat-7B', 
               'Baichuan2-13B-Chat', 'Baichuan-7B', 
               'chatglm3-6b', 
               'gpt-j-6b', 'gpt-neox-20b', 
               'Mixtral-8x7B-v0.1', 'phi-2',
               'Qwen2-beta-7B', 'Qwen2-beta-7B-Chat']

suffix = '_tp4_0331_10kreq_1.log'


# NEWROUND models
prefix = 'collect_output_lengths/no_robot/NEWROUND_'

model_names = [
                'vicuna-13b-v1.5',
                'oasst-sft-4-pythia-12b-epoch-3.5',
                'alpaca-13b',
                'baize-v2-13b',
                'koala-13B-HF',
                'dolly-v2-12b',
                'mpt-7b-chat',
            ]

suffix = '_tp2_0730_10kreq_1.log'



# NEWROUND models for routerbench
prefix = 'collect_output_lengths/no_robot/NEWROUND_'

model_names = [
                'Llama-2-70b-chat-hf',
                'Mixtral-8x7B-Instruct-v0.1',
                'WizardLM-13B-V1.2',
                'CodeLlama-34b-Instruct-hf',
                'Mistral-7B-Instruct-v0.2',     
            ]

model_paths = [
    'meta-llama/Llama-2-70b-chat-hf',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'WizardLMTeam/WizardLM-13B-V1.2',
    'meta-llama/CodeLlama-34b-Instruct-hf',
    'mistralai/Mistral-7B-Instruct-v0.2',     
]

suffix = '_tp2_1202_10kreq_1.log'
suffix = '_tp2_1205_10kreq_1.log'


dataset = {key:list() for key in model_names}

max_model_len_dict = {key:None for key in model_names}


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




def get_max_model_len_from_engine_args(max_model_len_dict, model_paths):
    import search_exec_plans 
    for model_path in model_paths:
        (model_config, cache_config, parallel_config, scheduler_config,
        device_config, lora_config) = search_exec_plans.get_engin_args(model_path, 1)
        max_model_len = model_config.max_model_len
        pos = model_path.find('/')
        model_name = model_path[pos+1:]
        max_model_len_dict[model_name] = max_model_len


for model in dataset.keys():
    filename = f'{model}{suffix}'
    update_dataset(dataset, model, filename)
    # get_max_model_len(max_model_len_dict, model, filename)


get_max_model_len_from_engine_args(max_model_len_dict, model_paths)



for k, vs in dataset.items():
    print(f"{k:<20}TOT_len: {str([sum([sum(i[:2]) for i in v]) for v in vs]):<20}AVG_out: {str([sum([i[1] for i in v]) / len(v) for v in vs]):<20}")





# ********************************************************
# we try to obtain the outlen distribution for each model 
# (exculding the cases where reqs are stopped due to the max_seq_len constraint).
# ********************************************************
import matplotlib.pyplot as plt
import numpy as np
fig_path_prefix = 'Cost_Model_per_iter/figures/cdf'
fig_path_suffix = 'norobot_0808_1.pdf'
fig_path_prefix = 'Cost_Model_per_iter_zxcpu/figures/cdf'
fig_path_suffix = 'norobot_1202_1.pdf'
fig_path_prefix = 'Cost_Model_per_iter_zxcpu/figures/cdf'
fig_path_suffix = 'norobot_1205_1.pdf'

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
    fig.savefig(f"{fig_path_prefix}_{k}_{fig_path_suffix}")
    plt.show()    



# store the pdf dict to file so that we can directly use it
with open('./collect_output_lengths/no_robot/out_len_sampler_2.py', 'a') as file:
    file.write(f"\npdf_dict.update({json.dumps(pdf_dict)})\n")





