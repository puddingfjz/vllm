""" This file does analysis to get good estimation for the overall computation throughput. """

import json

filename = 'baseline_tp1_llama2_7b_2.log'
surfix = 'eos_'
filename = 'baseline_tp1_llama2_7b_3.log'
surfix = ''

def get_lens(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'output_lens =' in line:
                pos = len('output_lens =')
                values = json.loads(line[pos:])
                return values



lens = get_lens(filename)
inps = [i[0] for i in lens]
outs = [i[1] for i in lens]


def get_per_iter_records(filename):
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'TFLOPs Time:' in line:
                items = line.split(' ')
                # ['(4,', 4086,', '2084896)', '28.07862132736', 'TFLOPs', 'Time:', '0.2851552767679095', 's', '0.00032312609255313873', 's', '0.27753941575065255', 's', '0.004479971248656511', 's', 'Throughput:', '98.46783003848617', 'TFLOPs/s', '101.16985096122868', 'TFLOPs/s']
                values = items[0][1:-1], items[1][:-1], items[2][:-1], items[3], items[6], items[8], items[10], items[12], items[15], items[17]
                values = [float(i) for i in values]
                ret.append(values)
    return ret



per_iter_records = get_per_iter_records(filename)


# compute average throughput
tot_flops = sum([i[3] for i in per_iter_records])
tot_time = sum([i[4] for i in per_iter_records])
avg_throughput = tot_flops / tot_time


# sampling 真的占了很大一部分时间，而且这部分时间很大程度是因为分配内存。
# 现在统计一下每个iter参与的seq数量。

# 直接根据总token量估计总iteration数
max_token_per_iter = 114859
tot_tokens = sum(inps) + sum([(i+i+j-1)*j//2 for i, j in zip(inps, outs)])
estimated_iter_num = tot_tokens / max_token_per_iter
estimated_iter_num = 900 
# 900 这个结果和 len(per_iter_records) 差别很大啊。所以不能这么估计。
# 而且这么估计我们也算不出来平均每个iter的seq num是多少。没法估计每个iter的时间从而估计整体时间。


# print the seq num per iter
seq_nums = [i[0] for i in per_iter_records if i[0] == i[1]]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(range(len(seq_nums)), seq_nums)
ax.set(xlabel='iter', ylabel='seq_num',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}seq_nums{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()





# print the latency per iter
latencys = [i[4] for i in per_iter_records if i[0] == i[1]]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(range(len(latencys)), latencys)
ax.set(xlabel='iter', ylabel='per iter latency (s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}per_iter_latencys{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()


# print the exec latency per iter
exec_latencys = [i[6] for i in per_iter_records if i[0] == i[1]]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(range(len(exec_latencys)), exec_latencys)
ax.set(xlabel='iter', ylabel='per iter exec latencys (s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}per_iter_exec_latencys{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()



# print the exec throughput latency per iter
exec_throughputs = [i[-1] for i in per_iter_records if i[0] == i[1]]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(range(len(exec_throughputs)), exec_throughputs)
ax.set(xlabel='iter', ylabel='per iter exec throughputs (TFLOPs/s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}per_iter_exec_throughputs{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()




# print the latency per iter
latencys = [i[4] for i in per_iter_records if i[0] == i[1]]
exec_latencys = [i[6] for i in per_iter_records if i[0] == i[1]]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(range(len(latencys)), latencys, label='latency')
ax.plot(range(len(latencys)), exec_latencys, label='exec latency')
ax.set(xlabel='iter', ylabel='per iter latency (s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures/{surfix}per_iter_latencysNexec{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()






# print the average exec latency per iter changing curve
import numpy as np
exec_latencys = [i[6] for i in per_iter_records if i[0] == i[1]]
avg_exec_latencys = np.cumsum(exec_latencys) / np.arange(1, len(exec_latencys)+1)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(range(len(exec_latencys)), avg_exec_latencys)
ax.set(xlabel='iter', ylabel='cum exec latency / iter num (s/iter#)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}avg_exec_latencys{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()



# print the per iter exec latency for both prefilling and decoding stages
exec_latencys = [(ind, i[6]) for ind, i in enumerate(per_iter_records) if i[0] == i[1]]
prefill_exec_latencys = [(ind, i[6]) for ind, i in enumerate(per_iter_records) if i[0] != i[1]]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.scatter([i[0] for i in exec_latencys], [i[1] for i in exec_latencys], label='decoding', s=4)
ax.scatter([i[0] for i in prefill_exec_latencys], [i[1] for i in prefill_exec_latencys], label='prefilling', s=4)
ax.set(xlabel='iter', ylabel='exec latency (s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}PrefillNDecode_per_iter_exec_latencys{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()









# print the lengths distribution---------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

elements, counts = np.unique(outs, return_counts=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(elements, counts, s=2)
ax.set(xlabel='out len', ylabel='num',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}outlen_distribution{'llama27b'}_{1}_0.{9}_{2}.pdf")
plt.show()


# print the tot length distribution
elements, counts = np.unique(np.asarray(outs)+np.asarray(inps), return_counts=True)

fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(elements, counts, s=10)
ax.set(xlabel='out len', ylabel='num',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}totlen_distribution{'llama27b'}_{1}_0.{9}_{2}.png")
plt.show()


# print out len vs inp len
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(inps, outs, s=2)
ax.set(xlabel='inp len', ylabel='out len',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}outlen_vs_inplen{'llama27b'}_{1}_0.{9}_{2}.pdf")
plt.show()


# print cumulative distribution curves
# 看起来累积概率分布函数还是可以拟合的
# 感觉需要为每个model的output Length绘制这个累积概率分布函数
elements, counts = np.unique(outs, return_counts=True)
cum_counts = np.cumsum(counts)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(elements, cum_counts, s=2)
ax.set(xlabel='out len', ylabel='cum num',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures/{surfix}outlen_cum_distribution{'llama27b'}_{1}_0.{9}_{2}.pdf")
plt.show()








# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# obtain data from log file directly (if there is something wrong during the measurement and
# we did not get the my_throughput_logger information in the end of the log file)


def get_exec_time(filename):
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'exec time:' in line:
                pos = len('exec time:')
                value = float(line[pos:])
                ret.append(value)
    return ret



def get_sample_time(filename):
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'sample time:' in line:
                pos = len('sample time:')
                value = float(line[pos:])
                ret.append(value)
    return ret


def get_prefill_step_metadata_WRONG_VERSION(filename):
    # prefilling: seq num: 2,tot_tokens: (tensor(2048, device='cuda:0'), tensor(2097152, device='cuda:0')),regarded as decode: (2048, 1047552)
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'prefilling: seq num:' in line:
                pos = line.find('regarded as decode: (') + len('regarded as decode: (')
                part_line = line[pos:]
                pos = part_line.find(')')
                part_line = part_line[:pos]
                seq_num, tot_token_num = part_line.split(',')
                seq_num = int(seq_num)
                tot_token_num = int(tot_token_num)
                ret.append((seq_num, tot_token_num))
    return ret



def get_prefill_step_metadata(filename):
    # prefilling: seq num: 2,tot_tokens: (tensor(2048, device='cuda:0'), tensor(2097152, device='cuda:0')),regarded as decode: (2048, 1047552)
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'prefilling: seq num:' in line:
                pos = line.find(',')
                B = int(line[len('prefilling: seq num:'):pos])
                # 
                pos = line.find('regarded as decode: (') + len('regarded as decode: (')
                part_line = line[pos:]
                pos = part_line.find(')')
                part_line = part_line[:pos]
                tot_token_num, _ = part_line.split(',')
                tot_token_num = int(tot_token_num)
                max_seq_len = (tot_token_num + B - 1) // B
                ret.append((B, max_seq_len))
    return ret







def get_decode_step_metadata(filename):
    # get seq num and tot context len
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'decoding:' in line:
                pos1 = len('decoding: seq num:')
                pos2 = line[pos1:].find(',') + pos1
                seq_num = int(line[pos1:pos2])
                pos3 = line[pos1:].find(':') + pos1 + 1
                tot_content_len = int(line[pos3:])
                ret.append((seq_num, tot_content_len))
    return ret




def cal_flops_for_llama2_7b_WRONG_VERSION(T, context_tot_len):
    V = 32000
    h = 4096
    I = 11008
    L = 32
    return 2*T*V*h+L*(4*T*h*h+2*context_tot_len*h+3*I*h*T)



def cal_flops_for_llama2_7b(B, s, context_tot_len, is_prompt):
    '''
    s = max_seq_len
    For prefill (padding to max_seq_len):
        L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)
    For decoding:
        L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)
    '''
    h = 4096
    I = 11008
    L = 32
    if is_prompt:
        return L*( 4*B*s*h*h + 2*B*s*s*h + 3*B*s*h*I)
    else:
        s = 1
        return L*( 4*B*s*h*h + 2*h*context_tot_len + 3*I*B*s*h)





import json
import numpy as np

filename = 'Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_1.log'


exec_times = get_exec_time(filename=filename)
sample_times = get_sample_time(filename=filename)
prefill_metadatas = get_prefill_step_metadata(filename=filename)
flops = cal_flops_for_llama2_7b(
    np.asarray([i[0] for i in prefill_metadatas]), 
    np.asarray([i[1] for i in prefill_metadatas]), 
    None, True) / 1e12


exec_throughputs = flops / np.asarray(exec_times)



# plot the throughput vs flops figure
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
ax.scatter(flops, exec_throughputs, s=2)
ax.set(xlabel='flops (TFLOPs)', ylabel='throughput (TFLOPs/s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures_per_iter_latency/Throughput_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test3.pdf")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
ax.scatter(flops, exec_times, s=2)
ax.set(xlabel='flops (TFLOPs)', ylabel='exec latency (s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
fig.savefig(f"./figures_per_iter_latency/Latency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test3.pdf")
plt.show()




# 分析一下奇怪的数据
indices = np.argsort(flops)
flops[indices][:10]
exec_times = np.asarray(exec_times)
exec_times[indices][:10]
exec_throughputs[indices][:10]

# [prefill_metadatas[i] for i in indices][:10]

prefill_metadatas = np.asarray(prefill_metadatas)
prefill_metadatas[indices][:100]





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 重新看一下小规模数据集上的数据~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_per_iter_records(filename):
    ret = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'TFLOPs Time:' in line:
                items = line.split(' ')
                # ['(4,', 4086,', '2084896)', '28.07862132736', 'TFLOPs', 'Time:', '0.2851552767679095', 's', '0.00032312609255313873', 's', '0.27753941575065255', 's', '0.004479971248656511', 's', 'Throughput:', '98.46783003848617', 'TFLOPs/s', '101.16985096122868', 'TFLOPs/s']
                # (400, 424, 24) 2.85698162688 TFLOPs Time: 0.15687668323516846 s 0.06678195018321276 s 0.007101176306605339 s Throughput: 18.211639664749903 TFLOPs/s 42.78074568116118 TFLOPs/s
                values = items[0][1:-1], items[1][:-1], items[2][:-1], items[3], items[6], items[8], items[10], items[13], items[15]
                values = [float(i) for i in values]
                ret.append(values)
    return ret


filename = 'Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_4.log'
per_iter_records = get_per_iter_records(filename)
# exec_throughputs = [i[-1] for i in per_iter_records]
exec_latencys = np.asarray([i[5] for i in per_iter_records])
# flops = [i[3] for i in per_iter_records]
seq_nums = np.asarray([i[0] for i in per_iter_records])
max_seq_lens = (np.asarray([i[1] for i in per_iter_records]) + seq_nums - 1)//seq_nums
flops = cal_flops_for_llama2_7b(
    seq_nums, 
    max_seq_lens, 
    None, True) / 1e12


exec_throughputs = flops / exec_latencys
# plot the throughput vs flops figure
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
for i in sorted(set(seq_nums)):
    indices = [j for j,k in enumerate(seq_nums) if k==i]
    tmp_flops = [flops[j] for j in indices]
    tmp_exec_throughputs = [exec_throughputs[j] for j in indices]
    ax.scatter(tmp_flops, tmp_exec_throughputs, s=2, label=f"#seq:{i}")


ax.set(xlabel='flops (TFLOPs)', ylabel='throughput (TFLOPs/s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/Throughput_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test4.pdf")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
for i in sorted(set(seq_nums)):
    indices = [j for j,k in enumerate(seq_nums) if k==i]
    tmp_flops = [flops[j] for j in indices]
    tmp_exec_latencys = [exec_latencys[j] for j in indices]   
    ax.scatter(tmp_flops, tmp_exec_latencys, s=2, label=f"#seq:{i}")


ax.set(xlabel='flops (TFLOPs)', ylabel='exec latency (s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/Latency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test4.pdf")
plt.show()



# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# 分析一下decoding阶段的数据
import json
import numpy as np

filename = 'Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_2.log'

decode_metadatas = get_decode_step_metadata(filename=filename)[:-1]
exec_times = get_exec_time(filename=filename)[-len(decode_metadatas):]
sample_times = get_sample_time(filename=filename)[-len(decode_metadatas):]
flops = cal_flops_for_llama2_7b(
    np.asarray([i[0] for i in decode_metadatas]), 
    1,
    np.asarray([i[1] for i in decode_metadatas]), 
    False) / 1e12


exec_throughputs = flops / np.asarray(exec_times)

# too many points to plot here, maybe just print a part of the points.
seq_nums = [1, 2, 3, 4, 5, 6, 10, 105]
tmp_exec_times = list()
tmp_exec_throughputs = list()
tmp_flops = list()
for seq_num in seq_nums:
    indices = [i for i,j in enumerate(decode_metadatas) if j[0] == seq_num]
    tmp_exec_times.append([exec_times[i] for i in indices])
    tmp_exec_throughputs.append([exec_throughputs[i] for i in indices])
    tmp_flops.append([flops[i] for i in indices])



# plot the throughput vs flops figure
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
for i in range(len(seq_nums)):
    ax.scatter(tmp_flops[i], tmp_exec_throughputs[i], s=2, label=f'#seq:{seq_nums[i]}')


ax.set(xlabel='flops (TFLOPs)', ylabel='throughput (TFLOPs/s)',)
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/DecodeThroughput_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test3.pdf")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
for i in range(len(seq_nums)):
    ax.scatter(tmp_flops[i], tmp_exec_times[i], s=2, label=f'#seq:{seq_nums[i]}')


ax.set(xlabel='flops (TFLOPs)', ylabel='exec latency (s)')
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/DecodeLatency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test3.pdf")
plt.show()









# ===============================================================
# ===============================================================
# ===============================================================
# ===============================================================
# ===============================================================
# ===============================================================
# 验证decoding阶段的latency可以被建模成 latency = a*FLOPs1+b*FLOPs2+c, where FLOPs1, FLOPs2 are flops for non-attention ops and attention


import json
import numpy as np


def cal_attention_flops_for_llama2_7b(B, s, context_tot_len, is_prompt):
    '''
    s = max_seq_len
    For prefill (padding to max_seq_len):
        L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)
    For decoding:
        L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)
    '''
    h = 4096
    I = 11008
    L = 32
    if is_prompt:
        return L*(2*B*s*s*h)
    else:
        s = 1
        return L*( 2*h*context_tot_len)



def cal_nonattention_flops_for_llama2_7b(B, s):
    '''
    s = max_seq_len
    For prefill (padding to max_seq_len):
        L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)
    For decoding:
        L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)
    '''
    h = 4096
    I = 11008
    L = 32
    return L*( 4*B*s*h*h + 3*B*s*h*I)




filename = 'Llama-2-7b-hf_0425_tp1_temp1.0_wldeg2_2.log'

decode_metadatas = get_decode_step_metadata(filename=filename)[:-1]
exec_times = get_exec_time(filename=filename)[-len(decode_metadatas):]
sample_times = get_sample_time(filename=filename)[-len(decode_metadatas):]

non_attention_flops = cal_nonattention_flops_for_llama2_7b(
    np.asarray([i[0] for i in decode_metadatas]), 
    1) / 1e12


attention_flops = cal_attention_flops_for_llama2_7b(
    np.asarray([i[0] for i in decode_metadatas]), 
    1,
    np.asarray([i[1] for i in decode_metadatas]), 
    False) / 1e12



# exec_throughputs = flops / np.asarray(exec_times)
assert len(decode_metadatas) == len(exec_times)

# too many points to plot here, maybe just print a part of the points.
seq_nums = [10, 105]
seq_nums = list(set([i[0] for i in decode_metadatas]))
tmp_exec_times = list()
# tmp_exec_throughputs = list()
tmp_attention_flops = list()
for seq_num in seq_nums:
    indices = [i for i,j in enumerate(decode_metadatas) if j[0] == seq_num]
    tmp_exec_times.append([exec_times[i] for i in indices])
    # tmp_exec_throughputs.append([exec_throughputs[i] for i in indices])
    tmp_attention_flops.append([attention_flops[i] for i in indices])



# plot attention flops 的增量和latency之间的增量之间的关系

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# ax.plot(range(len(seq_nums)), seq_nums)
# for i in range(len(seq_nums)):
for i in [1,10, 20, 30, 40, 50, 60, 70, 80, 90, 105]:
    xs = np.asarray(tmp_attention_flops[i])
    ys = np.asarray(tmp_exec_times[i])
    index = np.argmin(xs)
    xs = xs - xs[index]
    ys = ys - ys[index]
    ax.scatter(xs, ys, s=2, label=f'#seq:{seq_nums[i]}')
    # ax.scatter(xs, ys, s=2)


ax.set(xlabel='delta attention flops (TFLOPs)', ylabel='delta exec latency (s)')
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/delta_attention_flops_Vs_DecodeLatency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test.pdf")
plt.show()



# plot non attention flops 自身和假定没有attention op的时候的latency之间的关系~~~~~~~~~~~~~~~~~~~~~~

# too many points to plot here, maybe just print a part of the points.
seq_nums = [10, 105]
seq_nums = list(set([i[0] for i in decode_metadatas]))
tmp_exec_times = list()
# tmp_exec_throughputs = list()
tmp_nonattention_flops = list()
tmp_attention_flops = list()
for seq_num in seq_nums:
    indices = [i for i,j in enumerate(decode_metadatas) if j[0] == seq_num]
    tmp_exec_times.append([exec_times[i] for i in indices])
    # tmp_exec_throughputs.append([exec_throughputs[i] for i in indices])
    tmp_nonattention_flops.append([non_attention_flops[i] for i in indices])
    tmp_attention_flops.append([attention_flops[i] for i in indices])



import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
plot_xs = list()
plot_ys = list()
for i in range(len(seq_nums)):
# for i in [1,10, 20, 30, 40, 50, 60, 70, 80, 90, 105]:
    x = tmp_nonattention_flops[i][0]
    ys = np.asarray(tmp_exec_times[i])
    zs = np.asarray(tmp_attention_flops[i])
    ind1 = np.argmin(zs)
    ind2 = np.argmax(zs)
    coeff = (ys[ind1] - ys[ind2]) / (zs[ind1] - zs[ind2])
    y = ys[ind1] - zs[ind1]*coeff
    plot_xs.append(x)
    plot_ys.append(y)


# ax.scatter(plot_xs, plot_ys, s=2, label=f'#seq:{seq_nums[i]}')
ax.scatter(plot_xs, plot_ys, s=2)

ax.set(xlabel='non-attention flops (TFLOPs)', ylabel='exec latency (s)')
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/non_attention_flops_Vs_DecodeLatency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test.pdf")
plt.show()




import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
plot_xs = list()
plot_ys = list()
for i in range(len(seq_nums)):
# for i in [1,10, 20, 30, 40, 50, 60, 70, 80, 90, 105]:
    x = tmp_nonattention_flops[i][0]
    ys = np.asarray(tmp_exec_times[i])
    zs = np.asarray(tmp_attention_flops[i])
    ind1 = np.argmin(zs)
    ind2 = np.argmax(zs)
    coeff = (ys[ind1] - ys[ind2]) / (zs[ind1] - zs[ind2])
    y = ys[ind1] - zs[ind1]*coeff
    plot_xs.append(x)
    plot_ys.append(x/y)


# ax.scatter(plot_xs, plot_ys, s=2, label=f'#seq:{seq_nums[i]}')
ax.scatter(plot_xs, plot_ys, s=2)

ax.set(xlabel='non-attention flops (TFLOPs)', ylabel='exec throughput (TFLOPs/s)')
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/non_attention_flops_Vs_DecodeThroughput_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test.pdf")
plt.show()




# 直接画每个non-attention flops最大tot token num的时候的点，因为attention之和tot token num有关
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
plot_xs = list()
plot_ys = list()
for i in range(len(seq_nums)):
# for i in [1,10, 20, 30, 40, 50, 60, 70, 80, 90, 105]:
    x = tmp_nonattention_flops[i][0]
    y = max(tmp_exec_times[i])
    plot_xs.append(x)
    plot_ys.append(y)


# ax.scatter(plot_xs, plot_ys, s=2, label=f'#seq:{seq_nums[i]}')
ax.scatter(plot_xs, plot_ys, s=2)

ax.set(xlabel='non-attention flops (TFLOPs)', ylabel='exec latency (s)')
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
fig.savefig(f"./figures_per_iter_latency/non_attention_flops_Vs_DecodeLatency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test2.pdf")
plt.show()


















































# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# Analyse the sampling cost VS (seqnum, tot_token_num)

import matplotlib.pyplot as plt
import numpy as np
from vllm.engine.metrics import MyThroughputLogger

# filename = './Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_sample1.log'
filename = './Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_sample2.log'
per_iter_records = list()
is_prompt = None
with open(filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if '[[' in line:
            seqnum, context_tot_len, is_prompt, sample_latency = \
                MyThroughputLogger.get_seqnum_contextTotLen_isprompt_sampleLatency(line=line)
            per_iter_records.append((seqnum, context_tot_len, is_prompt, sample_latency))




# plot sample_latency VS context_tot_len per seqnum
seq_nums = np.asarray([i[0] for i in per_iter_records])
context_tot_lens = np.asarray([i[1] for i in per_iter_records])
sample_latencys = np.asarray([i[-1] for i in per_iter_records])

fig, ax = plt.subplots()
for i in np.unique(seq_nums)[::10]:
# for i in [451]:
    indices = (seq_nums == i)
    plot_xs = context_tot_lens[indices][1:]
    plot_ys = sample_latencys[indices][1:]
    ax.plot(plot_xs, plot_ys, label=f'#seq:{i}')
    # ax.scatter(plot_xs, plot_ys, s=2, label=f'#seq:{i}')
    # ax.scatter(plot_xs, plot_ys, s=2)


ax.set(xlabel='context tot len', ylabel='sample latency (s)')
    #    title='About as simple as it gets, folks')
ax.grid()
plt.legend()
tag = '' if is_prompt else 'Decode'
fig.savefig(f"./figures_per_iter_latency/context_tot_len_Vs_{tag}SampleLatency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test1.pdf")
plt.show()








# compare sum(latencys) with sum(exec_latencys) + sum(sample_latencys)
from vllm.engine.metrics import MyThroughputLogger
filename = './Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_sample1.log'
# filename = './baseline_tp1_llama2_7b_4.log'
filename = './baseline_tp1_llama2_7b_5.log'
per_iter_records = list()
with open(filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if '[[' == line[:2]:
            latencys = \
                MyThroughputLogger.get_all_type_latencys(line=line)
            per_iter_records.append(latencys)


print(sum([i[0] for i in per_iter_records]), sum([i[1] for i in per_iter_records])+sum([i[2] for i in per_iter_records]))








# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# Analyse the preparing-input cost VS (seqnum, tot_token_num, tot_padded_token_num [maybe for prefill stages])


import matplotlib.pyplot as plt
import numpy as np
from vllm.engine.metrics import MyThroughputLogger

filename = './Llama-2-7b-hf_0502_tp1_temp1.0_wldeg2_prepInp3.log'
per_iter_records = list()
is_prompt = None
with open(filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if '[[' in line:
            seqnum, context_tot_len, max_seqlen, is_prompt, prepInp_latency = \
                MyThroughputLogger.get_seqnum_contextTotLen_maxSeqlen_isprompt_prepInpLatency(line=line)
            per_iter_records.append((seqnum, context_tot_len, max_seqlen, is_prompt, prepInp_latency))




# plot sample_latency VS context_tot_len per seqnum
seq_nums = np.asarray([i[0] for i in per_iter_records])
context_tot_lens = np.asarray([i[1] for i in per_iter_records])
max_seqlens = np.asarray([i[2] for i in per_iter_records])
is_prompts = np.asarray([i[3] for i in per_iter_records])
prepInp_latencys = np.asarray([i[-1] for i in per_iter_records])

for is_prompt in [False, True]:
    fig, ax = plt.subplots()
    for i in np.unique(seq_nums)[::1]:
    # for i in [451]:
        indices = (seq_nums == i) * (is_prompts == is_prompt)
        if sum(indices) == 0:
            continue
        plot_xs = None
        if is_prompt:
            plot_xs = seq_nums[indices] * max_seqlens[indices]
            # plot_xs = context_tot_lens[indices]
        else:
            # plot_xs = context_tot_lens[indices]
            plot_xs = seq_nums[indices] * max_seqlens[indices]
        plot_ys = prepInp_latencys[indices]
        ax.plot(plot_xs, plot_ys, label=f'#seq:{i}')
        # ax.scatter(plot_xs, plot_ys, s=2, label=f'#seq:{i}')
        # ax.scatter(plot_xs, plot_ys, s=2)
        print(f"#seq:{i}  {min(plot_ys[:2])}s-{min(plot_ys[-2:])}s")
    # 
    ax.set(xlabel='context tot len', ylabel='prepInp latency (s)')
        #    title='About as simple as it gets, folks')
    ax.grid()
    plt.legend()
    tag = 'padded_tot_len_Vs_' if is_prompt else 'context_tot_len_Vs_Decode'
    fig.savefig(f"./figures_per_iter_latency/{tag}PrepInpLatency_Llama-2-7b-hf_tp{1}_temp{1.0}_wldeg{2}_test2.pdf")
    plt.show()







# compare sum(latencys) with sum(exec_latencys) + sum(sample_latencys)
from vllm.engine.metrics import MyThroughputLogger
filename = './baseline_tp1_llama2_7b_6.log'
per_iter_records = list()
with open(filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if '[[' == line[:2]:
            latencys = \
                MyThroughputLogger.get_all_type_latencys(line=line)
            per_iter_records.append(latencys)


print(sum([i[0]-i[-1] for i in per_iter_records]), sum([sum(i[1:-1]) for i in per_iter_records]))





# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# Analyse the exec, sample, prepInp cost for different exec plans




import matplotlib.pyplot as plt
import numpy as np
from vllm.engine.metrics import MyThroughputLogger


def get_per_iter_records(filename):
    per_iter_records = list()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if '[[' == line[:2]:
                seqnum, context_tot_len, max_seqlen, is_prompt, prepInp_latency = \
                    MyThroughputLogger.get_seqnum_contextTotLen_maxSeqlen_isprompt_prepInpLatency(line=line)
                _, _, flops, exec_latency = MyThroughputLogger.get_seqnum_isprompt_flops_execLatency(line=line)
                _, _, _, sample_latency = MyThroughputLogger.get_seqnum_contextTotLen_isprompt_sampleLatency(line=line)
                per_iter_records.append((
                    seqnum, context_tot_len, max_seqlen, flops, is_prompt, 
                    exec_latency, sample_latency, prepInp_latency))
    return per_iter_records[2:]



def get_lists_from_records(per_iter_records):
    seq_nums = np.asarray([i[0] for i in per_iter_records])
    context_tot_lens = np.asarray([i[1] for i in per_iter_records])
    max_seqlens = np.asarray([i[2] for i in per_iter_records])
    flops = np.asarray([i[3] for i in per_iter_records])
    is_prompts = np.asarray([i[4] for i in per_iter_records])
    exec_latencys = np.asarray([i[5] for i in per_iter_records])
    sample_latencys = np.asarray([i[6] for i in per_iter_records])
    prepInp_latencys = np.asarray([i[-1] for i in per_iter_records])
    return seq_nums, context_tot_lens, max_seqlens, flops, is_prompts, exec_latencys, sample_latencys, prepInp_latencys



def plot_exec(seq_nums, xs, ys, is_prompts, xlabel, ylabel, tag, model):
    # plot sample_latency VS context_tot_len per seqnum
    for is_prompt in [False, True]:
        fig, ax = plt.subplots()
        # for i in np.unique(seq_nums)[::1]:
        for i in [51]:
            indices = (seq_nums == i) * (is_prompts == is_prompt)
            if sum(indices) == 0:
                continue
            # plot_xs = seq_nums[indices] * max_seqlens[indices]
            plot_xs = xs[indices]
            plot_ys = ys[indices]
            # ax.plot(plot_xs, plot_ys, label=f'#seq:{i}')
            ax.scatter(plot_xs, plot_ys, s=2, label=f'#seq:{i}')
            # ax.scatter(plot_xs, plot_ys, s=2)
            print(f"#seq:{i}  {min(plot_ys[:2])}s-{min(plot_ys[-2:])}s")
        # 
        # ax.set(xlabel='context tot len', ylabel='prepInp latency (s)')
        ax.set(xlabel=xlabel, ylabel=ylabel)
            #    title='About as simple as it gets, folks')
        ax.grid()
        plt.legend()
        # tag = 'padded_tot_len_Vs_' if is_prompt else 'context_tot_len_Vs_Decode'
        decode_tag = '' if is_prompt else 'Decode'
        fig.savefig(f"./figures_per_iter_latency/{decode_tag}_{tag}_{model}.pdf")
        plt.show()




# for (tp, wldeg) in [(2, 2), (4, 2), (1, 8), (1, 16), (2, 8), (2, 16)]:
#     filename = f'./Llama-2-7b-hf_tp{tp}_temp1.0_wldeg{wldeg}_0504_1.log'
    # model = 'Llama-2-7b-hf'
for (tp, wldeg) in [(1, 2), (2, 2), (4, 2), (1, 20), (2, 20)]:
    model = 'Llama-2-13b-hf'
    filename = f'./{model}_tp{tp}_temp1.0_wldeg{wldeg}_0504_1.log'
    per_iter_records = get_per_iter_records(filename)
    seq_nums, context_tot_lens, max_seqlens, flops, is_prompts, \
        exec_latencys, sample_latencys, prepInp_latencys = \
        get_lists_from_records(per_iter_records)
    # 
    plot_exec(seq_nums, flops, exec_latencys, is_prompts, 
              xlabel='flops', ylabel='exec latency (s)', tag=f'flopsVsExec_tp{tp}_wldeg{wldeg}', 
              model=model)
    plot_exec(seq_nums, context_tot_lens, sample_latencys, is_prompts, 
              xlabel='Cl', ylabel='sample latency (s)', tag=f'ClVsSamp_tp{tp}_wldeg{wldeg}', 
              model=model)
    plot_exec(seq_nums, seq_nums*max_seqlens, prepInp_latencys, is_prompts, 
              xlabel='PCl', ylabel='prepInp latency (s)', tag=f'PClVsPrepInp_tp{tp}_wldeg{wldeg}', 
              model=model)
    


