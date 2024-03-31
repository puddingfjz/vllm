""" We do some preparations to construct our cost model using this file """


"""
cost model 要和tensor parallel degree, weight cache degree, gpu block number 数量相关
固定了这几个条件，具体的cost还和sequence number和sequence 总长度有关【总flops和这个优化，总cost根据之前的经验也有关系】。
先分开给每个条件都画一组图，然后再看是拟合函数还是用MLP来predict。
"""


# <jingzhi> about the decoding phase
def cal_flops(T,V,h,I,L,context_tot_len):
    return 2*T*V*h+L*(4*T*h*h+2*context_tot_len*h+3*I*h*T)




# get the profiling result from log files
def get_data_set(file_name):
    data = dict()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        # decoding: seq num: 512, tot_tokens: 193823
        # iter latency: 0.07324683794286102s abs: (766528.330424366, 766528.403671204)s
        seq_num, tot_tokens = None, None
        for line in lines:
            if 'decoding: seq num: ' in line:
                # get feature information
                pos0 = len('decoding: seq num: ')
                pos1 = line.find(',')
                pos2 = line[pos1:].find(':')+pos1+1
                # print(f"seq_num: {line[pos0:pos1]}, tot_tokens: {line[pos2:]}")
                seq_num = int(line[pos0:pos1])
                tot_tokens = int(line[pos2:])
            elif 'iter latency: ' in line:
                # get latency
                if seq_num==None:
                    # we want to get the decoding phase latency
                    continue
                pos0 = len('iter latency: ')
                pos1 = line.find('s abs')
                # print(f"latency: {line[pos0:pos1]}")
                latency = float(line[pos0:pos1])
                data[(seq_num, tot_tokens)] = latency
                seq_num, tot_tokens = None, None
    return data



# V: vocabulary size; h: hidden size; I: intermediate size
model_configs = {'llama_7b': {'V':32000,'h':4096,'I':11008, 'L':32},
                 'llama_13b':{'V':32000,'h':5120,'I':13824, 'L':40},
                 'llama_70b':{'V':32000,'h':8192,'I':28672, 'L':80}}


dataset_info = {'1k': {'tot_tokens': 967676, 'tot_context_lens': 335321634.0}}



tot_flops = dict()
for model in model_configs:
    V, h, I, L = model_configs[model]['V'], model_configs[model]['h'], model_configs[model]['I'], model_configs[model]['L']
    xs = cal_flops(dataset_info['1k']['tot_tokens'], V, h, I, L, dataset_info['1k']['tot_context_lens'])
    print(f"{model}: tot_flops: {xs/1e9} GFLOPS")
    tot_flops[(model, '1k')] = xs



tot_time = {('llama_7b', 1, 'vllm'):120.36901388294064,
            ('llama_7b', 2, 'vllm'):100.09013294405304,
            ('llama_7b', 4, 'vllm'):83.74368678301107,
            ('llama_70b', 4, 'vllm'):199.91717011400033,
            ('llama_70b', 2, 'ours'):337.83783783783787,}


average_throughput = {k:tot_flops[(k[0], '1k')]/v/1e13 for k, v in tot_time.items()}

# NOTE: unit is 1e13 FLOPS/s
average_throughput = {
    ('llama_7b', 1, 'vllm'): 5.489990967625622, 
    ('llama_7b', 2, 'vllm'): 6.602297145201372, 
    ('llama_7b', 4, 'vllm'): 7.8910402011749925, 
    ('llama_70b', 4, 'vllm'): 38.15421789743361,
    ('llama_70b', 2, 'ours'): 22.5779424791049}


'''
llama_7b: tot_flops: 6608247.989993472 GFLOPS
llama_13b: tot_flops: 12732085.948416 GFLOPS
llama_70b: tot_flops: 76276832.69967872 GFLOPS
'''



# log files for ours method
file_name_list = [(f'ours_0226_7b_1_tp{tp}_pd16_gpu0.{gpur}.log', 'llama_7b', 'ours', tp, gpur, 16) for tp in [1,2] for gpur in [5,6,7,8,9]] + \
    [(f'ours_0226_13b_1_tp{tp}_pd20_gpu0.{gpur}.log', 'llama_13b', 'ours', tp, gpur, 20) for tp in [1,2] for gpur in [5,6,7,8,9]] + \
    [(f'ours_0226_70b_1_tp2_pd{pd}_gpu0.9.log', 'llama_70b', 'ours', 2, 9, pd) for pd in [16, 20, 40]]

# log files for vllm method
file_name_list = file_name_list + \
    [(f'vllm_7b_tp{tp}_0315_1kreq_1.log', 'llama_7b', 'vllm', tp, 9, 2) for tp in [1,2,4]] + \
    [(f'vllm_13b_tp{tp}_0315_1kreq_1.log', 'llama_13b', 'vllm', tp, 9, 2) for tp in [1,2,4]] + \
    [(f'vllm_70b_tp4_0315_1kreq_1.log', 'llama_70b', 'vllm', 4, 9, 2), ('vllm_0226_70b_1_tp2.log', 'llama_70b', 'vllm', 2, 9, 2)]



# get the complete dataset
dataset = dict()
for file_name, model, method, tp, gpur, pd in file_name_list:
    print(f"{file_name}")
    data = get_data_set(file_name)
    dataset[(model, tp, gpur, pd)] = data




# plot figures to analyse the cost changing trend


import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
xs_dict = dict()
ys_dict = dict()
for k, v in dataset.items():
    V, h, I, L = model_configs[k[0]]['V'], model_configs[k[0]]['h'], model_configs[k[0]]['I'], model_configs[k[0]]['L']
    Ts = list()
    context_tot_lens = list()
    ys = list()
    for (seq_num, tot_tokens), latency in v.items():
        Ts.append(seq_num)
        context_tot_lens.append(tot_tokens)
        ys.append(latency)
    xs = cal_flops(np.asarray(Ts), V, h, I, L, np.asarray(context_tot_lens))
    xs_dict[k] = xs
    ys_dict[k] = ys


for k in xs_dict:
    model, tp, gpur, pd = k
    xs = xs_dict[k]
    ys = ys_dict[k]
    # 
    fig, ax = plt.subplots()
    ax.scatter(xs, ys,s=10)
    ax.set(xlabel='flops', ylabel='latency (s)',)
        #    title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig(f"./figures/flops_V_latency{model}_{tp}_0.{gpur}_{pd}.png")
    plt.show()






# draw figures about flops / throughput
for k in xs_dict:
    model, tp, gpur, pd = k
    xs = xs_dict[k]
    ys = xs/np.asarray(ys_dict[k])
    # 
    fig, ax = plt.subplots()
    ax.scatter(xs, ys,s=10)
    ax.set(xlabel='flops', ylabel='throughput (flops/s)',)
        #    title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig(f"./figures/flops_V_throughput{model}_{tp}_0.{gpur}_{pd}.png")
    plt.show()



print({k: sum(ys_dict[k]) for k in xs_dict})
{
    ('llama_7b', 1, 5, 16): 70.4880702663213,
    ('llama_7b', 1, 6, 16): 67.43871142715216,
    ('llama_7b', 1, 7, 16): 64.70476378314197,
    ('llama_7b', 1, 8, 16): 64.003512092866,
    ('llama_7b', 1, 9, 16): 63.46261085849255,
    ('llama_7b', 2, 5, 16): 40.04068243037909,
    ('llama_7b', 2, 6, 16): 39.40928698889911,
    ('llama_7b', 2, 7, 16): 39.163972224108875,
    ('llama_7b', 2, 8, 16): 39.38269891217351,
    ('llama_7b', 2, 9, 16): 39.224232078529894,
    ('llama_13b', 1, 5, 20): 160.44254561048,
    ('llama_13b', 1, 6, 20): 136.47538242023438,
    ('llama_13b', 1, 7, 20): 125.6946879317984,
    ('llama_13b', 1, 8, 20): 118.61462431680411,
    ('llama_13b', 1, 9, 20): 113.25763321854174,
    ('llama_13b', 2, 5, 20): 67.02398668043315,
    ('llama_13b', 2, 6, 20): 63.50744661036879,
    ('llama_13b', 2, 7, 20): 63.01945680193603,
    ('llama_13b', 2, 8, 20): 62.472794140689075,
    ('llama_13b', 2, 9, 20): 61.958249635994434,
    ('llama_70b', 2, 9, 16): 194.72765936795622,
    ('llama_70b', 2, 9, 20): 196.28665845375508,
    ('llama_70b', 2, 9, 40): 192.96710703056306,
    ('llama_7b', 1, 9, 2): 61.30582461983431,
    ('llama_7b', 2, 9, 2): 38.52509520598687,
    ('llama_7b', 4, 9, 2): 27.01033983938396,
    ('llama_13b', 1, 9, 2): 108.77647399215493,
    ('llama_13b', 2, 9, 2): 61.34749145247042,
    ('llama_13b', 4, 9, 2): 37.56811678700615,
    ('llama_70b', 4, 9, 2): 106.07120659807697,
    ('llama_70b', 2, 9, 2): 248.54294767044485,
}


print({k: max(xs_dict[k]/np.asarray(ys_dict[k]))/1e13 for k in xs_dict})
{('llama_7b', 1, 5, 16): 4.488782439855139, 
 ('llama_7b', 1, 6, 16): 4.429667225275381, 
 ('llama_7b', 1, 7, 16): 4.363097230400571, 
 ('llama_7b', 1, 8, 16): 4.768437359518608, 
 ('llama_7b', 1, 9, 16): 4.706104453849758, 
 ('llama_7b', 2, 5, 16): 7.799186097906833, 
 ('llama_7b', 2, 6, 16): 7.754022273915872, 
 ('llama_7b', 2, 7, 16): 7.735862332118531,
 ('llama_7b', 2, 8, 16): 7.710249800992397, 
 ('llama_7b', 2, 9, 16): 7.718257192540528, 
 ('llama_13b', 1, 5, 20): 4.633369844905017, 
 ('llama_13b', 1, 6, 20): 5.770886376679108, 
 ('llama_13b', 1, 7, 20): 5.006874736406543, 
 ('llama_13b', 1, 8, 20): 5.880905129621217, 
 ('llama_13b', 1, 9, 20): 5.69019994037764, 
 ('llama_13b', 2, 5, 20): 8.115552045155148, 
 ('llama_13b', 2, 6, 20): 8.800045787769099, 
 ('llama_13b', 2, 7, 20): 9.235313883894118, 
 ('llama_13b', 2, 8, 20): 9.179418510590507, 
 ('llama_13b', 2, 9, 20): 9.12792366690149, 
 ('llama_70b', 2, 9, 16): 16.772150315187663, 
 ('llama_70b', 2, 9, 20): 17.71703001014181, 
 ('llama_70b', 2, 9, 40): 17.361394237723985, 
 ('llama_7b', 1, 9, 2): 5.021288393914332, 
 ('llama_7b', 2, 9, 2): 8.077960270014238, 
 ('llama_7b', 4, 9, 2): 13.637959291700502, 
 ('llama_13b', 1, 9, 2): 6.070556158421993, 
 ('llama_13b', 2, 9, 2): 9.623457851793734, 
 ('llama_13b', 4, 9, 2): 17.593157510271627, 
 ('llama_70b', 4, 9, 2): 31.92388966289352, 
 ('llama_70b', 2, 9, 2): 14.395333220602716}




for flops, latency in zip(xs_dict[('llama_7b', 2, 8, 16)], ys_dict[('llama_7b', 2, 8, 16)]):
    if latency> 0.05:
        print(flops, latency)
'''
Correct:
3507619627008 0.06786405388265848
3507753844736 0.061133330687880516
3507888062464 0.061181322671473026
3507427999744 0.060861180536448956
3506264080384 0.060238552279770374
3505394810880 0.0596243916079402
3504203628544 0.05903193913400173
3503146926080 0.058523986488580704
3502401650688 0.058066338300704956
3501765427200 0.05779798608273268
3500893798400 0.05731580127030611
3499655692288 0.05655559618026018
3498066051072 0.05561350844800472
3497503752192 0.05629539769142866
3496022376448 0.05448293965309858
3495261896704 0.0540945902466774
3493013749760 0.05285591632127762
3492543463424 0.05258024111390114
3491927687168 0.05225819628685713
3490875441152 0.05159629508852959
3489758445568 0.05091289430856705
3473147691008 0.061687370762228966
Wrong:
30416395829248 0.06786405388265848
30416395829248 0.061133330687880516
30416395829248 0.061181322671473026
30416395829248 0.060861180536448956
30416395829248 0.060238552279770374
30416395829248 0.0596243916079402
30416395829248 0.05903193913400173
30416395829248 0.058523986488580704
30416395829248 0.058066338300704956
30416395829248 0.05779798608273268
30416395829248 0.05731580127030611
30416395829248 0.05655559618026018
30416395829248 0.05561350844800472
30416395829248 0.05629539769142866
30416395829248 0.05448293965309858
30416395829248 0.0540945902466774
30416395829248 0.05285591632127762
30416395829248 0.05258024111390114
30416395829248 0.05225819628685713
30416395829248 0.05159629508852959
30416395829248 0.05091289430856705
30402919530496 0.061687370762228966
'''


for k, v in dataset[('llama_7b', 2, 8, 16)].items():
    if v> 0.05:
        print(k, v)


{('llama_70b', 2, 9, 16): 16.772150315187663, 
 ('llama_7b', 1, 9, 2): 5.021288393914332, 
 ('llama_7b', 2, 9, 2): 8.077960270014238, 
 ('llama_7b', 4, 9, 2): 13.637959291700502, 
 ('llama_70b', 4, 9, 2): 31.92388966289352, 
 ('llama_70b', 2, 9, 2): 14.395333220602716}