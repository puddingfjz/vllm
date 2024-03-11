
import json

our_fname = 'ours_0220_7b_15-2.log'
vllm_fname = 'vllm_0220_7b_15-2.log'


our_fname = 'tmp_ours.log'
vllm_fname = 'tmp.log'



ours = list()
vllm = list()

for res_list, fname in zip([ours, vllm], [our_fname, vllm_fname]):
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ('x 2:' in line) and ('model.layer' not in line):
                # res = line[len('key block'):]
                # res_list.append(res)
                pos = line.find(':')
                res = line[pos+1:]
                res_list.append((line[:pos], res))




for count, ((i_name, i), (j_name, j)) in enumerate(zip(ours[:], vllm[:])):
    # if i_name =='key block 0':
    #     continue
    # if count == 96:
    #     continue
    if i!=j:
        print(count, i_name, j_name)
        a = json.loads(i)
        b = json.loads(j)
        for count_, (ii, jj) in enumerate(zip(a, b)):
            if ii != jj:
                print(count_, ii, jj)
        break
    





with open('tmp.json', 'a') as f:
    f.write(f'ours:')
    for i in ours:
        f.write(f"{i}\n")
    f.write(f'vllm:')
    for i in vllm:
        f.write(f"{i}\n")




weight_add = 139619929595904
last_layer_v_cache = 139619992076288
new_last_v_cache = 139617159741440

weight_add >= new_last_v_cache + math.prod([16202, 16, 16, 16, 8])*2

k_cache.shape: torch.Size([16888, 16, 16, 16, 8])

new_block_num: 16202 ==> shape torch.Size([16202, 16, 16, 16, 8])
k_cache.shape: torch.Size([16202, 16, 16, 16, 8])











# test function correctness--------------------------------------------------------------------------
def get_dependent_blk_moving_chains(fromblknum_to_blknum):
    '''
        NOTE: this function is called after reorganize_gpu_blocks.
        Get the blk moving chains where there is dependency.
        E.g., (1) move blk 3 to blk 1, (2) blk 5 to blk 3. then we cannot do (1)&(2) together, and (1) and (2) form a chain.
        Input:
            fromblknum_to_blknum: Dict[int, int]. 
        Output:
            (the chains connected together, the length of each chain): Tuple[List[Tuple[int, int]], List[int]].
    '''
    layer_num = 4
    curr_tot_gpu_blk_num = 10
    ori_tot_gpu_blk_num = 15    
    mapping_dict = fromblknum_to_blknum.copy()
    # get block mapping in the whole gpu cache
    for layer_i in range(1, 2*layer_num):
        # deal with key cache and value cache in every layer (except key cache in layer 0)
        mapping_dict.update([(k + ori_tot_gpu_blk_num * layer_i, v + curr_tot_gpu_blk_num * layer_i) \
                                for k, v in fromblknum_to_blknum.items()])
    print(f"mapping_dict: {mapping_dict}")
    # get dependent block mapping chains
    from_blk_gids = sorted(mapping_dict.keys())
    visited = {gid: False for gid in from_blk_gids}
    chains = list()
    chain_lens = [0]
    for i in range(len(from_blk_gids)-1, -1, -1):
        print(i)
        src_gid = from_blk_gids[i]
        if visited[src_gid]:
            continue
        if mapping_dict[src_gid] == src_gid:
            visited[src_gid] = True
            continue
        chains.append(src_gid)
        while(src_gid in mapping_dict):
            visited[src_gid] = True
            src_gid = mapping_dict[src_gid]
            chains.append(src_gid)
        chain_lens.append(len(chains))
    return chains, chain_lens




fromblknum_to_blknum = {1:1, 11:5}
get_dependent_blk_moving_chains(fromblknum_to_blknum)

# END test function correctness--------------------------------------------------------------------------










import json

our_fname = 'ours_0220_7b_2.log'
vllm_fname = 'vllm_0216_7b_2.log'


ours = list()
vllm = list()

for res_list, fname in zip([ours, vllm], [our_fname, vllm_fname]):
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ' ' in line:
                # res = line[len('key block'):]
                # res_list.append(res)
                pos = line.find(':')
                res = line[pos+1:]
                res_list.append((line[:pos], res))






for count, ((i_name, i), (j_name, j)) in enumerate(zip(ours, vllm)):
    if i!=j:
        print(count, i_name, j_name)
        break
        a = json.loads(i)
        b = json.loads(j)
        for count_, (ii, jj) in enumerate(zip(a, b)):
            if ii != jj:
                print(count_, ii, jj)
        break
    

















# debug results--------------------------------------------------------------------
('key & value cache address', ' (140001941651456, 140001947549696)\n')
('key & value cache address', ' (140001953447936, 140001959346176)\n')
('key & value cache address', ' (140001965244416, 140001971142656)\n')
('key & value cache address', ' (140001977040896, 140001982939136)\n')
('key & value cache address', ' (140001988837376, 140001994735616)\n')
('key & value cache address', ' (140002000633856, 140002006532096)\n')
('key & value cache address', ' (140002012430336, 140002018328576)\n')
('key & value cache address', ' (140002024226816, 140002030125056)\n')
('key & value cache address', ' (140002036023296, 140002041921536)\n')
('key & value cache address', ' (140002047819776, 140002053718016)\n')
('key & value cache address', ' (140002059616256, 140002065514496)\n')
('key & value cache address', ' (140002071412736, 140002077310976)\n')
('key & value cache address', ' (140002083209216, 140002089107456)\n')
('key & value cache address', ' (140002095005696, 140002100903936)\n')
('key & value cache address', ' (140002106802176, 140002112700416)\n')
('key & value cache address', ' (140002118598656, 140002124496896)\n')
('key & value cache address', ' (140002130395136, 140002136293376)\n')
('key & value cache address', ' (140002142191616, 140002148089856)\n')
('key & value cache address', ' (140002153988096, 140002159886336)\n')
('key & value cache address', ' (140002165784576, 140002171682816)\n')
('key & value cache address', ' (140002177581056, 140002183479296)\n')
('key & value cache address', ' (140002189377536, 140002195275776)\n')
('key & value cache address', ' (140002201174016, 140002207072256)\n')
('key & value cache address', ' (140002212970496, 140002218868736)\n')
('key & value cache address', ' (140002224766976, 140002230665216)\n')
('key & value cache address', ' (140002236563456, 140002242461696)\n')
('key & value cache address', ' (140002248359936, 140002254258176)\n')
('key & value cache address', ' (140002260156416, 140002266054656)\n')
('key & value cache address', ' (140002271952896, 140002277851136)\n')
('key & value cache address', ' (140002283749376, 140002289647616)\n')
('key & value cache address', ' (140002295545856, 140002301444096)\n')
('key & value cache address', ' (140002307342336, 140002313240576)\n')

('self.model.extra_weight_cache[18]', ' 140004582998016\n')
('self.model.extra_weight_cache[19]', ' 140004380606464\n')
('self.model.extra_weight_cache[20]', ' 140004178214912\n')
('self.model.extra_weight_cache[21]', ' 140003975823360\n')
('self.model.extra_weight_cache[22]', ' 140003773431808\n')
('self.model.extra_weight_cache[23]', ' 140003571040256\n')
('self.model.extra_weight_cache[24]', ' 140003368648704\n')
('self.model.extra_weight_cache[25]', ' 140003166257152\n')
('self.model.extra_weight_cache[26]', ' 140002963865600\n')
('self.model.extra_weight_cache[27]', ' 140002761474048\n')
('self.model.extra_weight_cache[28]', ' 140002559082496\n')
('self.model.extra_weight_cache[29]', ' 140002356690944\n')
('self.model.extra_weight_cache[30]', ' 140002154299392\n')
('self.model.extra_weight_cache[31]', ' 140001951907840\n')
# end of debug---------------------------------------------------------------------


5856

[[22274], [1063], [748], [29871], [15710], [2643]]


step i 128, 156 th  is what we need to check. ours
step i 155, 136 th  is what we need to check. vllm



>>> ya[:4]
[-0.0016613006591796875, 0.0169830322265625, -0.0167694091796875, 0.0130767822265625]
>>> yb[:4]
[-0.0016632080078125, 0.0169830322265625, -0.0167694091796875, 0.01308441162109375]
>>> wa*xa
array([ 2.33867468e-06,  1.96230940e-06,  2.54907645e-04, ...,
        5.81310815e-07,  2.81245593e-05, -1.35335722e-06])
>>> sum(wa*xa)
0.0019747960967073652
>>> sum(wb*xb)
0.0019747960967073652