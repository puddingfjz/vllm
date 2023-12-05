from gurobipy import *
import numpy as np

# requests = benchmark_throughput.my_sample_requests(
#     dataset_path: str,
#     num_requests: int,
#     tokenizer: PreTrainedTokenizerBase,
# )

# tokenizer = get_tokenizer('huggyllama/llama-7b',
#                           trust_remote_code=args.trust_remote_code)



def get_requests(requests=list()):
    # requests = [(5, 5), (8, 3)]
    if requests == list():
        requests = [(32, 4)] * 10
    indptr = np.cumsum([0] + [info[1] for info in requests]).tolist()
    T = indptr[-1] # the total number of generation tokens, and the maximum #iter
    KV_nums = np.concatenate([np.arange(info[0], info[0] + info[1]) for info in requests]).tolist()
    ins = np.concatenate([np.full(info[1], info[0]) for info in requests]).tolist()
    recomp_KV_nums = np.asarray(\
        [((info[0] + j) * (info[0] + j - 1)) // 2 for info in requests for j in range(info[0])]\
        ).tolist()
    # 
    return requests, (T, indptr, KV_nums, ins, recomp_KV_nums)




def set_vars(m, requests, T, iter_num, indptr):
    # the max #iter, also the total generation token number
    # vars: (iter_num, tot_gen_token)
    x = m.addVars(iter_num,T,vtype=GRB.BINARY)
    w1 = m.addVars(iter_num,T,vtype=GRB.BINARY)
    w2 = m.addVars(iter_num,T,vtype=GRB.BINARY)
    w3 = m.addVars(iter_num,T,vtype=GRB.BINARY)
    z1 = m.addVars(iter_num,T,vtype=GRB.BINARY)
    z2 = m.addVars(iter_num,T,vtype=GRB.BINARY)
    return x, w1, w2, w3, z1, z2
    # 





def set_obj(m, requests, iter_num, indptr, KV_nums, ins, recomp_KV_nums, 
    x, w1, w2, w3, z1, z2):
    # obj = sum of (prmpt cost + generation cost + recomp cost for each iteration)
    obj = gurobipy.quicksum(\
        # prompt cost
          gurobipy.quicksum(ins[i] * x[iter_i, i] for i in indptr[:-1]) \
        # generation cost
        + gurobipy.quicksum(KV_nums[i] * x[iter_i, i] \
                        for req_i in range(len(requests)) \
                        for i in range(indptr[req_i]+1, indptr[req_i+1])) \
        # recomputation cost
        + gurobipy.quicksum(recomp_KV_nums[i] * w3[iter_i, i] \
                        for req_i in range(len(requests)) \
                        for i in range(indptr[req_i]+1, indptr[req_i+1])) \
        for iter_i in range(iter_num))
    m.setObjective(obj, sense=gurobipy.GRB.MINIMIZE)





def set_constraints(m, requests, T, iter_num, KV_nums, indptr, tot_blk_num,
    x, w1, w2, w3, z1, z2):
    # schedule validity
    print("validity--------")
    m.addConstrs( gurobipy.quicksum(x[iter_i,i] for iter_i in range(iter_num)) == 1 for i in range(T) )
    # precedence constraint
    print("precedence------")
    m.addConstrs( gurobipy.quicksum( x[iter_i, i] for iter_i in range(t)) >= \
        gurobipy.quicksum( x[iter_i, i+1] for iter_i in range(t+1)) \
        for t in range(iter_num-1) \
        for req_i in range(len(requests)) for i in range(indptr[req_i], indptr[req_i+1]-1) )
    # memory constraint
    print("memory----------")
    m.addConstrs( gurobipy.quicksum(KV_nums[i]*x[iter_i, i] for i in range(T)) <= tot_blk_num \
        for iter_i in range(iter_num))
    # constraint for extra variables
    print("extra1----------")
    m.addConstrs( (x[iter_i, i] - x[iter_i-1, i-1] + w1[iter_i, i] - w3[iter_i, i]) == 0 \
        for iter_i in range(1, iter_num) \
        for req_i in range(len(requests)) for i in range(indptr[req_i] + 1, indptr[req_i+1]) )
    print("extra2----------")
    m.addConstrs( w1[iter_i, i] <= z1[iter_i, i] \
        for iter_i in range(1, iter_num) \
        for req_i in range(len(requests)) for i in range(indptr[req_i] + 1, indptr[req_i+1]) )
    print("extra3----------")
    m.addConstrs( w2[iter_i, i] <= (z1[iter_i, i] + z2[iter_i, i]) \
        for iter_i in range(1, iter_num) \
        for req_i in range(len(requests)) for i in range(indptr[req_i] + 1, indptr[req_i+1]) )
    print("extra4----------")
    m.addConstrs( w3[iter_i, i] <= z2[iter_i, i] \
        for iter_i in range(1, iter_num) \
        for req_i in range(len(requests)) for i in range(indptr[req_i] + 1, indptr[req_i+1]) )
    print("extra5----------")
    m.addConstrs( (w1[iter_i, i]+w2[iter_i, i]+w3[iter_i, i]) == 1 \
        for iter_i in range(1, iter_num) \
        for req_i in range(len(requests)) for i in range(indptr[req_i] + 1, indptr[req_i+1]) )
    print("extra6----------")
    m.addConstrs( (z1[iter_i, i]+z2[iter_i, i]) == 1 \
        for iter_i in range(1, iter_num) \
        for req_i in range(len(requests)) for i in range(indptr[req_i] + 1, indptr[req_i+1]) )
    print("extra7----------")
    m.addConstrs( (z1[iter_i, i]+z2[iter_i, i]+w1[iter_i, i]+w2[iter_i, i]+w3[iter_i, i]) == 0 \
        for iter_i in range(1, iter_num) for i in indptr[:-1] ) # there will not be recomputation for step0 
    print("extra8----------")
    m.addConstrs( (z1[0, i]+z2[0, i]+w1[0, i]+w2[0, i]+w3[0, i]) == 0 \
        for i in range(T) ) # there will not be recomputation in iteration 0
    


def print_solution(results):
    max_iter = max([_[0] for tmp in results for _ in tmp]) + 1
    for tmp in results:
        line = [' ']*max_iter
        for term in tmp:
            line[term[0]] = '*'
        print(''.join(line))



def solve_the_model(tot_blk_num, iter_num, requests, T, indptr, KV_nums, ins, recomp_KV_nums, 
    res_dict):
    # Create a new Gurobi Model
    m = Model("lp")
    m.setParam(GRB.Param.Threads, 128)
    print("adding vars-----------")
    x, w1, w2, w3, z1, z2 = set_vars(m, requests, T, iter_num, indptr)
    print("adding obj------------")
    set_obj(m, requests, iter_num, indptr, KV_nums, ins, recomp_KV_nums, 
        x, w1, w2, w3, z1, z2)
    print("adding constraints----")
    set_constraints(m, requests, T, iter_num, KV_nums, indptr, tot_blk_num, 
        x, w1, w2, w3, z1, z2)
    print("doing optimization----")
    m.optimize()        
    # Print the feasible solution if optimal.
    if m.status == GRB.Status.OPTIMAL:
        print('Obj Function:', m.objVal)
        # print("Optimal Solution:")
        # for v in m.getVars():
        #     print(v.varName, v.x)
    # Another way to print the variable
        # print("Optimal Solution:")
        # print(x.varName, x.x)
        # print(y.varName, y.x)        
    else:
        print(m.status)
    # 
    results = [[] for i in requests]
    for iter_i in range(iter_num):
        for req_i in range(len(requests)):
            tmp = [i-indptr[req_i]+requests[req_i][0] \
                for i in range(indptr[req_i], indptr[req_i+1]) if x[iter_i, i].x == 1]
            if tmp != []:
                results[req_i].append((iter_i, tmp[0]))
    # 
    for tmp in sorted(results):
        print(tmp)
    print_solution(results)
    res_dict[tot_blk_num, iter_num] = (m.objVal, results)
    return m





res_dict = dict()
tot_blk_num = 125 # 2000
requests, (T, indptr, KV_nums, ins, recomp_KV_nums) = get_requests()
iter_num = T
m = solve_the_model(tot_blk_num, iter_num, requests, T, indptr, KV_nums, ins, recomp_KV_nums,res_dict)


requests = [(32, i) for i in range(1, 11)]
requests = [(32+i-5, 5) for i in range(1, 11)]
requests = [(32+i-j, j) for i in range(1, 8) for j in [3,5,11]]
requests, (T, indptr, KV_nums, ins, recomp_KV_nums) = get_requests(requests)
tot_blk_num = 125 
iter_num = 50
m = solve_the_model(tot_blk_num, iter_num, requests, T, indptr, KV_nums, ins, recomp_KV_nums,res_dict)



# ============================================================================
# below is the script to run

tot_blk_num = 13 # 2000
# Create a new Gurobi Model
m = Model("lp")  

requests, (T, indptr, KV_nums, ins, recomp_KV_nums) = get_requests()
iter_num = T
x, w1, w2, w3, z1, z2 = set_vars(m, requests, T, iter_num, indptr)
set_obj(m, requests, iter_num, indptr, KV_nums, ins, recomp_KV_nums, 
    x, w1, w2, w3, z1, z2)
set_constraints(m, requests, T, iter_num, KV_nums, indptr, tot_blk_num, 
    x, w1, w2, w3, z1, z2)


# Create two new variables
# x = m.addVar(vtype=GRB.BINARY, name ="x")
# y = m.addVar(vtype=GRB.BINARY, name ="y")
# x = m.addVar(lb=0, name ="x")
# y = m.addVar(lb=0, name ="y")

    
# Solve the model
m.optimize()
    
# Print the feasible solution if optimal.
if m.status == GRB.Status.OPTIMAL:
    print('Obj Function:', m.objVal)
    print("Optimal Solution:")
    for v in m.getVars():
        print(v.varName, v.x)
# Another way to print the variable
    # print("Optimal Solution:")
    # print(x.varName, x.x)
    # print(y.varName, y.x)        
else:
    print(m.status)



results = [[] for i in requests]
for iter_i in range(iter_num):
    for req_i in range(len(requests)):
        tmp = [i-indptr[req_i]+requests[req_i][0] \
            for i in range(indptr[req_i], indptr[req_i+1]) if x[iter_i, i].x == 1]
        if tmp != []:
            results[req_i].append((iter_i, tmp[0]))



for tmp in results:
    print(tmp)


