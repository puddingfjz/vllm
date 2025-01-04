"""
This experiment results from log files.
"""
import json

def get_log_file_names():
    file_name_list = list()
    for gpu_name in ['A100-80G', 'A100-40G']:
        byte_per_gpu=85899345920
        if gpu_name == 'A100-40G':
            byte_per_gpu=42949672960
        # 
        for tot_gpu_num in [8, 4]:
            gpu_ids='0,1,2,3,4,5,6,7'
            if tot_gpu_num == 4:
                gpu_ids='0,1,2,3'
            for max_group_seq_num in [1, 20]:
                top_k=20
                similar_threshold=0.2
                fully_connected_gpu_unit=2
                machine_name='zxcpu'
                # 
                specify_outlen=''
                # chain-summary
                for summarize_model in ['lmsys/vicuna-13b-v1.5', 'mistralai/Mixtral-8x7B-Instruct-v0.1']:
                    summarize_model_setting='vicuna-13b-v1.5'
                    if summarize_model == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
                        summarize_model_setting='Mixtral-8x7B-Instruct-v0.1'
                    # 
                    # for reqnum in 100 300 500 700
                    for reqnum in [100, 300]:
                        # for evaluator_num in 1 2 3 4 5 6 7
                        for evaluator_num in [1, 2, 3, 4, 5, 6]:
                            for gen_execplans_baseline in ['ours', 'naive']:

                                file_name = f'test_end2end_schedule/test_1231_{tot_gpu_num}gpu-booookscore_{gen_execplans_baseline}_{gpu_name}_{machine_name}_{summarize_model_setting}_{evaluator_num}eval_maxlen_900_{reqnum}_{max_group_seq_num}_1.log'
                                file_name_info = ((gen_execplans_baseline, max_group_seq_num), f'test_1231_{tot_gpu_num}gpu-booookscore_{gpu_name}_{machine_name}_{summarize_model_setting}_{evaluator_num}eval_maxlen_900_{reqnum}_1.log')
                                file_name_list.append((file_name, file_name_info))
                # 
                # ensemble
                for reqnum in [100, 300, 500, 700]:
                    for max_token_num in [512, 256]:
                        for gen_execplans_baseline in ['ours', 'naive']:
                            file_name = f'test_end2end_schedule/test_1231_{tot_gpu_num}gpu-llm-blender_{gen_execplans_baseline}_{gpu_name}_{machine_name}_maxlen_{max_token_num}_{reqnum}_{max_group_seq_num}_1.log'
                            file_name_info = ((gen_execplans_baseline, max_group_seq_num), f'test_1231_{tot_gpu_num}gpu-llm-blender_{gpu_name}_{machine_name}_maxlen_{max_token_num}_{reqnum}_1.log')
                            file_name_list.append((file_name, file_name_info))
                # 
                # router
                for use_specify_outlen in [specify_outlen, '--specify_outlen']:
                    outlen_file_name_setting='maxlen_4096'
                    if use_specify_outlen == '--specify_outlen':
                        outlen_file_name_setting='setOutlen'
                    # 
                    for gen_execplans_baseline in ['ours', 'naive']:
                        file_name = f'test_end2end_schedule/test_1231_{tot_gpu_num}gpu-router_{gen_execplans_baseline}_{gpu_name}_{machine_name}_not_multiple_choice_question_{outlen_file_name_setting}_10000_{max_group_seq_num}_1.log'
                        file_name_info = ((gen_execplans_baseline, max_group_seq_num), f'test_1231_{tot_gpu_num}gpu-router_{gpu_name}_{machine_name}_not_multiple_choice_question_{outlen_file_name_setting}_10000_1.log')
                        file_name_list.append((file_name, file_name_info))
    return file_name_list



def extract_results_from_log(file_name):
    """
        Extract results from log.
        Output:
            predicted_running_time, real_running_time, search_time, prepare_time_before_search.
    """
    predicted_running_time, real_running_time, search_time, prepare_time_before_search = None, None, None, None 
    try:
        with open(file_name, 'r', errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if 'Best group seq: ' in line:
                    predicted_running_time = float(line.split()[-2])
                elif 'total running time: ' in line:
                    real_running_time = float(line[len('total running time: '):].split()[0][:-1])
                elif 'Total search time: ' in line:
                    search_time = float(line.split()[-1])
                elif 'Total time for preparation before search: ' in line:
                    prepare_time_before_search = float(line.split()[-1])
        print(predicted_running_time, real_running_time, search_time, prepare_time_before_search)
        if None in [predicted_running_time, real_running_time, search_time, prepare_time_before_search]:
            predicted_running_time, real_running_time, search_time, prepare_time_before_search = None, None, None, None 
    except Exception as e:
        print(e)
    return predicted_running_time, real_running_time, search_time, prepare_time_before_search



if __name__ == "__main__":
    res_dict = {(method, max_group_seq_num):dict() for method in ['ours', 'naive'] for max_group_seq_num in [1, 20]}
    file_name_list = get_log_file_names()
    for file_name, file_name_info in file_name_list:
        print(file_name)
        res = extract_results_from_log(file_name)
        if res[0] != None:
            # a valid res
            (method, max_group_seq_num), setting = file_name_info
            res_dict[(method, max_group_seq_num)][setting] = res
    print(f"res_dict: {res_dict}")
    # 
    # comp ratios
    with open(f"Exp_Analysis_extract_results_from_logs_outputs.json", 'a') as f:
        key_naive = ('naive', 1)
        for setting in res_dict[key_naive]:
            res = list()
            for max_group_seq_num in [1, 20]:
                key_ours = ('ours', max_group_seq_num)
                if setting in res_dict[key_ours]:
                    predicted_running_time_ours, real_running_time_ours, search_time_ours, prepare_time_before_search_ours = res_dict[key_ours][setting]

                    predicted_running_time_naive, real_running_time_naive, search_time_naive, prepare_time_before_search_naive = res_dict[key_naive][setting]

                    ratio_running_time = real_running_time_naive/real_running_time_ours

                    ratio_tot_time = sum([real_running_time_naive, search_time_naive, prepare_time_before_search_naive])/sum([real_running_time_ours, search_time_ours, prepare_time_before_search_ours])
                    res.append((ratio_running_time, ratio_tot_time, res_dict[key_ours][setting]))
                else:
                    res.append((None, None, None))
            res.append(res_dict[key_naive][setting])

            json.dump({setting: res}, f)
            f.write('\n')


            



'''
from Exp_Analysis_extract_results_from_logs import *


res_dict = {'ours':dict(), 'naive':dict()}
file_name_list = get_log_file_names()
for file_name, file_name_info in file_name_list:
    print(file_name)
    res = extract_results_from_log(file_name)
    if res[0] != None:
        # a valid res
        method, setting = file_name_info
        res_dict[method][setting] = res

        
print(f"res_dict: {res_dict}")



'''


