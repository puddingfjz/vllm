# first collect the data from log file and store them into a csv file
import json


cost_data = dict()
model_time_dict = dict()
sample_time_dict = dict()
with open("CostModel_log_peakaware_recomp4.log", 'r') as file:
	lines = file.readlines()
	req_num, token_num, padded_token_num, tot_seq_len, max_seq_len = None, None, None, None, None
	latency = None
	for line in lines:
		if '(req_num, token_num, padded_token_num, tot_seq_len, max_seq_len):' in line:
			# (req_num, token_num, padded_token_num, tot_seq_len, max_seq_len): (1, 1, 1, 640, 640)
			pos = line.find(':')
			tmp = line[pos:]
			pos1 = tmp.find('(')
			pos2 = tmp.find(')')
			tmp = f"[{tmp[pos1+1:pos2]}]"
			req_num, token_num, padded_token_num, tot_seq_len, max_seq_len = json.loads(tmp)
		elif 'finished num: ' in line:
			# finished num: 90, time point: 0.013905314728617668
			pos = line.find('time point:') + len('time point:')
			latency = float(line[pos:])
			key = (req_num, token_num, padded_token_num, tot_seq_len, max_seq_len)
			if key not in cost_data:
				cost_data[key] = list()
			cost_data[key].append(latency)
		elif 'breakdown' in line:
			# breakdown: 0.010584983974695206, 0.005603628989774734
			key = (req_num, token_num, padded_token_num, tot_seq_len, max_seq_len)
			if key not in cost_data:
				model_time_dict[key] = list()
				sample_time_dict[key] = list()			
			pos = len('breakdown: ')
			model_time, sample_time = json.loads(f"[{line[pos:]}]")
			model_time_dict[key].append(model_time)
			sample_time_dict[key].append(sample_time)



# store the data into a csv file
sorted_keys = sorted([k for k in cost_data if k[-1] == 0], key=lambda k: k[2])

with open('cost_data.csv', 'a') as file:
	for dict_name, data_dict in [('tot', cost_data), ('model', model_time_dict), ('sample', sample_time_dict)]:
		for k in sorted_keys:
			v = data_dict[k]
			json.dump([dict_name] + list(k) + [sum(v)/len(v)/k[1], min(v), max(v), sum(v)/len(v), max(v)/min(v)], file)
			file.write('\n')



sorted_keys = sorted([k for k in cost_data if k[-1] != 0], key=lambda k: k[-2])

with open('cost_data.csv', 'a') as file:
	for dict_name, data_dict in [('tot', cost_data), ('model', model_time_dict), ('sample', sample_time_dict)]:
		for k in sorted_keys:
			v = data_dict[k]
			json.dump([dict_name] + list(k) + [sum(v)/len(v)/k[-2], min(v), max(v), sum(v)/len(v), max(v)/min(v)], file)
			file.write('\n')



