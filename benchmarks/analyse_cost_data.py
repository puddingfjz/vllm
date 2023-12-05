# first collect the data from log file and store them into a csv file
import json


cost_data = dict()
with open("CostModel_log_peakaware_recomp2.log", 'r') as file:
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



# store the data into a csv file
sorted_keys = sorted([k for k in cost_data if k[-1] == 0], key=lambda k: k[2])

with open('cost_data.csv', 'a') as file:
	for k in sorted_keys:
		v = cost_data[k]
		json.dump(list(k) + [sum(v)/len(v)/k[1], min(v), max(v), sum(v)/len(v)], file)
		file.write('\n')



sorted_keys = sorted([k for k in cost_data if k[-1] != 0], key=lambda k: k[-2])

with open('cost_data.csv', 'a') as file:
	for k in sorted_keys:
		v = cost_data[k]
		json.dump(list(k) + [sum(v)/len(v)/k[-2], min(v), max(v), sum(v)/len(v)], file)
		file.write('\n')



