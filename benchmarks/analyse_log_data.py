# breakdown analysis of the log file


file_names = ["log_plan3.log", "log_vllm_swap3.log"]

for fname in file_names:
	prompt_costs = list()
	decode_costs = list()
	cost_list = None
	with open(fname, 'r') as file:
		lines = file.readlines()
		latency = None
		for line in lines:
			if '(req_num, token_num, padded_token_num, tot_seq_len, max_seq_len):' in line:
				# (req_num, token_num, padded_token_num, tot_seq_len, max_seq_len): (1, 1, 1, 640, 640)
				if '0, 0)' in line:
					# prompt stage
					cost_list = prompt_costs
				else:
					cost_list = decode_costs
			elif 'finished num: ' in line:
				# finished num: 90, time point: 0.013905314728617668
				pos = line.find('time point:') + len('time point:')
				latency = float(line[pos:])
				cost_list.append(latency)
	print(f"{fname}: promp stage num: {len(prompt_costs)}, prompt tot cost: {sum(prompt_costs)}, decode_stage_num: {len(decode_costs)}, decode tot cost: {sum(decode_costs)}")





