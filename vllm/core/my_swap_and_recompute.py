# implement the method of page-swap and page-recompute
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import math


# 目前这个cost model里面所有的常数都是随便设的。
# 其效果为（1）我们不会考虑任何recomputation的可能；（2）我们每个iteration可以swap 10个block
class myCostModel(object):
	"""docstring for myCostModel"""
	def __init__(self, comp_unit_cost: float, mem_bandwidth: float) -> None:
		super(myCostModel, self).__init__()
		self.comp_unit_cost = comp_unit_cost
		self.mem_bandwidth = mem_bandwidth
	
	def get_comp_cost(self, seq_group_lens):
		''' 
		Return the computation cost of this iteration.
		INPUT:	batch size and seq group lens are all information of the current iteration
		'''
		return float('inf')

		batch_size = len(seq_group_lens)
		# temporariry set this number
		return 1


	def get_swap_blk_num(self, seq_group_lens, blk_mem_size):
		# return the number of blocks that can be swapped out whose latency will be overlapped by the computation cost
		return 10

		comp_cost = self.get_comp_cost(seq_group_lens)
		return int((comp_cost*self.mem_bandwidth) / blk_mem_size)


	def get_swap_cost(self, blk_num, blk_mem_size):
		return 1

		return blk_num*blk_mem_size/(self.mem_bandwidth)




class myMemScheduler(object):
	"""docstring for myMemScheduler"""
	def __init__(self):
		super(myMemScheduler, self).__init__()
		# stores the blocks of each request that have been swapped out or released for recomputation.
		# self.swapped_out = dict()
		# self.release_for_recompute = dict()
		self.release_infor:Dict[str, List[Tuple[int, str]]] = dict() # {req_id: [(blk_i, release_way), (), ...]} where release_way is 'swap' or 'recompute'
		# stores the number of blocks on card for a seq group with a specific req_id \
		# used for partially swapped out requests only
		self.on_card_info:Dict[str, int] = dict() # {req_id: block number on card}



	def no_interrupted_requests(self):
		# return (len(self.release_infor) == 0) or (sum([len(v) for v in self.release_infor.values()])==0)
		return len(self.on_card_info) == 0


	def request_is_interrupted(self, req_i):
		return req_i in self.on_card_info


	def request_has_released_blks(self, req_i):
		return (req_i in self.release_infor) and (len(self.release_infor[req_i])>0)


	# <jingzhi> @TODO: consider merge this function with updating on_card_info
	def update_release_info(self, to_swap: Dict[str, int], to_recompute: Dict[str, int]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
		swap_list = list()
		recompute_list = list()
		for k, v in to_swap.items():
			if k not in self.release_infor:
				self.release_infor[k] = [(blk_i, 'swap') for blk_i in range(v)]
				swap_list = swap_list + [(k, blk_i) for blk_i in range(v)]
			else:
				released = len(self.release_infor[k])
				self.release_infor[k] = self.release_infor[k] + [(blk_i, 'swap') for blk_i in range(released, released + v)]
				swap_list = swap_list + [(k, blk_i) for blk_i in range(released, released + v)]
		for k, v in to_recompute.items():
			if k not in self.release_infor:
				self.release_infor[k] = [(blk_i, 'recompute') for blk_i in range(v)]
				recompute_list = recompute_list + [(k, blk_i) for blk_i in range(v)]
			else:
				released = len(self.release_infor[k])
				self.release_infor[k] = self.release_infor[k] + [(blk_i, 'recompute') for blk_i in range(released, released + v)]
				recompute_list = recompute_list + [(k, blk_i) for blk_i in range(released, released + v)]
		return swap_list, recompute_list


	def decrease_on_card_blk_num(self, req_i, decrease, tot=None):
		if decrease == 0:
			return

		if req_i not in self.on_card_info:
			assert tot!=None, 'No total block number infor provided'
			self.on_card_info[req_i] = tot

		# <jingzhi> DEBUG
		# print(req_i, self.on_card_info[req_i], decrease, tot)

		self.on_card_info[req_i] = self.on_card_info[req_i] - decrease


	def increase_on_card_blk_num(self, req_i, increase):
		assert req_i in self.on_card_info, "Error: increase on card blk num for a non-interrupted request"
		self.on_card_info[req_i] = self.on_card_info[req_i] + increase




	def delete_useless_on_card_info(self, running_req_ids):
		for req_i in running_req_ids:
			del self.on_card_info[req_i]
			del self.release_infor[req_i]
		# # Ensure on_card_info only contains information for partially released requests
		# req_is = list(self.on_card_info.keys())
		# for req_i in req_is:
		# 	# when the request is fully released, delete its on card block number info
		# 	if self.on_card_info[req_i] == 0:
		# 		del self.on_card_info[req_i]
		# 	else:
		# 		# when the request is ready to run, delete its on card block number info
		# 		if len(self.release_infor[req_i]) == 0:
		# 			del self.on_card_info[req_i]



	def delete_finished_released_requests(self, request_ids):
		for req_i in request_ids:
			del self.on_card_info[req_i]
			del self.release_infor[req_i]


	# NOTE: on_card_info只存储partial released 的, 完全release，ready to run, 的 request的blk 信息。 
	# 不包含 running的request的信息


	# NOTE: 我们不应该把把ready to run 的request和running 的request 在找swap out的block的时候混在一起，
	# 因为ready to run 的request 不参与计算，所以他们的block 不会需要再某个iteration之后才会变得完整。
	# 所以当一个request ready to run 的时候，我们不应该把他们从on card info当中删掉，
	# 应该把他们和其他partially released 的request放在一起考虑。




cost_model = myCostModel(1, 1)
mem_scheduler = myMemScheduler()



# def need_start_swap(capacity: int, blk_mem_size: int, seq_group_lens: List[int], seq_group_tot_lens: List[int]) -> Tuple[List[int], List[int]]:
# 	'''
# 	INPUT:	
# 		capacity: the total number of GPU blocks available
# 		seq_group_lens: the lenghts of the sequence group lengths in the current batch (prompt+output)
# 		seq_group_tot_lens: the expected total lenghts of the sequence group lengths in the current batch (prompt+output)
# 	OUTPUT:
# 		to_swap: the indices of the seq_groups to be swapped out.
# 		to_recompute: the indices of the seq_groups to be recomputed.
# 	'''
# 	max_left_len = max(np.asarray(seq_group_tot_lens) - np.asarray(seq_group_lens))
# 	batch_size = len(seq_group_lens)
# 	current_tot_len = np.sum(seq_group_lens)

# 	left_lens = np.asarray(seq_group_tot_lens) - np.asarray(seq_group_lens)
# 	assert 0 not in left_lens, "ERROR: there are sequence groups which have been finished."
	
# 	# compute the total len increase in each iteration
# 	left_lens, counts = np.unique(left_lens, return_counts=True)
# 	tot_len_each_iter = np.zeros(max_left_len)
# 	for i, j in zip(left_lens, counts):
# 		tot_len_each_iter[i-1] = j
# 	tot_len_each_iter = np.cumsum(tot_len_each_iter)
# 	tot_len_each_iter = (batch_size - tot_len_each_iter) * np.arange(1, max_left_len+1)

# 	capacity_is_enough = tot_len_each_iter <= capacity - current_tot_len
# 	fail_iter = numpy.argsort(capacity_is_enough, kind='stable')[0]
# 	if capacity_is_enough[fail_iter]:
# 		# the capacity will always be enough
# 		return (list(), list())

# 	# we need to swap out or recompute
# 	# which to swap and which to recompute，这个地方怎么选择呢？
# 	# 首先要计算需要swap out多少block，以及提前多少轮swap out。但是这个需要cost model。如果后续还需要挪出更多的内存，应该怎么办？我们的quota就不够了。quota不够的情况下应该怎么办？
# 	# quota不够，有三种方法：1.不够的部分直接delete+recompute；2.不够的部分暂停一部分参与计算的batch内的request；3.不够的部分依然swap，但是整个机器就idle了。而且以block size为单位
# 	# 做决定刚好和page Attention相符合。所以quota就是一个block size的iterations下我们总共能swap out的内存数。而且还有一个地方，就是如果当前每个iteration的计算时间很短的话，
# 	# 不足以overlap swap的时间，那我们swap就也有额外的开销。所以这个应该也被考虑到每个iteration能swap out多少内存中去。

# 	# how many extra blocks need to be taken from on card requests

# 	# 这个地方写得有问题，因为是在将来的第fail_iter我们需要释放一些request来获得额外的内存，而不是在目前的这个iter我们就已经爆内存而需要再前几轮就已经释放request！！！！
# 	# 回家再重新写，怎么重新写？我们做决策也是以block size为单位来做的吗，感觉应该这样，但是可能会有计算上的浪费。不管了，先这样来，感觉会更regular一点。
# 	# 中间如果有request结束怎么处理？空着不管？就空着不管吧，毕竟现在是以block size为单位来做决策。
# 	# 但是如果当前没有需要等待的cuda event，也是可以立即中断当前的block然后重新开始的。
# 	# 或者如果当前结束的request空出来的空间已经足够cuda event需要挪出来的空间的时候，也可以立即中断当前block然后重新开始。
# 	# 但是这两种提前中断的先不管。

# 	# we have fail_iter iterations to do swap, the swap should be finished before the fail_iter-th iteration, now is before the 0-th iteration.


# 	# should consider that some requests may already be finished
# 	sorted_lens = sorted(((seq_group_lens + fail_iter) < seq_group_tot_lens) * (seq_group_lens + fail_iter))
# 	finished_num = sum(sorted_lens == 0)
# 	sorted_lens[ finished_num: ] = 1
# 	tot_blk_available = np.cumsum(sorted_lens)
# 	tmp_idx = np.argsort((current_tot_len + tot_len_each_iter[fail_iter] - capacity) > tot_blk_available, kind='stable')[0]

# 	req_to_release_indices = np.argsort(((seq_group_lens + fail_iter) < seq_group_tot_lens) * (seq_group_lens + fail_iter))[ finished_num:tmp_idx+1 ]


# 	# 啊，不对啊，这里的写法也不是page swap 或者page recompute，依然还是完整req的swap和recompute

# 	# estimate the computation cost of the fail_iter iterations based on the batch information in the current iteration
# 	blk_num_can_swap = int(cost_model.get_swap_blk_num(seq_group_lens, blk_mem_size)) * fail_iter
# 	# for the remaining blocks, (1) recompute? (2) swap and idle? (3) stop more requests but keep them on card
# 	# 但是其实我不知道（3）这里的cost应该怎么计算，暂时不考虑（3）的这种选择了。

# 	# estimate the cost of the remaining swap and the remaining recompute
# 	remain_blk_num = sum(seq_group_lens[req_to_release_indices]) - blk_num_can_swap
# 	remain_req_indices = seq_group_lens[req_to_release_indices]+fail_iter


# 	swap_cost = cost_model.get_swap_cost(blk_num_can_swap, blk_mem_size)
# 	# 暂时不考虑把每个iteration做swap的时候可能会有一部分block没有overlap的情况
# 	tpm_indices = 
# 	recompute_cost = cost_model.get_comp_cost(seq_group_lens[req_to_release_indices]+fail_iter)

# 	if swap_cost < recompute_cost:
# 		return ()






'''
还是得重新写整个逻辑，直接简单一点，分三种方案讨论：（1）纯swap out，（2）纯recompute，（3）swap out和recompute混合。但是其实还是和前面的逻辑是一样的。
写法上面换成以block size为单位来。
感觉前面的这个函数的逻辑有点乱。

'''


'''
下面这个函数还是没有写对，因为还需要考虑到之前的轮次就已经有request被部分释放的情况
现在开始fix这个问题
希望今天能跑出来一个结果。先fix这个问题再躺一会吧。
已经fix完了。
接下来就是和schedule的过程整合起来。然后再补充partial compute的过程
'''



def update_dict(data, k, v):
	if k not in data:
		data[k] = v
	else:
		data[k] = data[k] + v



def merge_dict(data1, data2):
	ret = {k:v for k, v in data1.items()}
	for k, v in data2.items():
		update_dict(ret, k, v)
	return ret









def _determine_swap_recompute(
	capacity: int, blk_size: int, blk_mem_size: int, 
	seq_group_lens: List[int], seq_group_tot_lens: List[int], request_ids: List[str]
	) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], int]:
	'''
	INPUT:	
		capacity: the total number of GPU blocks available
		blk_size: the number of tokens in a block
		blk_mem_size: the total memory consumption of a block (including all layers)
		seq_group_lens: the lenghts of the sequence group lengths in the current batch (prompt+output)
		seq_group_tot_lens: the expected total lenghts of the sequence group lengths in the current batch (prompt+output)
		# release_seq_remain_lens: the remaining lens of the requests where partial blocks have been released
		request_ids: the request id strings of the sequences corresponding to seq_group_lens
	OUTPUT:
		to_swap: the indices of the seq_groups to be swapped out.
		to_recompute: the indices of the seq_groups to be recomputed.
		number of swap-out blocks in each iteration.
	
	NOTE: we make decision every block-size iterations.
	'''
	# the current total block number (we will allocate enough blocks for each request for they to do block-size iterations)
	# include (1) the running requests and (2) the ready-to-run requests and (3) the partially swapped requests
	# as we will only allocate KV cache space for (sequence length - 1) tokens
	current_blk_nums = np.asarray([ math.ceil((length+blk_size-1) / blk_size)  for length in seq_group_lens ])
	# 为什么我们需要加上 release_infor 的block数量？这些block应该已经被释放了才对啊。这个地方确实写错了。
	# <jingzhi>@BUG here: already fixed
	# current_blk_num = sum(current_blk_nums) + sum([len(tmps) for tmps in mem_scheduler.release_infor.values()])
	current_blk_num = sum(current_blk_nums) + sum(mem_scheduler.on_card_info.values())


	# in next iteration group, how many blocks do we need

	# <jingzhi> TODO: 这个地方有2个bug：1. 如果一个request被release了，就不会再贡献block num increase 了。
	# 2. 有可能所有request都被release了（或者partially release了）所以需要再整理一下，确保至少有一个request活着。
	# 关于2，这里的选择逻辑可能还需要再仔细思考一下。
	# <jingzhi> TODO: 在决定swap in的时候也和问题1有同样的问题。

	extra = None
	if len(seq_group_tot_lens) > 0:
		# we know the expected total length of each request
		decrease = sum(current_blk_nums[np.asarray(seq_group_lens) + blk_size >= np.asarray(seq_group_tot_lens)])
		increase = sum(np.asarray(seq_group_lens) + blk_size < np.asarray(seq_group_tot_lens))
		extra = current_blk_num - decrease + increase - capacity
	else:
		# assume no requests will finish in this iteration group
		decrease = 0
		increase = len(seq_group_lens)
		extra = current_blk_num - decrease + increase - capacity
	
	# NOTE: all the requests in the current batch will attend the computation in case we do not know their exact total lengths
	# NOTE: a block can either be swapped or be recomputed
	# 还是不要搞得太复杂好了。
	if extra > 0:
		# need to do swap or recomputation
		# how many block swapping can be overlapped
		overlap_swap_blk_num = cost_model.get_swap_blk_num(seq_group_lens, blk_mem_size) * blk_size # get_swap_blk_num() is the blk num for each iteration
		cand_swap_list = np.argsort(seq_group_lens)

		# if we know the exact output length of each request, we should exclude the ones which will finish
		if len(seq_group_tot_lens) > 0:
			cand_swap_list = np.asarray([req_i for req_i in cand_swap_list \
				if seq_group_lens[req_i] + blk_size < seq_group_tot_lens[req_i] ])


		# store the remaining blk number that can be overlapped
		tmp_blk_num = min(overlap_swap_blk_num, extra)
		partial_swapped_req_is = dict()
		complete_swapped_req_is = dict()

		# first check the requests that have been partially released or the ready-to-run requests
		# for req_i, tmps in mem_scheduler.release_infor.items():
		
		# <jingzhi> DEBUG 
		print("checking on card info (requests not running)")

		for req_i, tmp in mem_scheduler.on_card_info.items():
			if tmp >= tmp_blk_num:
				# can stop here
				update_dict(partial_swapped_req_is, req_i, tmp_blk_num)
				mem_scheduler.decrease_on_card_blk_num(req_i, tmp_blk_num)

				if mem_scheduler.on_card_info[req_i] < 0:
					assert False, (mem_scheduler.on_card_info[req_i], tmp_blk_num)

				tmp_blk_num = 0
				break
			else:
				update_dict(complete_swapped_req_is, req_i, tmp)
				mem_scheduler.decrease_on_card_blk_num(req_i, tmp)

				if mem_scheduler.on_card_info[req_i] < 0:
					assert False, (mem_scheduler.on_card_info[req_i], tmp)


				tmp_blk_num = tmp_blk_num - tmp
			if tmp_blk_num == 0:
				break

		

		if (tmp_blk_num == 0) and (overlap_swap_blk_num>=extra):
			return (merge_dict(partial_swapped_req_is, complete_swapped_req_is), dict())


		quota_each_iter = np.full(blk_size, overlap_swap_blk_num//blk_size) # stores how many blocks can be swapped in each iteration
		# update quota
		swapped_num = min(overlap_swap_blk_num,extra)-tmp_blk_num
		quota_each_iter[swapped_num//quota_each_iter[0]+1] = quota_each_iter[0] - swapped_num%quota_each_iter[0]
		quota_each_iter[:swapped_num//quota_each_iter[0]] = 0




		# <jingzhi> DEBUG
		print("checking running requests")

		for enu_i, req_i in enumerate(cand_swap_list):
			if tmp_blk_num == 0:
				break
			tmp = (seq_group_lens[req_i]-1) // blk_size
			if tmp == 0:
				continue
			# <jingzhi> DEBUG
			# print(request_ids[req_i], tmp, tmp_blk_num)
			if tmp >= tmp_blk_num:
				# can stop here
				if tmp*blk_size < (seq_group_lens[req_i]-1):
					update_dict(partial_swapped_req_is, request_ids[req_i], tmp_blk_num)
				else:
					update_dict(complete_swapped_req_is, request_ids[req_i], tmp_blk_num)
				mem_scheduler.decrease_on_card_blk_num(
					request_ids[req_i], tmp_blk_num, tot=current_blk_nums[req_i] )

				if mem_scheduler.on_card_info[request_ids[req_i]] < 0:
					assert False, (mem_scheduler.on_card_info[request_ids[req_i]], tmp_blk_num, current_blk_nums[req_i])


				tmp_blk_num = 0
				break
			else:
				# should update quota_each_iter
				iter_i = np.searchsorted(np.cumsum(quota_each_iter), tmp)
				quota_each_iter[iter_i] = sum(quota_each_iter[:iter_i+1]) - tmp
				if iter_i>0:
					quota_each_iter[:iter_i-1] = 0

				if tmp*blk_size < (seq_group_lens[req_i]-1):
					# check whether the current block can be swapped
					last_blk_swap_iter = current_blk_nums[req_i]*blk_size - seq_group_lens[req_i] + 1
					if sum(quota_each_iter[last_blk_swap_iter:]) > 0:
						tmp = tmp + 1
						# which iter to swap the last complete block
						iter_i = np.argsort(quota_each_iter[last_blk_swap_iter:] <= 0, kind='stable')[0]
						quota_each_iter[last_blk_swap_iter+iter_i] = quota_each_iter[last_blk_swap_iter+iter_i] - 1
					update_dict(partial_swapped_req_is, request_ids[req_i], tmp)
				else:
					update_dict(complete_swapped_req_is, request_ids[req_i], tmp)
				

				mem_scheduler.decrease_on_card_blk_num(
					request_ids[req_i], tmp, tot=current_blk_nums[req_i] )

				if mem_scheduler.on_card_info[request_ids[req_i]] < 0:
					assert False, (mem_scheduler.on_card_info[request_ids[req_i]], tmp, current_blk_nums[req_i])


				tmp_blk_num = tmp_blk_num - tmp
			if tmp_blk_num == 0:
				break
		# 
		# max_swap_enu_i = enu_i

		if (tmp_blk_num == 0) and (extra <= overlap_swap_blk_num):
			# we only need swapping out, and it can be totally overlapped
			return (merge_dict(partial_swapped_req_is, complete_swapped_req_is), dict())
		else:

			# in this case, we can only swap the last blocks (which are being updated) of the running requests

			tmp_blk_num = (min(overlap_swap_blk_num, extra) - tmp_blk_num) + \
				(extra - min(overlap_swap_blk_num, extra))

			# <jingzhi> DEBUG
			print("checking blocks whose swapping cannot be overlapped")

			# not all blocks can be swapped and overlapped
			# consider swap or recompute to release the remaining blocks
			swap_cost = cost_model.get_swap_cost(tmp_blk_num, blk_mem_size)

			# recompute sequence lengths
			recomp_seq_is = dict()
			
			'''
			这个地方有个bug啊，因为partial_swapped_req_is 可能并没有对应的current_blk_nums，
			其实应该全都利用 on card info 来获取信息，I mean partial_swap 的部分.
			但是其实这个bug也不算是bug，因为如果有来自之前就存在的partially released requests，
			这个block选择的过程应该早就结束了。
			所以此处我们暂时不做修改
			'''

			# for req_i, tmp in \
				# [ (req_i, current_blk_nums[req_i] - tmp) for req_i, tmp in partial_swapped_req_is.items()] + \
				# 	list(zip([request_ids[_] for _ in cand_swap_list[max_swap_enu_i:]], current_blk_nums[cand_swap_list[max_swap_enu_i:]])):
			# 
			# we should consider both partial_swapped_req_is and complete_swapped_req_is here
			for req_idx in cand_swap_list:
				req_i = request_ids[req_idx]
				tmp = mem_scheduler.on_card_info[req_i] if req_i in mem_scheduler.on_card_info \
														else current_blk_nums[req_idx]

				if tmp == 0:
					continue
				if tmp >= tmp_blk_num:
					# can stop here
					update_dict(recomp_seq_is, req_i, tmp_blk_num)
					# NOTE: tot may not be tmp, but in that case, req_i must already in on_card_info\
					# so tot will not be used
					mem_scheduler.decrease_on_card_blk_num(req_i, tmp_blk_num, tot=tmp )

					if mem_scheduler.on_card_info[req_i] < 0:
						assert False, (mem_scheduler.on_card_info[req_i], tmp_blk_num, tmp)

					tmp_blk_num = 0
					break
				else:
					update_dict(recomp_seq_is, req_i, tmp)
					mem_scheduler.decrease_on_card_blk_num(req_i, tmp, tot=tmp )

					if mem_scheduler.on_card_info[req_i] < 0:
						assert False, (mem_scheduler.on_card_info[req_i], tmp, tmp)


					tmp_blk_num = tmp_blk_num - tmp
				if tmp_blk_num == 0:
					break

			comp_cost = cost_model.get_comp_cost([_[1] for _ in recomp_seq_is])
			if swap_cost <= comp_cost:
				return (merge_dict(merge_dict(partial_swapped_req_is, complete_swapped_req_is), recomp_seq_is), dict())
			else:
				return (merge_dict(partial_swapped_req_is, complete_swapped_req_is), recomp_seq_is)
	else:
		# we do not need to do swapping-out or recomputation
		return (dict(), dict())






def determine_swap_recompute(
	capacity: int, blk_size: int, blk_mem_size: int, 
	seq_group_lens: List[int], seq_group_tot_lens: List[int], request_ids: List[str]
	) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], int]:
	'''
	INPUT:	
		capacity: the total number of GPU blocks available
		blk_size: the number of tokens in a block
		blk_mem_size: the total memory consumption of a block (including all layers)
		seq_group_lens: the lenghts of the sequence group lengths in the current batch (prompt+output)
		seq_group_tot_lens: the expected total lenghts of the sequence group lengths in the current batch (prompt+output)
		# release_seq_remain_lens: the remaining lens of the requests where partial blocks have been released
		request_ids: the request id strings of the sequences corresponding to seq_group_lens
	OUTPUT:
		to_swap: the indices of the seq_groups to be swapped out.
		to_recompute: the indices of the seq_groups to be recomputed.
	
	NOTE: we make decision every block-size iterations.
	'''

	swap_dict, recompute_dict = _determine_swap_recompute(
		capacity, blk_size, blk_mem_size, 
		seq_group_lens, seq_group_tot_lens, request_ids)
	swap_list, recompute_list = mem_scheduler.update_release_info(swap_dict, recompute_dict)
	return swap_list, recompute_list, cost_model.get_swap_blk_num(seq_group_lens, blk_mem_size)





# 有可能出现一些block刚挪进gpu就又需要挪出来的情况，为了避免这种情况，需要watermark？
# watermark的判断需要再一开始决定swap in 和recompute的时候考虑，[已经考虑了]
# 同时还需要给能够完全swap in/ recompute 的request预留下一个iteration group的空间
def get_swapped_in_ready_to_run(
	free_block_num:int, ready_to_run:List[str], swap_in_list:List[str]
	) -> Tuple[List[str], List[str], List[str]]:
	run = list()
	ready, unready = list(), list()
	for req_i in swap_in_list:
		if (req_i not in mem_scheduler.release_infor) or (len(mem_scheduler.release_infor[req_i]) == 0):
			ready.append(req_i)
		else:
			unready.append(req_i)

	ready = ready_to_run + ready
	run = ready[:free_block_num]
	ready = ready[free_block_num:]
	return run, ready, unready






def determine_swap_in_do_recompute(
	capacity: int, watermark: int, blk_size: int, blk_mem_size: int, seq_group_lens: List[int], seq_group_tot_lens: List[int]
	) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
	'''
	Determine which blocks can be swapped in or recomputed in this block-size iterations, so that they can attend computation in next block-size iterations.
	INPUT:	
		capacity: the total number of GPU blocks available
		watermark: the reserved watermark of the gpu allocator
		blk_size: the number of tokens in a block
		blk_mem_size: the total memory consumption of a block (including all layers)
		seq_group_lens: the lenghts of the sequence group lengths in the current batch (prompt+output)
		seq_group_tot_lens: the expected total lenghts of the sequence group lengths in the current batch (prompt+output)
		# release_seq_lens: the lengths of released requests
	OUTPUT:
		to_swap: the indices of the seq_groups to be swapped in.
		to_recompute: the indices of the seq_groups to be recomputed.
	
	NOTE: we make decision every block-size iterations.
		This function is called when there is no swapping-out or release-for-recomputation.
	'''

	# include (1) the running requests and (2) the ready-to-run requests and (3) the partially swapped requests
	# we only allocate KV cache for (sequence length - 1) tokens
	current_blk_nums = np.asarray([ math.ceil((length+blk_size-1) / blk_size)  for length in seq_group_lens])
	# current_blk_num = sum(current_blk_nums) + sum([len(tmps) for tmps in mem_scheduler.release_infor.values()])
	current_blk_num = sum(current_blk_nums) + sum(mem_scheduler.on_card_info.values())


	# in next iteration group, how many spare blocks can we use
	extra = None
	if len(seq_group_tot_lens) > 0:
		# we know the expected total length of each request
		decrease = sum(current_blk_nums[np.asarray(seq_group_lens) + blk_size >= np.asarray(seq_group_tot_lens)])
		increase = sum(np.asarray(seq_group_lens) + blk_size < np.asarray(seq_group_tot_lens))
		extra = capacity - (current_blk_num - decrease + increase) - watermark
		extra = min(extra, capacity - current_blk_num) # because we will start swapping from the first iteration in this iteration group
	else:
		# assume no requests will finish in this block-size iteration group
		decrease = 0
		increase = len(seq_group_lens)
		extra = capacity - (current_blk_num - decrease + increase) - watermark


	# <jingzhi> DEBUG
	print(f"in determine_swap_in_do_recompute, current_blk_num: {current_blk_num}, extra: {extra}")
	if extra <= 0:
		# we cannot swap in any blocks to gpu
		return (list(), list())


	# 
	release_infor = mem_scheduler.release_infor

	# 应该按什么顺序来把request加回去？按倒序的顺序加回去，而且一次加一个完整的request
	# 但是倒序的顺序从哪里获得？
	cand_req_is = list(release_infor.keys())[::-1]

	# estimate the number of swapping-in blocks that can be overlapped
	swap_in_quota = cost_model.get_swap_blk_num(seq_group_lens, blk_mem_size) * blk_size

	tmp_blk_num = extra
	swap_in_blks = list()
	recompute_blks = list()
	for req_i in cand_req_is:
		# tmp = math.ceil((release_seq_lens[req_i] + blk_size)/blk_size)
		# 还需要考虑一个block-size iterations里面能被overlap的swap的量。对，所以我们只能swap in 这个量的block，其余的block留到下一个block-size iterations再做
		# 如果空间不够容纳当前的这个request，那还需要把它挪进去吗？还是不挪他挪别的优先级更低的request？这些都是choice啊，好难。还是严格挪进去吧，效果不好再改。
		# 不管还原哪些block应该都是OK的。
		# 如果swap的quota不够了，但是整体的空闲block还有空间，应该跳过swap 的quota去找别的request的recompute的block吗？可以是可以，但是先不这么干，因为感觉没啥好处
		if (tmp_blk_num == 0) or (swap_in_quota == 0):
			break
		# 

		count = 0
		for (blk_i, release_way) in release_infor[req_i][::-1]:
			if (tmp_blk_num == 0) or (swap_in_quota == 0):
				break
			# 
			if release_way == 'recompute':
				tmp_blk_num = tmp_blk_num - 1
				recompute_blks.append((req_i, blk_i))
			else:
				swap_in_quota = swap_in_quota - 1
				tmp_blk_num = tmp_blk_num - 1
				swap_in_blks.append((req_i, blk_i))
			count += 1

		
		# update release_infor and on_card_info
		release_infor[req_i] = release_infor[req_i][:len(release_infor[req_i]) - count]
		mem_scheduler.increase_on_card_blk_num(req_i, count)


	# 
	return (swap_in_blks, recompute_blks)













# ========================================================================================
# 下面这组函数的逻辑相比于前面的写法更简单一点：我们不以block size为单位来分摊swapping的压力，也不以block size为单位来做KV cache调度的决策
# 我们先在每次有新的request被release（不管是不是完全被release）的时候，都等级在可以选用的quota里面



class KVCacheState(object):
	"""docstring for KVCacheState"""
	def __init__(self):
		super(KVCacheState, self).__init__()
		self.released:Dict[str, List[Tuple[int, str]]] = dict() # {req_id: [(blk_i, release_way), (), ...]} where release_way is 'swap' or 'recompute'
		self.to_release:Dict[str, Tuple[int, int]] = dict() # {req_id: block number on card}
		self.tot_blk_num_to_release = 0



cache_state = KVCacheState()

def update_KV_state_peak_demand_aware_paged(to_release: List[Tuple[str, int]]):
	# print(to_release)
	for request_id, blk_num in to_release:
		cache_state.to_release[request_id] = (0, blk_num) # start block id, total blocks that can be released
		cache_state.tot_blk_num_to_release = cache_state.tot_blk_num_to_release + blk_num



# when the KV cache is not enough, get the blocks to release
def get_blocks_to_release(blk_num_to_release: int) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[str]]:

	def _add_released_blocks(request_id, start, end, to_swap, to_recompute, release_way='swap'):
		if request_id not in cache_state.released:
			cache_state.released[request_id] = list()
		cache_state.released[request_id] = cache_state.released[request_id] + \
			[(i, release_way) for i in range(start, end)]
		if release_way == 'swap':
			to_swap.extend([(request_id, i) for i in range(start, end)])
		else:
			to_recompute.extend([(request_id, i) for i in range(start, end)])


	cache_state.tot_blk_num_to_release = cache_state.tot_blk_num_to_release - blk_num_to_release
	
	to_swap = list()
	to_recompute = list()
	request_ids = list()
	for request_id, (start, end) in cache_state.to_release.items():
		if start == end:
			# this request has been completely released
			continue
		request_ids.append(request_id)
		if end-start >= blk_num_to_release:
			# can stop here
			cache_state.to_release[request_id] = (start+blk_num_to_release, end)
			_add_released_blocks(request_id, start, start + blk_num_to_release, to_swap, to_recompute, release_way='recompute')
			blk_num_to_release = 0
			break
		else:
			# release all remaining blocks of this request
			cache_state.to_release[request_id] = (end, end)
			_add_released_blocks(request_id, start, end, to_swap, to_recompute, release_way='recompute')
			blk_num_to_release = blk_num_to_release - (end - start)

	assert blk_num_to_release == 0
	# print(f"returned to_swap: {to_swap}")
	return to_swap, to_recompute, request_ids






def get_blocks_to_reload(request_ids_to_reload: List[str]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
	to_swap, to_recompute = list(), list()
	for request_id in request_ids_to_reload:
		to_reload = cache_state.released[request_id]
		cache_state.tot_blk_num_to_release = cache_state.tot_blk_num_to_release \
			- (cache_state.to_release[request_id][1]-cache_state.to_release[request_id][0])

		for i, release_way in to_reload:
			if release_way == 'swap':
				to_swap.append((request_id, i))
			else:
				to_recompute.append((request_id, i))
		del cache_state.released[request_id]

		del cache_state.to_release[request_id]

	return to_swap, to_recompute








