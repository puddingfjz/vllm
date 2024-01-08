




# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP



import torch.utils.benchmark as benchmark


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)






def cleanup():
    dist.destroy_process_group()





class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)
    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)



def transfer(a, b):
    _ = b.copy_(a)
    torch.cuda.synchronize()


def demo_model_parallel(rank, world_size, a, b):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)
    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1

    size = 202383360
    # a = torch.Tensor(range(size)).to(dev0)
    # b = torch.Tensor([1] * (size)).to(dev1)
    # c = a.to(1)

    print(a, b)

    t0 = benchmark.Timer(
        stmt='transfer(a, b)',
        setup='from __main__ import transfer',
        globals={'a': a, 'b': b})


    print(t0.timeit(100))


    cleanup()





def run_demo(demo_fn, world_size, a, b):
    mp.spawn(demo_fn,
             args=(world_size, a, b),
             nprocs=world_size,
             join=True)




if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    world_size = n_gpus//2
    world_size = 1

    size = 202383360
    a = torch.Tensor(range(size)).to(2)
    b = torch.Tensor([1] * (size)).to(1)

    print(a, b)

    t0 = benchmark.Timer(
        stmt='transfer(a, b)',
        setup='from __main__ import transfer',
        globals={'a': a, 'b': b})


    print(t0.timeit(100))



    t0 = benchmark.Timer(
        stmt='transfer(a, b)',
        setup='from __main__ import transfer',
        globals={'a': a, 'b': b})


    print(t0.timeit(100))


    run_demo(demo_model_parallel, world_size, a, b)


