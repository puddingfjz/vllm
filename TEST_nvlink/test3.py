
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'




import ray
# import ray.util.collective as collective
from ray.air.util.torch_dist import init_torch_dist_process_group
from ray.air.util.torch_dist import TorchDistributedWorker
from vllm.engine.arg_utils import EngineArgs


import torch


# vllm
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel_withOtherGPUAsCache)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)




gpu_i_map = {0:0, 1:1}



def _init_distributed_environment() -> None:
    """Initialize the distributed environment."""
    assert torch.distributed.is_initialized()
    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())



# define worker to transfer data between GPU 
# try to use Ray first to see performance
@ray.remote(num_gpus=1)
class ParamLoader(TorchDistributedWorker):
    def __init__(self):
        self.gpu_i = int(ray.get_gpu_ids()[0])
        # self.weights_gpu = weights_gpus[self.gpu_i[0]]
        size = 202383360
        # size = 200
        print(f"gpu_i: {self.gpu_i}")
        # self.weights_gpu = torch.Tensor(range(size)).to(0).view(2, -1)
        self.weights_gpu = torch.Tensor(range(size)).view(2, -1)
        self.rank = None
        self.device = None
        self.wait_ops = []
        # 
        self.llm = LLM(model="huggyllama/llama-7b", gpu_memory_utilization=0.2)
        self.model_runner = None
    # def setup(self, world_size, rank):
    #     collective.init_collective_group(world_size, rank, "nccl", "param_loading")
    #     return True
    # def load_weights_from_gpu(self, layer_i, tgt_rank):
    #     collective.send(self.weights_gpu2[layer_i], tgt_rank, group_name="param_loading" )
    # def store_weights_to_gpu(self, layer_i, src_rank):
    #     collective.recv(self.weights_gpu2[layer_i], tgt_rank, group_name="param_loading" )
    # 
    def load_weights_from_gpu(self, layer_i, tgt_rank):
        print("loading Tensor: ", self.weights_gpu[layer_i])
        self.wait_ops = [torch.distributed.isend(self.weights_gpu[layer_i], tgt_rank, group=None, tag=0)]
        return True
    def store_weights_to_gpu(self, layer_i, src_rank):
        print("storing Tensor: ", self.weights_gpu[layer_i])
        self.wait_ops = [torch.distributed.irecv(self.weights_gpu[layer_i], src=src_rank, group=None, tag=0)]
        return True
    def synchronize_loading(self):
        for event in self.wait_ops:
            event.wait()
        return True
    # 
    def init_status(self):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        # 
        print(self.rank, os.getenv("RANK", "-1"), torch.distributed.get_rank(), ray.get_gpu_ids())
        # 
        print(os.environ["CUDA_VISIBLE_DEVICES"], os.environ["LOCAL_WORLD_SIZE"])
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        print(f"local_rank: {local_rank}")
        # self.device = torch.device(f"cuda:{local_rank}")
        self.device = torch.device(f"cuda:{0}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)
        # 
        # _check_if_gpu_supports_dtype(self.model_config.dtype)
        # 
        # Initialize the distributed environment.
        _init_distributed_environment()
    # 
    def get_gpu_i(self):
        return self.gpu_i
    # 
    def generate_outputs(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs
    # 
    def init_model(self, model_name, **kwargs):
        engine_args = EngineArgs(
            model_name,
            tokenizer = None,
            tokenizer_mode = "auto",
            trust_remote_code = False,
            tensor_parallel_size = 1,
            dtype = "auto",
            quantization = None,
            revision = None,
            tokenizer_revision = None,
            seed = 0,
            gpu_memory_utilization = 0.9,
            swap_space = 4,
            **kwargs,
        )
        engine_configs = engine_args.create_engine_configs()
        model_config, cache_config, parallel_config, scheduler_config = engine_configs

        initialize_model_parallel_withOtherGPUAsCache(parallel_config.tensor_parallel_size,
                          parallel_config.pipeline_parallel_size)
        self.model_runner = ModelRunner(model_config, parallel_config,
                        scheduler_config)
    def load_model(self):
        self.model_runner.load_model()



def init_param_loaders():
    # imperative
    num_workers = 2
    workers = []
    init_rets = []
    # 
    # maybe the first solution to do initialization?
    # for i in range(num_workers):
    #    w = ParamLoader.remote()
    #    workers.append(w)
    #    init_rets.append(w.setup.remote(num_workers, i))
    # _ = ray.get(init_rets)
    # 
    # maybe the second solution to do initialization?
    workers = [0, 0]
    for i in range(num_workers):
        w = ParamLoader.remote()
        workers[gpu_i_map[ray.get(w.get_gpu_i.remote())]] = w
    # 
    # _options = {
    #     "group_name": "param_loading",
    #     "world_size": 2,
    #     "ranks": [0, 1],
    #     "backend": "nccl"
    # }
    # # Put A and B in a collective group
    # collective.create_collective_group(workers, **_options)
    # 
    return workers



















def init_distribute_env_in_torch(workers):
    init_torch_dist_process_group(workers, backend="nccl")
    # init param loader status
    ray.get([w.init_status.remote() for w in workers])








def load_layer_weights(param_loaders, layer_i:int):
    # let A to send a message to B; a send/recv has to be specified once at each worker
    res = ray.get([param_loaders[0].load_weights_from_gpu.remote(layer_i, 1), param_loaders[1].store_weights_to_gpu.remote(layer_i, 0)])




def synchronize_loading(param_loaders):
    res = ray.get([w.synchronize_loading.remote() for w in param_loaders])




# size = 202383360
# weight_cache = torch.Tensor(range(size)).cuda(0).view(2, -1)
# weight_cache_spare_gpu = torch.Tensor([1]*size).cuda(1).view(2, -1)
# workers = init_param_loaders(weight_cache, weight_cache_spare_gpu)

workers = init_param_loaders()
init_distribute_env_in_torch(workers)

load_layer_weights(workers, 0)



synchronize_loading(workers)


outputs = ray.get([w.generate_outputs.remote(prompts, sampling_params) for workers])
outputs = outputs[0] + outputs[1]
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



