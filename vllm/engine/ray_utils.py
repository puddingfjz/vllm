import socket
from typing import Optional, Tuple, TYPE_CHECKING

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)

try:
    import ray
    from ray.air.util.torch_dist import TorchDistributedWorker

    class RayWorkerVllm(TorchDistributedWorker):
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, init_cached_hf_modules=False) -> None:
            if init_cached_hf_modules:
                from transformers.dynamic_module_utils import init_hf_modules
                init_hf_modules()
            self.worker = None

        def init_worker(self, worker_init_fn):
            self.worker = worker_init_fn()

        def __getattr__(self, name):
            return getattr(self.worker, name)

        def execute_method(self, method, *args, **kwargs):
            executor = getattr(self, method)
            return executor(*args, **kwargs)

except ImportError as e:
    logger.warning(f"Failed to import Ray with {e!r}. "
                   "For distributed inference, please install Ray with "
                   "`pip install ray pandas pyarrow`.")
    ray = None
    TorchDistributedWorker = None
    RayWorkerVllm = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, Optional["PlacementGroup"]]:
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        A tuple of (`distributed_init_method`, `placement_group`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError(
                "Ray is not installed. Please install Ray to use distributed "
                "serving.")
        # Connect to a ray cluster.
        if is_hip():
            ray.init(address=ray_address,
                     ignore_reinit_error=True,
                     num_gpus=parallel_config.world_size)

            # <jingzhi>
            print(f"is hip-------")

        else:
            ray.init(address=ray_address, ignore_reinit_error=True)


            # <jingzhi>
            print(f"not hip-------")

    if not parallel_config.worker_use_ray:
        # Initialize cluster locally.
        port = get_open_port()
        # We need to setup the distributed init method to make sure
        # the distributed megatron code (e.g., get world size) works correctly.
        distributed_init_method = f"tcp://localhost:{port}"
        return distributed_init_method, None


    # <jingzhi>
    print(f"parallel_config.worker_use_ray: {parallel_config.worker_use_ray}")


    current_placement_group = ray.util.get_current_placement_group()

    print(f"current_placement_group: {current_placement_group}")

    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if parallel_config.world_size > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the placement group.")
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)


        # <jingzhi>
        print(f'num_gpus_in_cluster: {num_gpus_in_cluster}, parallel_config.world_size: {parallel_config.world_size}')


        if parallel_config.world_size > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster.")
        # Create a new placement group

        # <jingzhi>---------------------------------------------------------------
        # consider cache gpu at the same time
        # gpu_num_per_worker = num_gpus_in_cluster//parallel_config.world_size
        # current_placement_group = ray.util.placement_group([{
        #     "GPU": gpu_num_per_worker # 1
        # }] * parallel_config.world_size)

        current_placement_group = ray.util.placement_group(get_gpu_assignment(parallel_config.world_size, num_gpus_in_cluster))
        # ------------------------------------------------------------------------

        # current_placement_group = ray.util.placement_group([{
        #     "GPU": 1
        # }] * parallel_config.world_size)
        
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        ray.get(current_placement_group.ready(), timeout=1800)

    return None, current_placement_group







# <jingzhi>
def get_gpu_assignment(worker_num: int, tot_gpu_num: int):
    '''
    Get the gpu number assignment plan given the worker number and the total gpu number.
    Principle: the purpose is to make all gpus of the machine node visible to the workers on the machine.
    '''
    # --------------------------------------------------------
    # gpu_num_per_worker = tot_gpu_num//worker_num
    # plan = [{"GPU": gpu_num_per_worker}] * worker_num
    # --------------------------------------------------------


    # an example of the policy below: there are 4 GPUs, but 3 workers, will generate an assignment plan as {GPU:1, GPU:1, GPU:2}

    gpu_num_per_worker = tot_gpu_num//worker_num
    plan = [{"GPU": gpu_num_per_worker}] * (worker_num-1) + [{"GPU": tot_gpu_num - gpu_num_per_worker*(worker_num-1)}]
    return plan

