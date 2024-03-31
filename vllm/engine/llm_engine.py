import copy
from collections import defaultdict
import os
import time
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
                    Union)

from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               TokenizerGroup)
from vllm.utils import Counter, set_cuda_visible_devices, get_ip, get_open_port, get_distributed_init_method

if ray:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


# <jingzhi>
from vllm.core.block_manager import KVBlkPerLayerWeight


logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        device_config: The configuration related to the device.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"disable_custom_all_reduce={parallel_config.disable_custom_all_reduce}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"kv_cache_dtype={cache_config.cache_dtype}, "
            f"device_config={device_config.device}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.log_stats = log_stats
        self._verify_args()

        self._init_tokenizer()
        self.seq_counter = Counter()

        # Create the parallel GPU workers.

        # <jingzhi> Support backup workers to avoid re-launch ray actors
        self.backup_workers = list()
        self.backup_worker_node_and_gpu_ids = list()
        self.backup_node_workers = defaultdict(list)
        self.backup_distributed_init_method = None
        self.backup_driver_node_id = None


        if self.parallel_config.worker_use_ray:
            # Disable Ray usage stats collection.
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self._init_workers_ray(placement_group)
        else:
            self._init_workers()



        # <jingzhi> For Profiling
        start_time = time.perf_counter()

        # Profile the memory usage and initialize the cache.
        self._init_cache()


        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_workers_ray _init_cache time: {end_time - start_time}s")


        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC)

    def get_tokenizer_for_seq(self, sequence: Sequence):
        return self.tokenizer.get_lora_tokenizer(sequence.lora_request)

    def _init_workers(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self._run_workers("init_model")
        self._run_workers("load_model")

    def _init_tokenizer(self, **tokenizer_init_kwargs):
        init_kwargs = dict(
            enable_lora=bool(self.lora_config),
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        init_kwargs.update(tokenizer_init_kwargs)
        self.tokenizer: TokenizerGroup = TokenizerGroup(
            self.model_config.tokenizer, **init_kwargs)




    # TODO (jingzhi): check what is driver worker and whether our code can still work
    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        if (os.environ['USE_VLLM'] == 'True') or (os.environ['SOFT_RESCHEDULE'] == 'False'):
            self._init_workers_ray_ori(placement_group, **ray_remote_kwargs)
        else:
            if len(self.backup_workers) > 0:
                # we already have backup workers
                self._init_workers_ray_no_actor_init()
            else:
                self._init_workers_ray_init_actor(placement_group, **ray_remote_kwargs)





    # TODO (jingzhi): check what is driver worker and whether our code can still work
    def _init_workers_ray_ori(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        

        # <jingzhi> For Profiling
        start_time = time.perf_counter()



        if self.parallel_config.tensor_parallel_size == 1:
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            num_gpus = 1

        self.driver_dummy_worker: RayWorkerVllm = None
        self.workers: List[RayWorkerVllm] = []

        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):

            # <jingzhi>
            print(f"bundle: {bundle}")

            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            # <jingzhi> consider the cache gpus
            num_gpus = bundle['GPU']

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                
                # TODO <jingzhi> support changing env variables as we cannot init ray twice where the env variables in actors are initialized
                runtime_env={"env_vars": dict(os.environ)},

                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.model_config.trust_remote_code)

            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
            else:
                self.workers.append(worker)

        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        driver_node_id, driver_gpu_ids = ray.get(
            self.driver_dummy_worker.get_node_and_gpu_ids.remote())

        # <jingzhi> For DEBUG
        print(f"driver_node_id, driver_gpu_ids: {driver_node_id, driver_gpu_ids}")
        print(f"self.workers: {self.workers}")

        worker_node_and_gpu_ids = ray.get(
            [worker.get_node_and_gpu_ids.remote() for worker in self.workers])

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        node_workers[driver_node_id].append(0)
        node_gpus[driver_node_id].extend(driver_gpu_ids)
        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids,
                                               start=1):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)

        # <jingzhi> For DEBUG
        print(f"node_workers: {node_workers} node_gpus: {node_gpus}")


        for node_id, gpu_ids in node_gpus.items():
            # <jingzhi> we keep the visible gpus in the same order as we specify them
            if os.environ['USE_VLLM'] == 'False':
                if node_id == driver_node_id:
                    node_gpus[node_id] = [int(gpu_i) for gpu_i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
                    continue

            node_gpus[node_id] = sorted(gpu_ids)

        # <jingzhi> For DEBUG
        print(f"node_workers: {node_workers} node_gpus: {node_gpus}")


        # Set CUDA_VISIBLE_DEVICES for the driver.
        # <jingzhi>
        print(f"in main process os.environ['CUDA_VISIBLE_DEVICES']:{os.environ['CUDA_VISIBLE_DEVICES']}")

        set_cuda_visible_devices(node_gpus[driver_node_id])

        # <jingzhi>
        print(f"in main process os.environ['CUDA_VISIBLE_DEVICES']:{os.environ['CUDA_VISIBLE_DEVICES']}")


        for worker, (node_id, _) in zip(self.workers, worker_node_and_gpu_ids):
            worker.set_cuda_visible_devices.remote(node_gpus[node_id])

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        # Initialize torch distributed process group for the workers.
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        device_config = copy.deepcopy(self.device_config)


        # # <jingzhi>
        # import os
        # tot_ordered_gpus = ','.join(cuda_name for cuda_name in os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        # print(f"BEFORE init workers: os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}, tot_ordered_gpus: {tot_ordered_gpus}")


        # <jingzhi> For Profiling
        start_create_Worker = time.perf_counter()    


        for rank, (worker, (node_id, _)) in enumerate(zip(self.workers, worker_node_and_gpu_ids), start=1):
            local_rank = node_workers[node_id].index(rank)
            worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: Worker(
                    model_config,
                    parallel_config,
                    scheduler_config,
                    device_config,
                    local_rank,
                    rank,
                    distributed_init_method,
                    lora_config=self.lora_config,
                    kv_cache_dtype=self.cache_config.cache_dtype,
                ))

        driver_rank = 0
        driver_local_rank = node_workers[driver_node_id].index(driver_rank)
        self.driver_worker = Worker(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            driver_local_rank,
            driver_rank,
            distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )


        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_workers_ray only create Worker instances time: {end_time - start_create_Worker}")
        print(f"_init_workers_ray init workers time: {end_time - start_time}s")



        self._run_workers("init_model")


        # <jingzhi> For Profiling
        start_time = end_time
        end_time = time.perf_counter()
        print(f"_init_workers_ray init_model time: {end_time - start_time}s")


        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )


        # <jingzhi> For Profiling
        start_time = end_time
        end_time = time.perf_counter()
        print(f"_init_workers_ray load_model time: {end_time - start_time}s")










    # TODO (jingzhi): check what is driver worker and whether our code can still work
    # NOTE: the function below will not init new ray workers
    def _init_workers_ray_no_actor_init(self):
        

        # directly get the workers we need in this round of inference
        # num_actors should exclude the driver worker
        num_actors = self.parallel_config.tensor_parallel_size - 1
        print(f"in _init_workers_ray_no_actor_init, num_actors: {num_actors}")
        self.workers = self.backup_workers[:num_actors]
        print(f"in _init_workers_ray_no_actor_init, self.workers: {self.workers}")
        worker_node_and_gpu_ids = self.backup_worker_node_and_gpu_ids[:num_actors]
        node_workers = self.backup_node_workers
        distributed_init_method = self.backup_distributed_init_method
        driver_node_id = self.backup_driver_node_id


        for worker in self.workers:
            worker.update_envs.remote(dict(os.environ))


        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        # Initialize torch distributed process group for the workers.
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        device_config = copy.deepcopy(self.device_config)


        # <jingzhi> For Profiling
        start_create_Worker = time.perf_counter()    


        for rank, (worker, (node_id, _)) in enumerate(zip(self.workers, worker_node_and_gpu_ids), start=1):
            local_rank = node_workers[node_id].index(rank)
            worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: Worker(
                    model_config,
                    parallel_config,
                    scheduler_config,
                    device_config,
                    local_rank,
                    rank,
                    distributed_init_method,
                    lora_config=self.lora_config,
                    kv_cache_dtype=self.cache_config.cache_dtype,
                ))

        driver_rank = 0
        driver_local_rank = node_workers[driver_node_id].index(driver_rank)
        self.driver_worker = Worker(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            driver_local_rank,
            driver_rank,
            distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )


        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_workers_ray only create Worker instances time: {end_time - start_create_Worker}")
        # print(f"_init_workers_ray init workers time: {end_time - start_time}s")



        self._run_workers("init_model")


        # <jingzhi> For Profiling
        start_time = end_time
        end_time = time.perf_counter()
        print(f"_init_workers_ray init_model time: {end_time - start_time}s")


        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )


        # <jingzhi> For Profiling
        start_time = end_time
        end_time = time.perf_counter()
        print(f"_init_workers_ray load_model time: {end_time - start_time}s")






    # TODO (jingzhi): check what is driver worker and whether our code can still work
    # NOTE: the function below will init new ray workers, it is called when the model inference is started (not at rescheduled time)
    def _init_workers_ray_init_actor(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        

        # <jingzhi> For Profiling
        start_time = time.perf_counter()

        if self.parallel_config.tensor_parallel_size == 1:
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            num_gpus = 1

        self.driver_dummy_worker: RayWorkerVllm = None
        self.workers: List[RayWorkerVllm] = []

        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):

            # <jingzhi>
            print(f"bundle: {bundle}")

            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            # <jingzhi> consider the cache gpus
            num_gpus = bundle['GPU']

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                
                # TODO <jingzhi> support changing env variables as we cannot init ray twice where the env variables in actors are initialized
                runtime_env={"env_vars": dict(os.environ)},

                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.model_config.trust_remote_code)

            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
            else:
                self.workers.append(worker)



        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        driver_node_id, driver_gpu_ids = ray.get(
            self.driver_dummy_worker.get_node_and_gpu_ids.remote())

        # <jingzhi> For DEBUG
        print(f"driver_node_id, driver_gpu_ids: {driver_node_id, driver_gpu_ids}")
        print(f"self.workers: {self.workers}")

        worker_node_and_gpu_ids = ray.get(
            [worker.get_node_and_gpu_ids.remote() for worker in self.workers])

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        node_workers[driver_node_id].append(0)
        node_gpus[driver_node_id].extend(driver_gpu_ids)
        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids,
                                               start=1):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)

        # <jingzhi> For DEBUG
        print(f"node_workers: {node_workers} node_gpus: {node_gpus}")


        for node_id, gpu_ids in node_gpus.items():
            # <jingzhi> we keep the visible gpus in the same order as we specify them
            if os.environ['USE_VLLM'] == 'False':
                if node_id == driver_node_id:
                    node_gpus[node_id] = [int(gpu_i) for gpu_i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
                    continue

            node_gpus[node_id] = sorted(gpu_ids)

        # <jingzhi> For DEBUG
        print(f"node_workers: {node_workers} node_gpus: {node_gpus}")


        # Set CUDA_VISIBLE_DEVICES for the driver.
        # <jingzhi>
        print(f"in main process os.environ['CUDA_VISIBLE_DEVICES']:{os.environ['CUDA_VISIBLE_DEVICES']}")

        set_cuda_visible_devices(node_gpus[driver_node_id])

        # <jingzhi>
        print(f"in main process os.environ['CUDA_VISIBLE_DEVICES']:{os.environ['CUDA_VISIBLE_DEVICES']}")


        for worker, (node_id, _) in zip(self.workers, worker_node_and_gpu_ids):
            worker.set_cuda_visible_devices.remote(node_gpus[node_id])

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())
        

        # get backup variables so that we do not need to init ray actors again
        self.backup_workers = self.workers
        self.backup_worker_node_and_gpu_ids = worker_node_and_gpu_ids
        self.backup_node_workers = node_workers
        self.backup_distributed_init_method = distributed_init_method
        self.backup_driver_node_id = driver_node_id

        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_workers_ray init ray actors time: {end_time - start_time}s")

        self._init_workers_ray_no_actor_init()





    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)

    def _init_cache_ori(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameters.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.

        # <jingzhi> For Profiling
        start_time = time.perf_counter()
  
        
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_dtype=self.cache_config.cache_dtype,
        )


        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_cache profile_num_available_blocks time: {end_time - start_time}")


        # <jingzhi> after updating KVBlkPerLayerWeight in each worker, we update KVBlkPerLayerWeight in the main process as well------
        blk_num_per_layer, cached_layer_num = num_blocks[0][2]
        for b in num_blocks[1:]:
            assert b[2] == num_blocks[0][2]
        KVBlkPerLayerWeight.blk_num_per_layer, KVBlkPerLayerWeight.cached_layer_num = blk_num_per_layer, cached_layer_num
        print(f"in init_cache: num_blocks: {num_blocks}")
        # -----------------------------------------------------------------------------------------------------------------------------



        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.

        # <jingzhi> For Profiling
        start_time = time.perf_counter()


        self._run_workers("init_cache_engine", cache_config=self.cache_config)

        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_cache init_cache_engine time: {end_time - start_time}")     


        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        # TODO (jingzhi) check this code
        self._run_workers("warm_up_model")

        # <jingzhi> For Profiling
        start_time = end_time
        end_time = time.perf_counter()
        print(f"_init_cache warm_up_model time: {end_time - start_time}")  



    # <jingzhi> this function gets the reused cache profiling results from our profiling database
    def get_reused_cache_profile_result(self) -> Tuple[int, int, Tuple[int, int]]:
        '''
            The content of each profiling result record.
            num_gpu_blocks, num_cpu_blocks, (KVBlkPerLayerWeight.blk_num_per_layer, KVBlkPerLayerWeight.cached_layer_num)
        '''
        num_workers = self.parallel_config.world_size
        model_name = self.model_config.model
        cached_layer_num = KVBlkPerLayerWeight.cached_layer_num
        blk_num_per_layer = KVBlkPerLayerWeight.blk_num_per_layer
        # this dictionary returns the result when gpu utilization ratio = 0.9
        # TODO (jingzhi) need to update the database
        data_base = {('NousResearch/Llama-2-70b-hf', 2): (2057, 1638), 
                     ('NousResearch/Llama-2-70b-hf', 4): (29278, 3276),
                     ('NousResearch/Llama-2-13b-hf', 1): (3727, 327),
                     ('NousResearch/Llama-2-13b-hf', 2): (9369, 655),
                     ('NousResearch/Llama-2-13b-hf', 4): (20319, 1310),
                     ('huggyllama/llama-7b', 1): (7348, 512),
                     ('huggyllama/llama-7b', 2): (16209, 1024),
                     ('huggyllama/llama-7b', 4): (33287, 2048)}
        num_gpu_blocks, num_cpu_blocks = data_base[(model_name, num_workers)]
        print(f"in get_reused_cache_profile_result: num_gpu_blocks, num_cpu_blocks, cached_layer_num, blk_num_per_layer: {num_gpu_blocks, num_cpu_blocks, cached_layer_num, blk_num_per_layer}")
        # need consider gpu utilization ratio
        reduced_tot_mem = KVBlkPerLayerWeight.tot_gpu_mem*(0.9-self.cache_config.gpu_memory_utilization)
        num_gpu_blocks = num_gpu_blocks - \
            (reduced_tot_mem+KVBlkPerLayerWeight.block_size-1)//KVBlkPerLayerWeight.block_size
        print(f"in get_reused_cache_profile_result: after consider gpu_memory_utilization: num_gpu_blocks: {num_gpu_blocks}")
        num_gpu_blocks = num_gpu_blocks + cached_layer_num * blk_num_per_layer
        return int(num_gpu_blocks), int(num_cpu_blocks), (blk_num_per_layer, cached_layer_num)
    



    # <jingzhi> this function init the KV cache by reusing previous profiling result
    def _init_cache_no_profile(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameters.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.

        # <jingzhi> For Profiling
        start_time = time.perf_counter()

        # fake_profile_num_available_blocks will update KVBlkPerLayerWeight in each worker
        self._run_workers(
            "fake_profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_dtype=self.cache_config.cache_dtype,
        )

        
        num_blocks = self.get_reused_cache_profile_result()


        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_cache profile_num_available_blocks time: {end_time - start_time}")      


        # <jingzhi> after updating KVBlkPerLayerWeight in each worker, we update KVBlkPerLayerWeight in the main process as well------
        # blk_num_per_layer, cached_layer_num = num_blocks[0][2]
        # for b in num_blocks[1:]:
        #     assert b[2] == num_blocks[0][2]
        # KVBlkPerLayerWeight.blk_num_per_layer, KVBlkPerLayerWeight.cached_layer_num = blk_num_per_layer, cached_layer_num
        print(f"in init_cache: num_blocks: {num_blocks}")
        # -----------------------------------------------------------------------------------------------------------------------------



        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = num_blocks[0]
        num_cpu_blocks = num_blocks[1]

        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.

        # <jingzhi> For Profiling
        start_time = time.perf_counter()


        self._run_workers("init_cache_engine", cache_config=self.cache_config)

        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_cache init_cache_engine time: {end_time - start_time}")     


        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        # TODO (jingzhi) check this code
        self._run_workers("warm_up_model")

        # <jingzhi> For Profiling
        start_time = end_time
        end_time = time.perf_counter()
        print(f"_init_cache warm_up_model time: {end_time - start_time}")  



    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameters.
        """
        if (os.environ['USE_VLLM'] == 'True') or (os.environ['SOFT_RESCHEDULE'] == 'False'):
            self._init_cache_ori()
        else:
            self._init_cache_no_profile()



    # <jingzhi>
    def disable_P2P_access(self) -> None:
        '''
            Disable the P2P access between gpu cards
        '''
        self._run_workers("disable_p2ps") #, get_all_outputs=True)


    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]

        # <jingzhi> For Profiling
        start_time = time.perf_counter()

        # Initialize the cluster.
        placement_group = initialize_cluster(parallel_config)

        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"initialize_cluster time: {end_time - start_time}s")

        # Create the LLM engine.
        engine = cls(*engine_configs,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine
    

    # <jingzhi> support soft reschedule
    def update_from_engine_args(self, engine_args: EngineArgs) -> None:
        ''' Update an LLM engine from the engine arguments. '''
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]

        # <jingzhi> For Profiling
        start_time = time.perf_counter()

        # Initialize the cluster.
        # placement_group = initialize_cluster(parallel_config)
        placement_group = None # we do not need placement group when rescheduling models

        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"initialize_cluster time: {end_time - start_time}s")

        # Create the LLM engine.
        self.update_engine(*engine_configs,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)



    # <jingzhi> support soft reschedule
    def update_engine(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"disable_custom_all_reduce={parallel_config.disable_custom_all_reduce}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"kv_cache_dtype={cache_config.cache_dtype}, "
            f"device_config={device_config.device}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.log_stats = log_stats
        self._verify_args()

        self._init_tokenizer()
        # self.seq_counter = Counter() # <jingzhi>

        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            # Disable Ray usage stats collection.
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self._init_workers_ray(placement_group)
        else:
            self._init_workers()



        # <jingzhi> For Profiling
        start_time = time.perf_counter()

        # Profile the memory usage and initialize the cache.
        self._init_cache()


        # <jingzhi> For Profiling
        end_time = time.perf_counter()
        print(f"_init_workers_ray _init_cache time: {end_time - start_time}s")


        # Create the scheduler.
        # TODO (jingzhi) need to update scheduler instead of reinitialize it if we want to reuse KV cache
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC)







    def encode_request(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(request_id=request_id,
                                                     prompt=prompt,
                                                     lora_request=lora_request)
        return prompt_token_ids

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        prefix_pos: Optional[int] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `best_of` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.monotonic()
        prompt_token_ids = self.encode_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size,
                       lora_request)

        # Check whether the input specifies prefix
        prefix = self.scheduler.prefix_pool.add_or_get_prefix(
            prompt_token_ids[:prefix_pos], lora_request.lora_int_id
            if lora_request else 0) if prefix_pos is not None else None

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time, lora_request, prefix)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`
              from class :class:`~vllm.core.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = (current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.get_tokenizer_for_seq(
                current_worst_seq).eos_token_id))
        if early_stopping is False:
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=self.get_tokenizer_for_seq(
                    best_running_seq).eos_token_id))
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.get_tokenizer_for_seq(
                            best_running_seq).eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.get_tokenizer_for_seq(
                            best_running_seq).eos_token_id))
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:

        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params, best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in scheduled_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        for seq_group in scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        # Update prefix state, now all the uncomputed prefixes are computed.
        for seq_group in scheduled_seq_groups:
            if (seq_group.prefix is not None and seq_group.prefix.allocated
                    and not seq_group.prefix.computed):
                seq_group.prefix.computed = True

        # Log stats.
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs))

        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the workers to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id), prompt, sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            all_outputs = self._run_workers(
                "execute_model",
                driver_kwargs={
                    "seq_group_metadata_list": seq_group_metadata_list,
                    "blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
                    "blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
                    "blocks_to_copy": scheduler_outputs.blocks_to_copy,
                    # TODO <jingzhi>
                    "load_more_layer_on_card_num":KVBlkPerLayerWeight.load_more_layer_on_card_num,
                    "blocks_to_reorganize":scheduler_outputs.blocks_to_reorganize,
                })

            # Only the driver worker returns the sampling results.
            output = all_outputs[0]
        else:
            output = []


        # <jingzhi> we need to reset KVBlkPerLayerWeight.load_more_layer_on_card_num after we use it
        KVBlkPerLayerWeight.load_more_layer_on_card_num = 0

        return self._process_model_outputs(output, scheduler_outputs)

    def do_log_stats(self) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs=None))

    def _get_stats(self,
                   scheduler_outputs: Optional[SchedulerOutputs]) -> Stats:
        """Get Stats to be Logged to Prometheus."""
        now = time.monotonic()

        # KV Cache Usage in %.
        num_total_gpu = self.cache_config.num_gpu_blocks
        num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks()
        gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage = 0.
        if num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks(
            )
            cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

        # Scheduler State
        num_running = len(self.scheduler.running)
        num_swapped = len(self.scheduler.swapped)
        num_waiting = len(self.scheduler.waiting)

        # Iteration stats if we have scheduler output.
        num_prompt_tokens = 0
        num_generation_tokens = 0
        time_to_first_tokens = []
        time_per_output_tokens = []
        time_e2e_requests = []
        if scheduler_outputs is not None:
            prompt_run = scheduler_outputs.prompt_run

            # Number of Tokens.
            if prompt_run:
                num_prompt_tokens = scheduler_outputs.num_batched_tokens
            else:
                num_generation_tokens = scheduler_outputs.num_batched_tokens

            # Latency Timings.
            time_last_iters = []
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                # Time since last token. (n.b. updates seq_group.last_token_time)
                time_last_iters.append(seq_group.get_last_latency(now))
                # Time since arrival for all finished requests.
                if seq_group.is_finished():
                    time_e2e_requests.append(now - seq_group.arrival_time)

            time_to_first_tokens = time_last_iters if prompt_run else []
            time_per_output_tokens = [] if prompt_run else time_last_iters

        return Stats(
            now=now,
            num_running=num_running,
            num_swapped=num_swapped,
            num_waiting=num_waiting,
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_generation_tokens,
            time_to_first_tokens=time_to_first_tokens,
            time_per_output_tokens=time_per_output_tokens,
            time_e2e_requests=time_e2e_requests,
        )

    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             self.get_tokenizer_for_seq(seq),
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                if not sampling_params.include_stop_str_in_output:
                    # Truncate the output text so that the stop string is
                    # not included in the output.
                    seq.output_text = seq.output_text[:-len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos) and seq.get_last_token_id()
                == self.get_tokenizer_for_seq(seq).eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )

    def list_loras(self) -> List[int]:
        return self._run_workers("list_loras")

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the ray workers first.
        ray_worker_outputs = [
            worker.execute_method.remote(method, *args, **kwargs)
            for worker in self.workers
        ]

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Start the driver worker after all the ray workers.
        driver_worker_output = getattr(self.driver_worker,
                                       method)(*driver_args, **driver_kwargs)

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs





    # <jingzhi> delete worker to release resources
    def delete_workers(self, exit_actor=True) -> None:
        '''
            Input: exit_actor determines whether to kill the actor processes or not
        '''
        ray_worker_outputs = [
            worker.delete_worker.remote(exit_actor)
            for worker in self.workers
        ]

        # Start the driver worker after all the ray workers.
        del self.driver_worker
        
        import torch
        import gc
        # first destroy distributed process groups in torch
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        from vllm.model_executor.parallel_utils.custom_all_reduce import delete_handle
        delete_handle()
        destroy_model_parallel()
        torch.distributed.destroy_process_group()

        gc.collect()
        torch.cuda.empty_cache()


        # Get the results of the ray workers.
        if self.workers:
            # ray_worker_outputs = ray.get(ray_worker_outputs)
            # as we call actor exit inside the worker process, we should use ray.wait here
            ray.wait(ray_worker_outputs, num_returns=len(self.workers))

