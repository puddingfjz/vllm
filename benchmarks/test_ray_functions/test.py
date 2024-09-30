# 简单测试ray actor

import ray


class Counter_WORKER:
    def __init__(self):
        self.value = -1

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value += 1
        return self.value
    def get_counter(self):
        return self.value
    def init_worker(self):
        self.worker = Counter_WORKER()
        return self.worker.value

# Create an actor from this class.
counter = Counter.remote()
ray.get(counter.increment.remote())
ray.get(counter.init_worker.remote())



# 测试一下ray actor里能否在issue 一个ray actor，以及能否继承变量
import ray


@ray.remote
class Counter:
    def __init__(self, value):
        self.value = value
    def increment(self):
        self.value += 1
        return self.value
    def get_counter(self):
        return self.value
    def new_actor(self, value):
        counter = Counter.remote(value)
        return counter


# Create an actor from this class.
# counter = Counter.remote(0)
import time
start_time = time.time()
counter = Counter.remote(list(range(10**6)))
end_time = time.time()
print(f"total init time: {end_time-start_time}")

ray.get(counter.increment.remote())
counter1 = ray.get(counter.new_actor.remote(ray.get(counter.get_counter.remote())))
ray.get(counter.get_counter.remote())
ray.get(counter1.get_counter.remote())

