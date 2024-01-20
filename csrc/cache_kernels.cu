#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  char *src_ptr = static_cast<char*>(src.data_ptr());
  char *dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(
      dst_ptr + dst_offset,
      src_ptr + src_offset,
      block_size_in_bytes,
      memcpy_type,
      stream);
  }
}




void init_P2P_access(
  const int src_device_idx, 
  const int dst_device_idx,
  const int curr_device_idx) {

  // not sure whether we need to do this every time
  cudaSetDevice(src_device_idx);
  cudaDeviceEnablePeerAccess(dst_device_idx, 0);
  cudaSetDevice(dst_device_idx);
  cudaDeviceEnablePeerAccess(src_device_idx, 0);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("Cuda failure '%s'\n", \
             cudaGetErrorString(e));
  }

  // go back to current device
  cudaSetDevice(curr_device_idx);
}


void load_layer_weights(
  torch::Tensor& src,
  torch::Tensor& dst,
  const int layer_idx, 
  const int src_device_idx,
  const int dst_device_idx,
  const int curr_device_idx) {

  // // not sure whether we need to do this every time
  // cudaSetDevice(src_device_idx);
  // cudaDeviceEnablePeerAccess(dst_device_idx, 0);
  // cudaSetDevice(dst_device_idx);
  // cudaDeviceEnablePeerAccess(src_device_idx, 0);

  // cudaError_t e = cudaGetLastError();
  // if (e != cudaSuccess) {
  //   printf("Cuda failure '%s'\n", \
  //            cudaGetErrorString(e));
  // }

  // // go back to current device
  // cudaSetDevice(curr_device_idx);


  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // not sure whether we need synch here, I think we do not need this, otherwise we may need to wait previous copy to finish
  // cudaStreamSynchronize(stream);

  char *src_ptr = static_cast<char*>(src.data_ptr());
  char *dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t layer_weight_in_bytes = src.element_size() * src.numel();
  // printf("load amount '%d' '%d'\n", src.element_size(), src.numel());
  cudaMemcpyPeerAsync( dst_ptr, dst_device_idx, src_ptr, src_device_idx, layer_weight_in_bytes, stream );


  // cudaSetDevice(dst_device_idx);
  // cudaDeviceDisablePeerAccess(src_device_idx);
  // cudaSetDevice(src_device_idx);
  // cudaDeviceDisablePeerAccess(dst_device_idx);

  // // go back to current device
  // cudaSetDevice(curr_device_idx);
}





void disable_P2P_access(
  const int src_device_idx,
  const int dst_device_idx,
  const int curr_device_idx) {

  cudaSetDevice(dst_device_idx);
  cudaDeviceDisablePeerAccess(src_device_idx);
  cudaSetDevice(src_device_idx);
  cudaDeviceDisablePeerAccess(dst_device_idx);


  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("Cuda failure '%s'\n", \
             cudaGetErrorString(e));
  }

  // go back to current device
  cudaSetDevice(curr_device_idx);
}




namespace vllm {

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel_vllm(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}




// Grid: (num_layers, num_pairs)
// 
// <jingzhi> used when we change the KV cache layout
// the original data layout: 
// key: [num_blocks, num_kv_heads, head_size/x, block_size, x] for each layer
// value: [num_blocks, num_kv_heads, head_size, block_size] for each layer
// kv_caches layout: [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]; it contains both keys and values
// (for values: we can view kv_caches as [num_blocks, num_layers, 2, num_heads, head_size, block_size])
template<typename scalar_t>
__global__ void copy_blocks_kernel_layout_changed(
  int64_t kv_cache_ptr,
  const int64_t* __restrict__ block_mapping,
  const int64_t kv_cache_stride0,
  const int kv_cache_stride2) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  scalar_t* kv_cache_pointer = reinterpret_cast<scalar_t*>(kv_cache_ptr);

  int64_t base_sum = kv_cache_stride2 * 2 * layer_idx;
  scalar_t* key_cache_src = kv_cache_pointer + src_block_number * kv_cache_stride0 + base_sum;
  scalar_t* value_cache_src = kv_cache_pointer + src_block_number * kv_cache_stride0 + base_sum + kv_cache_stride2;
  scalar_t* key_cache_dst = kv_cache_pointer + dst_block_number * kv_cache_stride0 + base_sum;
  scalar_t* value_cache_dst = kv_cache_pointer + dst_block_number * kv_cache_stride0 + base_sum + kv_cache_stride2;

  for (int i = threadIdx.x; i < kv_cache_stride2; i += blockDim.x) {
    key_cache_dst[i] = key_cache_src[i];
  }
  for (int i = threadIdx.x; i < kv_cache_stride2; i += blockDim.x) {
    value_cache_dst[i] = value_cache_src[i];
  }
}


} // namespace vllm

// this is the original version function in vllm
void copy_blocks_vllm(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }
  // Create block mapping array.
  std::vector<int64_t> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    for (int64_t dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int64_t* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt64).to(cache_device);

  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key_caches[0].scalar_type(), "copy_blocks_kernel_vllm", ([&] {
      vllm::copy_blocks_kernel_vllm<scalar_t><<<grid, block, 0, stream>>>(
        key_cache_ptrs_tensor.data_ptr<int64_t>(),
        value_cache_ptrs_tensor.data_ptr<int64_t>(),
        block_mapping_tensor.data_ptr<int64_t>(),
        numel_per_block);
    }));
}





// <jingzhi> This is the version used when we change the KV cache layout
// the original data layout: 
// key: [num_blocks, num_kv_heads, head_size/x, block_size, x] for each layer
// value: [num_blocks, num_kv_heads, head_size, block_size] for each layer
// kv_caches layout: [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]; it contains both keys and values
// (for values: we can view kv_caches as [num_blocks, num_layers, 2, num_heads, head_size, block_size])
void copy_blocks_layout_changed(
  torch::Tensor& kv_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  int num_layers = kv_caches.size(1);
  // TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = kv_caches.device();
  TORCH_CHECK(cache_device.is_cuda());

  // using T = typename kv_caches.scalar_type();
  // T* kv_cache_ptr = reinterpret_cast<T*>(kv_caches.data_ptr());
  int64_t kv_cache_ptr = reinterpret_cast<int64_t>(kv_caches.data_ptr());

  // Create block mapping array.
  std::vector<int64_t> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    for (int64_t dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int64_t* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt64).to(cache_device);

  // Launch the kernel.
  // const int numel_per_block = kv_caches[0][0][0].numel();
  const int64_t kv_cache_stride0 = kv_caches.stride(0);
  const int kv_cache_stride2 = kv_caches.stride(2);
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, kv_cache_stride2));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    kv_caches.scalar_type(), "copy_blocks_kernel_layout_changed", ([&] {
      vllm::copy_blocks_kernel_layout_changed<scalar_t><<<grid, block, 0, stream>>>(
        kv_cache_ptr,
        block_mapping_tensor.data_ptr<int64_t>(),
        kv_cache_stride0,
        kv_cache_stride2);
    }));
}





// this is the dispatch function
// <jingzhi> used when we change the KV cache layout
// the original data layout: 
// key: [num_blocks, num_kv_heads, head_size/x, block_size, x] for each layer
// value: [num_blocks, num_kv_heads, head_size, block_size] for each layer
// kv_caches layout: [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]; it contains both keys and values
// (for values: we can view kv_caches as [num_blocks, num_layers, 2, num_heads, head_size, block_size])
void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {

  if (key_caches[0].sizes().size() > 5) {
    copy_blocks_layout_changed(key_caches[0], block_mapping);
  } else {
    copy_blocks_vllm(key_caches, value_caches, block_mapping);
  }
}





namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_kernel_vllm(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}





template<typename scalar_t>
__global__ void reshape_and_cache_kernel_layout_changed(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,           // [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,         // should be [num_blocks, num_layers, 2, num_heads, head_size, block_size]: the same tensor as key_cache
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x, 
  // added parameters
  const int layer_idx, 
  const int num_layers) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;

  // comp some constants
  const int64_t base_v = num_heads * (head_size / x) * block_size * x;
  const int64_t base_sum_key = base_v * 2 * (num_layers * block_idx + layer_idx);
  const int64_t base_sum_value = base_sum_key + base_v;


  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;


    // the KV cache layout is changed
    // key_value layout [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]
    const int64_t tgt_key_idx = base_sum_key
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;

    // should be [num_blocks, num_layers, 2, num_heads, head_size, block_size]
    const int64_t tgt_value_idx = base_sum_value
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;


    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}




} // namespace vllm

void reshape_and_cache_vllm(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_kernel_vllm",
    [&] {
      vllm::reshape_and_cache_kernel_vllm<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}


// <jingzhi> this version support the changed KV cache layout
void reshape_and_cache_layout_changed(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // should be [num_blocks, num_layers, 2, num_heads, head_size, block_size]: the same tensor as key_cache
  torch::Tensor& slot_mapping,  // [num_tokens]
  // added parameters
  const int layer_idx)
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(5);
  int x = key_cache.size(6);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  // added variable
  int num_layers = key_cache.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_kernel_layout_changed",
    [&] {
      vllm::reshape_and_cache_kernel_layout_changed<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x,
        // added parameters
        layer_idx, 
        num_layers);
    });
}





// <jingzhi> this version support the changed KV cache layout
void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x] or [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size] or [num_blocks, num_layers, 2, num_heads, head_size/x, block_size, x]: the same tensor as key_cache
  torch::Tensor& slot_mapping,  // [num_tokens]
  // added parameters
  const int layer_idx)
{
  if (key_cache.sizes().size() > 5){
    // the KV cache layout is changed
    reshape_and_cache_layout_changed(key, value, key_cache, value_cache, slot_mapping, layer_idx);
  }
  else{
    // the KV cache layout is the same as in vllm
    reshape_and_cache_vllm(key, value, key_cache, value_cache, slot_mapping);
  }
}







namespace vllm {

// Grid: (num_blocks, block_size).
template<typename scalar_t>
__global__ void gather_cached_kv_kernel(
  scalar_t* __restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
  scalar_t* __restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
  const scalar_t* __restrict__ key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int num_tokens = num_heads * head_size;
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
      const int tgt_key_idx = token_idx * key_stride + i;
      const int tgt_value_idx = token_idx * value_stride + i;
  
      const int head_idx = i / head_size;
      const int head_offset = i % head_size;
      const int x_idx = head_offset / x;  // the offset of the [head_size/x] dimension
      const int x_offset = head_offset % x;
  
      const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                              + head_idx * (head_size / x) * block_size * x
                              + x_idx * block_size * x
                              + block_offset * x
                              + x_offset;
      const int src_value_idx = block_idx * num_heads * head_size * block_size
                                + head_idx * head_size * block_size
                                + head_offset * block_size
                                + block_offset;

      key[tgt_key_idx] = VLLM_LDG(&key_cache[src_key_idx]);
      value[tgt_value_idx] = VLLM_LDG(&value_cache[src_value_idx]);
    }
}

template <typename scalar_t>
__global__ void gather_cached_kv_kernel_optimized(
    scalar_t *__restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
    scalar_t *__restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
    const scalar_t *__restrict__ key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    const scalar_t *__restrict__ value_cache, // [num_blocks, num_heads, head_size, block_size]
    const int *__restrict__ slot_mapping,   // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x)
{
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int dim = num_heads * head_size;
    assert(dim % 4 == 0);  // this is true for known use cases
    const int unroll_factor = 4;
    const int unrolled_dim = dim / unroll_factor;

    for (int i = threadIdx.x; i < unrolled_dim; i += blockDim.x)
    {
        int tgt_key_indices[unroll_factor];
        int tgt_value_indices[unroll_factor];
        int src_key_indices[unroll_factor];
        int src_value_indices[unroll_factor];
        scalar_t keys_to_store[unroll_factor];
        scalar_t values_to_store[unroll_factor];

        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j)
        {
            int index = i + j * unrolled_dim;

            const int tgt_key_idx = token_idx * key_stride + index;
            const int tgt_value_idx = token_idx * value_stride + index;

            const int head_idx = index / head_size;
            const int head_offset = index % head_size;
            const int x_idx = head_offset / x;
            const int x_offset = head_offset % x;

            const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                    + head_idx * (head_size / x) * block_size * x
                                    + x_idx * block_size * x
                                    + block_offset * x
                                    + x_offset;
            const int src_value_idx = block_idx * num_heads * head_size * block_size
                                      + head_idx * head_size * block_size
                                      + head_offset * block_size
                                      + block_offset;

            tgt_key_indices[j] = tgt_key_idx;
            tgt_value_indices[j] = tgt_value_idx;
            src_key_indices[j] = src_key_idx;
            src_value_indices[j] = src_value_idx;

            keys_to_store[j] = VLLM_LDG(&key_cache[src_key_idx]);
            values_to_store[j] = VLLM_LDG(&value_cache[src_value_idx]);
        }

        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j)
        {
            key[tgt_key_indices[j]] = keys_to_store[j];
            value[tgt_value_indices[j]] = values_to_store[j];
        }
    }
}

} // namespace vllm

void gather_cached_kv(
  torch::Tensor& key,           // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [in]  [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [in]  [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [in]  [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "gather_cached_kv_kernel_optimized",
    [&] {
      vllm::gather_cached_kv_kernel_optimized<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}
