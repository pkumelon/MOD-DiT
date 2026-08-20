#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <type_traits>

namespace {

using namespace nvcuda;

constexpr int kWarpSize = 32;
constexpr int kWmmaTile = 16;
constexpr int kMaxWarps = 8;

template <typename scalar_t, typename wmma_t>
__device__ __forceinline__ wmma_t scaled_cast(scalar_t value, float scale);

template <>
__device__ __forceinline__ __half scaled_cast<at::Half, __half>(at::Half value, float scale) {
  return __float2half_rn(static_cast<float>(value) * scale);
}

template <>
__device__ __forceinline__ __nv_bfloat16 scaled_cast<at::BFloat16, __nv_bfloat16>(
    at::BFloat16 value,
    float scale) {
  return __float2bfloat16_rn(static_cast<float>(value) * scale);
}

template <typename scalar_t, typename wmma_t, int kBlock, int kHeadDim>
__global__ void flash_sparsity_replay_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const float* __restrict__ lse,
    at::Half* __restrict__ sparsity,
    float softmax_scale,
    float log_threshold,
    int batch_heads,
    int sequence_length,
    int q_video_offset,
    int k_video_offset,
    int video_blocks) {
  constexpr int kWarps = kBlock / kWmmaTile;
  static_assert(kWarps <= kMaxWarps, "too many warps");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return;
#else

  const int query_block = static_cast<int>(blockIdx.x);
  const int batch_head = static_cast<int>(blockIdx.y);
  if (query_block >= video_blocks || batch_head >= batch_heads) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  // Dynamic shared-memory layout:
  //   scaled Q [kBlock, kHeadDim]
  //   K        [kBlock, kHeadDim]
  //   LSE      [kBlock]
  //   WMMA accumulator scratch [kWarps, 16, 16]
  extern __shared__ __align__(32) unsigned char shared_raw[];
  auto* shared_q = reinterpret_cast<wmma_t*>(shared_raw);
  auto* shared_k = shared_q + kBlock * kHeadDim;
  auto* shared_lse = reinterpret_cast<float*>(shared_k + kBlock * kHeadDim);
  auto* score_scratch = shared_lse + kBlock;
  __shared__ int warp_counts[kMaxWarps];

  const int q_start = q_video_offset + query_block * kBlock;
  const int64_t q_base = static_cast<int64_t>(batch_head) * sequence_length * kHeadDim;
  const int64_t lse_base = static_cast<int64_t>(batch_head) * sequence_length;

  for (int linear = tid; linear < kBlock * kHeadDim; linear += blockDim.x) {
    const int row = linear / kHeadDim;
    const int dim = linear - row * kHeadDim;
    shared_q[linear] = scaled_cast<scalar_t, wmma_t>(
        query[q_base + static_cast<int64_t>(q_start + row) * kHeadDim + dim],
        softmax_scale);
  }
  for (int row = tid; row < kBlock; row += blockDim.x) {
    shared_lse[row] = lse[lse_base + q_start + row];
  }
  __syncthreads();

  for (int key_block = 0; key_block < video_blocks; ++key_block) {
    const int k_start = k_video_offset + key_block * kBlock;
    for (int linear = tid; linear < kBlock * kHeadDim; linear += blockDim.x) {
      const int row = linear / kHeadDim;
      const int dim = linear - row * kHeadDim;
      shared_k[linear] = reinterpret_cast<const wmma_t*>(key)[
          q_base + static_cast<int64_t>(k_start + row) * kHeadDim + dim];
    }
    __syncthreads();

    if (warp_id < kWarps) {
      int warp_total = 0;
#pragma unroll
      for (int column_tile = 0; column_tile < kBlock; column_tile += kWmmaTile) {
        wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float> acc;
        wmma::fill_fragment(acc, 0.0f);

#pragma unroll
        for (int dim = 0; dim < kHeadDim; dim += kWmmaTile) {
          wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile,
                         wmma_t, wmma::row_major> q_fragment;
          wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile,
                         wmma_t, wmma::col_major> k_fragment;
          wmma::load_matrix_sync(
              q_fragment,
              shared_q + warp_id * kWmmaTile * kHeadDim + dim,
              kHeadDim);
          // K is stored row-major as [token, dim]. Interpreting the same memory
          // as a column-major [dim, token] matrix gives K^T without a transpose.
          wmma::load_matrix_sync(
              k_fragment,
              shared_k + column_tile * kHeadDim + dim,
              kHeadDim);
          wmma::mma_sync(acc, q_fragment, k_fragment, acc);
        }

        float* warp_scores = score_scratch + warp_id * kWmmaTile * kWmmaTile;
        wmma::store_matrix_sync(warp_scores, acc, kWmmaTile, wmma::mem_row_major);
        __syncwarp();

        int tile_count = 0;
        for (int element = lane_id; element < kWmmaTile * kWmmaTile; element += kWarpSize) {
          const int local_row = element / kWmmaTile;
          const float threshold_line = shared_lse[warp_id * kWmmaTile + local_row] + log_threshold;
          tile_count += static_cast<int>(warp_scores[element] < threshold_line);
        }
        tile_count = __reduce_add_sync(0xffffffffu, tile_count);
        if (lane_id == 0) {
          warp_total += tile_count;
        }
      }
      if (lane_id == 0) {
        warp_counts[warp_id] = warp_total;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      int block_count = lane_id < kWarps ? warp_counts[lane_id] : 0;
      block_count = __reduce_add_sync(0xffffffffu, block_count);
      if (lane_id == 0) {
        const float ratio = static_cast<float>(block_count) /
                            static_cast<float>(kBlock * kBlock);
        reinterpret_cast<__half*>(sparsity)[
            (static_cast<int64_t>(batch_head) * video_blocks + query_block) * video_blocks +
            key_block] = __float2half_rn(ratio);
      }
    }
    __syncthreads();
  }
#endif
}

template <typename scalar_t, typename wmma_t, int kBlock, int kHeadDim>
void launch_typed(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& lse,
    torch::Tensor& output,
    float softmax_scale,
    float log_threshold,
    int q_video_offset,
    int k_video_offset,
    int video_blocks,
    cudaStream_t stream) {
  constexpr int kWarps = kBlock / kWmmaTile;
  constexpr int kThreads = kWarps * kWarpSize;
  const size_t shared_bytes =
      2 * kBlock * kHeadDim * sizeof(wmma_t) +
      kBlock * sizeof(float) +
      kWarps * kWmmaTile * kWmmaTile * sizeof(float);

  auto kernel = flash_sparsity_replay_kernel<scalar_t, wmma_t, kBlock, kHeadDim>;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device >= 0 && device < 32, "unsupported CUDA device index: ", device);
  static std::mutex attribute_mutex;
  static bool attribute_configured[32] = {};
  {
    std::lock_guard<std::mutex> guard(attribute_mutex);
    if (!attribute_configured[device]) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_bytes)));
      attribute_configured[device] = true;
    }
  }

  const int batch_heads = static_cast<int>(query.size(0) * query.size(1));
  const int sequence_length = static_cast<int>(query.size(2));
  const dim3 grid(video_blocks, batch_heads);
  kernel<<<grid, kThreads, shared_bytes, stream>>>(
      query.data_ptr<scalar_t>(),
      key.data_ptr<scalar_t>(),
      lse.data_ptr<float>(),
      output.data_ptr<at::Half>(),
      softmax_scale,
      log_threshold,
      batch_heads,
      sequence_length,
      q_video_offset,
      k_video_offset,
      video_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename wmma_t>
void dispatch_shape(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& lse,
    torch::Tensor& output,
    float softmax_scale,
    float log_threshold,
    int q_video_offset,
    int k_video_offset,
    int video_blocks,
    int block_size,
    cudaStream_t stream) {
  const int head_dim = static_cast<int>(query.size(3));
#define LAUNCH(BLOCK, DIM)                                                                  \
  launch_typed<scalar_t, wmma_t, BLOCK, DIM>(                                               \
      query, key, lse, output, softmax_scale, log_threshold, q_video_offset,                \
      k_video_offset, video_blocks, stream)

  if (block_size == 64 && head_dim == 64) {
    LAUNCH(64, 64);
  } else if (block_size == 64 && head_dim == 128) {
    LAUNCH(64, 128);
  } else if (block_size == 128 && head_dim == 64) {
    LAUNCH(128, 64);
  } else if (block_size == 128 && head_dim == 128) {
    LAUNCH(128, 128);
  } else {
    TORCH_CHECK(false, "unsupported block_size/head_dim combination");
  }
#undef LAUNCH
}

void check_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& lse,
    int q_video_offset,
    int k_video_offset,
    int video_blocks,
    int block_size) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && lse.is_cuda(), "Q, K, and LSE must be CUDA tensors");
  TORCH_CHECK(query.device() == key.device() && query.device() == lse.device(), "Q, K, and LSE must share a device");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "Q and K must have shape [B, H, S, D]");
  TORCH_CHECK(query.sizes() == key.sizes(), "Q and K shapes must match");
  TORCH_CHECK(lse.dim() == 3, "LSE must have shape [B, H, S]");
  TORCH_CHECK(lse.size(0) == query.size(0) && lse.size(1) == query.size(1) &&
              lse.size(2) == query.size(2), "LSE shape must match Q except for head_dim");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && lse.is_contiguous(), "Q, K, and LSE must be contiguous");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Q and K dtypes must match");
  TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
              "Q and K must be float16 or bfloat16");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "LSE must be float32");
  TORCH_CHECK(block_size == 64 || block_size == 128, "block_size must be 64 or 128");
  TORCH_CHECK(query.size(3) == 64 || query.size(3) == 128, "head_dim must be 64 or 128");
  TORCH_CHECK(video_blocks > 0, "video_blocks must be positive");
  TORCH_CHECK(q_video_offset >= 0 && k_video_offset >= 0, "video offsets must be non-negative");
  const int64_t video_tokens = static_cast<int64_t>(video_blocks) * block_size;
  TORCH_CHECK(q_video_offset + video_tokens <= query.size(2), "Q video region exceeds sequence length");
  TORCH_CHECK(k_video_offset + video_tokens <= key.size(2), "K video region exceeds sequence length");
}

}  // namespace

torch::Tensor flash_sparsity_replay_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor lse,
    double softmax_scale,
    double log_threshold,
    int64_t q_video_offset,
    int64_t k_video_offset,
    int64_t video_blocks,
    int64_t block_size) {
  check_inputs(
      query, key, lse, static_cast<int>(q_video_offset), static_cast<int>(k_video_offset),
      static_cast<int>(video_blocks), static_cast<int>(block_size));

  c10::cuda::CUDAGuard device_guard(query.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major >= 8, "CUDA sparsity backend requires SM80 or newer; got SM",
              properties->major, properties->minor);

  auto output = torch::empty(
      {query.size(0), query.size(1), video_blocks, video_blocks},
      query.options().dtype(torch::kFloat16));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();

  if (query.scalar_type() == at::kHalf) {
    dispatch_shape<at::Half, __half>(
        query, key, lse, output, static_cast<float>(softmax_scale),
        static_cast<float>(log_threshold), static_cast<int>(q_video_offset),
        static_cast<int>(k_video_offset), static_cast<int>(video_blocks),
        static_cast<int>(block_size), stream);
  } else {
    dispatch_shape<at::BFloat16, __nv_bfloat16>(
        query, key, lse, output, static_cast<float>(softmax_scale),
        static_cast<float>(log_threshold), static_cast<int>(q_video_offset),
        static_cast<int>(k_video_offset), static_cast<int>(video_blocks),
        static_cast<int>(block_size), stream);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "sparsity_map",
      &flash_sparsity_replay_cuda,
      "Exact block sparsity-map replay with SM80 Tensor Cores");
}
