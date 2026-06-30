#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel用于计算M^T·M，其中M由特殊结构矩阵组成
// M = [S_0, C_1, ..., C_{2n-1}, D_1, ..., D_n, E]
// 支持多头attention，并使用blocks_per_frame而非block_starts/block_sizes
// 利用矩阵的稀疏性和特殊结构避免n^5的计算复杂度

#define BLOCK_SIZE 16
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Kernel 2: 计算S_0与对角线矩阵C_k的内积，支持batch维度
// 输出：shape (head, 2n-1)
__global__ void compute_s0_diag_kernel(
    const float* __restrict__ S_0,
    float* __restrict__ results,
    int head_num,
    int n
) {
    int k = blockIdx.x;  // 对角线索引
    int head_idx = blockIdx.y;  // head索引
    if (k >= 2 * n - 1 || head_idx >= head_num) return;
    
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    const float* S_0_head = S_0 + head_idx * n * n;
    
    float local_sum = 0.0f;
    
    // 计算对角线上的元素个数和起始位置
    int diag_offset = k - (n - 1);
    int diag_len = (diag_offset >= 0) ? (n - diag_offset) : (n + diag_offset);
    
    // 遍历该对角线上的元素
    for (int idx = tid; idx < diag_len; idx += blockDim.x) {
        int row, col;
        if (diag_offset >= 0) {
            row = idx;
            col = idx + diag_offset;
        } else {
            row = idx - diag_offset;
            col = idx;
        }
        int linear_idx = row * n + col;
        local_sum += S_0_head[linear_idx];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        results[head_idx * (2 * n - 1) + k] = shared_sum[0];
    }
}

// Kernel 3: 计算S_0与垂直线矩阵D_k的内积，支持batch维度
// 输出：shape (head, n)
__global__ void compute_s0_vert_kernel(
    const float* __restrict__ S_0,
    float* __restrict__ results,
    int head_num,
    int n
) {
    int k = blockIdx.x;  // 列索引
    int head_idx = blockIdx.y;  // head索引
    if (k >= n || head_idx >= head_num) return;
    
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    const float* S_0_head = S_0 + head_idx * n * n;
    
    float local_sum = 0.0f;
    
    // 遍历第k列的所有元素
    for (int row = tid; row < n; row += blockDim.x) {
        int linear_idx = row * n + k;
        local_sum += S_0_head[linear_idx];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        results[head_idx * n + k] = shared_sum[0];
    }
}

// Kernel 4: 计算S_0与块矩阵E的内积，支持batch维度
// 块是均匀分布的，每个块大小为blocks_per_frame
// 输出：shape (head,)
__global__ void compute_s0_block_kernel(
    const float* __restrict__ S_0,
    float* __restrict__ results,
    int head_num,
    int n,
    int blocks_per_frame
) {
    int head_idx = blockIdx.y;
    if (head_idx >= head_num) return;
    
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    const float* S_0_head = S_0 + head_idx * n * n;
    
    float local_sum = 0.0f;
    
    int num_blocks = n / blocks_per_frame;
    
    // 遍历所有块
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int start = block_idx * blocks_per_frame;
        int size = blocks_per_frame;
        
        // 遍历块内的所有元素
        int total_elements = size * size;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int local_row = idx / size;
            int local_col = idx % size;
            int global_row = start + local_row;
            int global_col = start + local_col;
            int linear_idx = global_row * n + global_col;
            local_sum += S_0_head[linear_idx];
        }
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        results[head_idx] = shared_sum[0];
    }
}

// Kernel 5: 计算对角线矩阵C_i与C_j的内积
// 结果：两条对角线重叠的元素个数（与head无关）
__global__ void compute_diag_diag_kernel(
    float* __restrict__ results,
    int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int num_diags = 2 * n - 1;
    if (i >= num_diags || j >= num_diags) return;
    
    // 只计算上三角（包括对角线），利用对称性
    if (i > j) return;
    
    int offset_i = i - (n - 1);
    int offset_j = j - (n - 1);
    
    // 计算两条对角线的重叠长度
    float overlap = 0.0f;
    if (offset_i == offset_j) {
        // 同一条对角线
        int diag_len = (offset_i >= 0) ? (n - offset_i) : (n + offset_i);
        overlap = (float)diag_len;
    }
    
    results[i * num_diags + j] = overlap;
    if (i != j) {
        results[j * num_diags + i] = overlap;
    }
}

// Kernel 6: 计算垂直线矩阵D_i与D_j的内积
__global__ void compute_vert_vert_kernel(
    float* __restrict__ results,
    int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n || j >= n) return;
    
    // 只计算上三角
    if (i > j) return;
    
    float overlap = (i == j) ? (float)n : 0.0f;
    
    results[i * n + j] = overlap;
    if (i != j) {
        results[j * n + i] = overlap;
    }
}

// Kernel 7: 计算对角线矩阵C_i与垂直线矩阵D_j的内积
__global__ void compute_diag_vert_kernel(
    float* __restrict__ results,
    int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 对角线索引
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 垂直线索引
    
    int num_diags = 2 * n - 1;
    if (i >= num_diags || j >= n) return;
    
    int offset = i - (n - 1);
    
    // 对角线与垂直线的交点
    float overlap = 0.0f;
    
    if (offset >= 0) {
        // 主对角线及其上方
        if (j >= offset && j < n) {
            overlap = 1.0f;
        }
    } else {
        // 主对角线下方
        if (j < n + offset) {
            overlap = 1.0f;
        }
    }
    
    results[i * n + j] = overlap;
}

// Kernel 8: 计算对角线矩阵C_i与块矩阵E的内积
// 块是均匀分布的，返回shape (2n-1,)
__global__ void compute_diag_block_kernel(
    float* __restrict__ results,
    int n,
    int blocks_per_frame
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 对角线索引
    
    int num_diags = 2 * n - 1;
    if (i >= num_diags) return;
    
    int offset = i - (n - 1);
    int num_blocks = n / blocks_per_frame;
    
    // 计算对角线与所有块的重叠元素数
    float overlap = 0.0f;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int start = block_idx * blocks_per_frame;
        int size = blocks_per_frame;
        
        // 遍历块内的对角线元素
        for (int k = 0; k < size; k++) {
            int row = start + k;
            int col;
            if (offset >= 0) {
                col = row + offset;
            } else {
                col = row + offset;
            }
            
            // 检查col是否在块内
            if (col >= start && col < start + size) {
                overlap += 1.0f;
            }
        }
    }
    
    results[i] = overlap;
}

// Kernel 9: 计算垂直线矩阵D_i与块矩阵E的内积
// 返回shape (n,)
__global__ void compute_vert_block_kernel(
    float* __restrict__ results,
    int n,
    int blocks_per_frame
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 垂直线索引
    
    if (i >= n) return;
    
    // 检查第i列是否穿过任何块
    float overlap = 0.0f;
    
    int num_blocks = n / blocks_per_frame;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int start = block_idx * blocks_per_frame;
        int size = blocks_per_frame;
        
        if (i >= start && i < start + size) {
            overlap += (float)size;
        }
    }
    
    results[i] = overlap;
}

// 主函数：计算完整的M^T·M矩阵
// 返回(head, total_features, total_features)的矩阵
torch::Tensor compute_mtm_cuda(
    torch::Tensor S_T,  // (head, n, n)矩阵（仅用于获取shape）
    int blocks_per_frame,  // 每个块的大小
    float regularization = 1e-3
) {
    // 已根据 Python 端 compute_mtm_pytorch 修改：移除与 S_0 的交叉项计算，
    // MTM 现在只包含 C（对角线）、D（垂直线）和 E（块）三类特征的内积。
    const int head_num = S_T.size(0);
    const int n = S_T.size(1);
    const int num_diags = 2 * n - 1;
    const int total_features = num_diags + n + 1;  // C (num_diags), D (n), E (1)

    // 分配输出矩阵
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(S_T.device());

    torch::Tensor MTM = torch::zeros({head_num, total_features, total_features}, options);

    // 计算对角线-对角线、垂直-垂直、对角-垂直、对角-块、垂直-块、块-块（与 head 无关或共享）
    // 复用已有 kernel 计算这些项（kernel 已在文件中定义）

    // 5. 计算对角线与对角线的内积（与 head 无关）
    torch::Tensor diag_diag = torch::zeros({num_diags, num_diags}, options);
    dim3 threads_2d(16, 16);
    dim3 blocks_2d((num_diags + 15) / 16, (num_diags + 15) / 16);
    compute_diag_diag_kernel<<<blocks_2d, threads_2d>>>(
        diag_diag.data_ptr<float>(),
        n
    );

    // 6. 垂直线与垂直线
    torch::Tensor vert_vert = torch::zeros({n, n}, options);
    dim3 blocks_2d_vert((n + 15) / 16, (n + 15) / 16);
    compute_vert_vert_kernel<<<blocks_2d_vert, threads_2d>>>(
        vert_vert.data_ptr<float>(),
        n
    );

    // 7. 对角线与垂直线
    torch::Tensor diag_vert = torch::zeros({num_diags, n}, options);
    dim3 blocks_2d_dv((n + 15) / 16, (num_diags + 15) / 16);
    compute_diag_vert_kernel<<<blocks_2d_dv, threads_2d>>>(
        diag_vert.data_ptr<float>(),
        n
    );

    // 8. 对角线与块
    torch::Tensor diag_block = torch::zeros({num_diags}, options);
    int threads = 256;
    int blocks_1d_db = (num_diags + threads - 1) / threads;
    compute_diag_block_kernel<<<blocks_1d_db, threads>>>(
        diag_block.data_ptr<float>(),
        n,
        blocks_per_frame
    );

    // 9. 垂直线与块
    torch::Tensor vert_block = torch::zeros({n}, options);
    int blocks_1d_vb = (n + threads - 1) / threads;
    compute_vert_block_kernel<<<blocks_1d_vb, threads>>>(
        vert_block.data_ptr<float>(),
        n,
        blocks_per_frame
    );

    // 10. 块与块
    int num_blocks = n / blocks_per_frame;
    float block_block_val = (float)(num_blocks * blocks_per_frame * blocks_per_frame);

    cudaDeviceSynchronize();

    // 填充 MTM（所有 head 共享这些结构化值）
    for (int h = 0; h < head_num; h++) {
        // C block
        MTM.index_put_({h, torch::indexing::Slice(0, 0 + num_diags), 
                       torch::indexing::Slice(0, 0 + num_diags)}, diag_diag);
        // D block
        MTM.index_put_({h, torch::indexing::Slice(num_diags, num_diags + n),
                       torch::indexing::Slice(num_diags, num_diags + n)}, vert_vert);
        // C-D cross
        MTM.index_put_({h, torch::indexing::Slice(0, 0 + num_diags),
                       torch::indexing::Slice(num_diags, num_diags + n)}, diag_vert);
        MTM.index_put_({h, torch::indexing::Slice(num_diags, num_diags + n),
                       torch::indexing::Slice(0, 0 + num_diags)}, diag_vert.t());

        // C-E and E-C
        MTM.index_put_({h, torch::indexing::Slice(0, 0 + num_diags), num_diags + n}, diag_block);
        MTM.index_put_({h, num_diags + n, torch::indexing::Slice(0, 0 + num_diags)}, diag_block);

        // D-E and E-D
        MTM.index_put_({h, torch::indexing::Slice(num_diags, num_diags + n), num_diags + n}, vert_block);
        MTM.index_put_({h, num_diags + n, torch::indexing::Slice(num_diags, num_diags + n)}, vert_block);

        // E-E
        MTM[h][num_diags + n][num_diags + n] = block_block_val;
    }

    // 正则化（对角线）
    auto eye = torch::eye(total_features, options);
    for (int h = 0; h < head_num; h++) {
        MTM[h] += eye * regularization;
    }

    return MTM;
}

// ...existing code...
torch::Tensor compute_mts_cuda(
    torch::Tensor S_T,  // (head, n, n)矩阵（当前步的attention map）
    int blocks_per_frame
) {
    // 已根据 Python 端 compute_mts_pytorch 修改：不再计算 S_0^T·S_T 等交叉项。
    const int head_num = S_T.size(0);
    const int n = S_T.size(1);
    const int num_diags = 2 * n - 1;
    const int total_features = num_diags + n + 1; // C, D, E

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(S_T.device());

    torch::Tensor MTS = torch::zeros({head_num, total_features}, options);

    // 展平 S_T 并获取指针供 kernel 使用
    torch::Tensor S_T_flat = S_T.reshape({head_num, n * n});
    const float* S_T_ptr = S_T_flat.data_ptr<float>();

    int threads = 256;

    // 2. C_k^T · S_T（对角线） - per-head
    torch::Tensor temp_diag = torch::zeros({head_num, num_diags}, options);
    dim3 grid_diag(num_diags, head_num);
    compute_s0_diag_kernel<<<grid_diag, threads>>>(
        S_T_ptr,
        temp_diag.data_ptr<float>(),
        head_num,
        n
    );

    // 3. D_k^T · S_T（垂直线）
    torch::Tensor temp_vert = torch::zeros({head_num, n}, options);
    dim3 grid_vert(n, head_num);
    compute_s0_vert_kernel<<<grid_vert, threads>>>(
        S_T_ptr,
        temp_vert.data_ptr<float>(),
        head_num,
        n
    );

    // 4. E^T · S_T（块）
    torch::Tensor temp_block = torch::zeros({head_num}, options);
    dim3 grid_block(1, head_num);
    compute_s0_block_kernel<<<grid_block, threads>>>(
        S_T_ptr,
        temp_block.data_ptr<float>(),
        head_num,
        n,
        blocks_per_frame
    );

    cudaDeviceSynchronize();

    // 填充结果：C (0:num_diags), D (num_diags:num_diags+n), E (num_diags+n)
    MTS.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 0 + num_diags)}, temp_diag);
    MTS.index_put_({torch::indexing::Slice(), torch::indexing::Slice(num_diags, num_diags + n)}, temp_vert);
    MTS.index_put_({torch::indexing::Slice(), num_diags + n}, temp_block);

    return MTS;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mtm", &compute_mtm_cuda, "Compute M^T·M matrix (CUDA)");
    m.def("compute_mts", &compute_mts_cuda, "Compute M^T·S_T vector (CUDA)");
}

