"""
GPU加速的最小二乘求解器，用于attention map特征提取

该模块提供CUDA加速的M^T·M和M^T·S_T计算，避免显式构造大矩阵
"""

import torch
import os
import threading
from torch.utils.cpp_extension import load
_FACTOR_CACHE = {}
_FACTOR_CACHE_LOCK = threading.Lock()

# 尝试加载编译好的CUDA扩展
_cuda_lstsq = None

def _load_cuda_extension():
    """延迟加载CUDA扩展"""
    global _cuda_lstsq
    if _cuda_lstsq is not None:
        return _cuda_lstsq
    
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_file = os.path.join(current_dir, 'cuda_lstsq_kernel.cu')
        
        # 使用JIT编译
        _cuda_lstsq = load(
            name='cuda_lstsq',
            sources=[cuda_file],
            extra_cuda_cflags=['-O3', '--use_fast_math', '--extended-lambda'],
            verbose=True
        )
        print("CUDA extension loaded successfully")
        return _cuda_lstsq
    except Exception as e:
        print(f"Warning: Failed to load CUDA extension: {e}")
        print("Falling back to PyTorch implementation")
        return None

def compute_mtm_pytorch(S_T, blocks_per_frame, regularization=1e-3):
    """构造所有 attention head 共享的单份 M^T·M 矩阵。"""
    n = S_T.shape[1]
    num_diags = 2 * n - 1
    total_features = num_diags + n + 1
    device = S_T.device
    dtype = S_T.dtype

    MTM = torch.zeros(total_features, total_features, device=device, dtype=dtype)

    offsets = torch.arange(-(n - 1), n, device=device)
    diag_indices = torch.arange(num_diags, device=device)
    MTM[diag_indices, diag_indices] = (n - offsets.abs()).to(dtype)

    vert_indices = num_diags + torch.arange(n, device=device)
    MTM[vert_indices, vert_indices] = float(n)

    columns = torch.arange(n, device=device).unsqueeze(0)
    offset_grid = offsets.unsqueeze(1)
    diag_vert = torch.where(
        offset_grid >= 0,
        columns >= offset_grid,
        columns < n + offset_grid,
    ).to(dtype)
    MTM[:num_diags, num_diags:num_diags + n] = diag_vert
    MTM[num_diags:num_diags + n, :num_diags] = diag_vert.transpose(0, 1)

    num_blocks = n // blocks_per_frame
    covered_columns = num_blocks * blocks_per_frame
    diag_block = torch.where(
        offsets.abs() < blocks_per_frame,
        num_blocks * (blocks_per_frame - offsets.abs()),
        torch.zeros_like(offsets),
    ).to(dtype)
    MTM[:num_diags, -1] = diag_block
    MTM[-1, :num_diags] = diag_block

    vert_block = torch.where(
        torch.arange(n, device=device) < covered_columns,
        blocks_per_frame,
        0,
    ).to(dtype)
    MTM[num_diags:num_diags + n, -1] = vert_block
    MTM[-1, num_diags:num_diags + n] = vert_block
    MTM[-1, -1] = float(num_blocks * blocks_per_frame * blocks_per_frame)

    MTM += torch.eye(total_features, device=device, dtype=dtype) * regularization
    return MTM

def compute_mts_pytorch(S_T, blocks_per_frame):
    """
    PyTorch实现的M^T·S_T计算（作为fallback）
    
    参数:
        S_T: 当前步的attention map, shape (n, n)
        block_starts: 块起始索引, shape (num_blocks,)
        block_sizes: 块大小, shape (num_blocks,)
    
    返回:
        MTS: M^T·S_T向量, shape (total_features,)
    """
    n = S_T.shape[1]
    num_diags = 2 * n - 1
    total_features = 1 + num_diags + n
    num_heads = S_T.shape[0]
    
    device = S_T.device
    dtype = S_T.dtype
    
    MTS = torch.zeros(num_heads, total_features, device=device, dtype=dtype)
    
    
    # 2. C_k^T · S_T（对角线）
    for k in range(num_diags):
        offset = k - (n - 1)
        total = torch.zeros(num_heads, device=device, dtype=dtype)
        if offset >= 0:
            for i in range(n - offset):
                total += S_T[:, i, i + offset]
        else:
            for i in range(n + offset):
                total += S_T[:, i - offset, i]
        MTS[:, k] = total
    
    # 3. D_k^T · S_T（垂直线）
    for k in range(n):
        MTS[:, num_diags + k] = S_T[:, :, k].sum(dim=1)

    # 4. E_k^T · S_T（块）
    num_blocks = n // blocks_per_frame
    block_starts = torch.arange(0, n, blocks_per_frame, device=device)
    total = torch.zeros(num_heads, device=device, dtype=dtype)
    for idx in range(num_blocks):
        start = block_starts[idx].item()
        total += S_T[:, start:start+blocks_per_frame, start:start+blocks_per_frame].sum(dim=(1, 2))
    MTS[:, num_diags + n] = total

    return MTS

def _factor_cache_key(S_T, blocks_per_frame, regularization):
    device = S_T.device
    return (
        device.type,
        device.index,
        S_T.shape[1],
        int(blocks_per_frame),
        float(regularization),
        S_T.dtype,
    )


def _get_cached_factor(S_T, blocks_per_frame, regularization, cuda_ext):
    """按矩阵结构缓存一次分解；所有 head 和后续调用共享。"""
    key = _factor_cache_key(S_T, blocks_per_frame, regularization)
    factor = _FACTOR_CACHE.get(key)
    if factor is not None:
        return factor

    with _FACTOR_CACHE_LOCK:
        factor = _FACTOR_CACHE.get(key)
        if factor is not None:
            return factor

        try:
            if cuda_ext is None:
                raise RuntimeError("CUDA extension is unavailable")
            MTM = cuda_ext.compute_mtm(S_T, blocks_per_frame, regularization)
        except Exception as e:
            print(f"CUDA MTM computation failed: {e}, falling back to PyTorch")
            MTM = compute_mtm_pytorch(S_T, blocks_per_frame, regularization)

        if MTM.dim() != 2 or MTM.shape[0] != MTM.shape[1]:
            raise RuntimeError(f"Expected a square 2D MTM matrix, got shape {tuple(MTM.shape)}")

        try:
            factor = ("cholesky", torch.linalg.cholesky(MTM))
        except RuntimeError:
            try:
                lu, pivots = torch.linalg.lu_factor(MTM)
                factor = ("lu", lu, pivots)
            except RuntimeError:
                factor = ("pinv", torch.linalg.pinv(MTM))

        _FACTOR_CACHE[key] = factor
        return factor


def _solve_all_heads(factor, MTS):
    """把所有 head 作为多个 RHS，一次求解共享系数矩阵。"""
    if MTS.dim() != 2:
        raise RuntimeError(f"Expected MTS with shape [head, feature], got {tuple(MTS.shape)}")

    rhs = MTS.transpose(0, 1).contiguous()
    method = factor[0]
    matrix_size = factor[1].shape[-1]
    if rhs.shape[0] != matrix_size:
        raise RuntimeError(
            f"MTS feature dimension {rhs.shape[0]} does not match factor size {matrix_size}"
        )

    if method == "cholesky":
        solution = torch.cholesky_solve(rhs, factor[1])
    elif method == "lu":
        solution = torch.linalg.lu_solve(factor[1], factor[2], rhs)
    else:
        solution = factor[1] @ rhs
    return solution.transpose(0, 1).contiguous()


def solve_lstsq(warmup_state, S_T, step, blocks_per_frame, layer_idx=0, regularization=1e-5, use_cuda=True):
    """
    求解最小二乘问题: min ||S_T - MX||^2
    理论解: X = (M^T·M)^{-1} · M^T·S_T
    
    参数:
        S_T: 当前步的attention map, shape (head, n, n)
        blocks_per_frame: 每个帧包含的attention块数量
        regularization: 正则化系数，用于数值稳定性
        use_cuda: 是否使用CUDA加速
    
    返回:
        特征字典 {
            'c': Tensor，垂直线亮度值，shape (head, n)
            'd': Tensor，对角线亮度值，shape (head, 2n-1)
            'b_d': Tensor，块对角亮度值，shape (head,)
        }
    """
    num_heads = S_T.shape[0]
    n = S_T.shape[1]
    S_T = S_T.to(torch.float32)
    num_diags = 2 * n - 1
    
    # 尝试使用CUDA加速
    cuda_ext = _load_cuda_extension() if use_cuda else None

    try:
        if cuda_ext is None:
            raise RuntimeError("CUDA extension is unavailable")
        MTS = cuda_ext.compute_mts(S_T, blocks_per_frame)
    except Exception as e:
        print(f"CUDA MTS computation failed: {e}, falling back to PyTorch")
        MTS = compute_mts_pytorch(S_T, blocks_per_frame)

    if step > warmup_state['warmup_steps'] - 2:
        factor = _get_cached_factor(S_T, blocks_per_frame, regularization, cuda_ext)
        X = _solve_all_heads(factor, MTS)

        total_features = X.shape[1]
        
        # 安全的索引范围检查
        idx_d_start = 0
        idx_d_end = min(0 + num_diags, total_features)
        idx_c_start = idx_d_end
        idx_c_end = min(idx_d_end + n, total_features)

        features = {
            'd': X[:, idx_d_start:idx_d_end],  # (head, min(num_diags, available))
            'c': X[:, idx_c_start:idx_c_end] / 2 ,  # (head, min(n, available))
            'b_d': X[:, -1]  # (head,)
        }

        
        return features
    else:
        return{
            'd': torch.zeros((num_heads,2*n-1), device=S_T.device, dtype=S_T.dtype),
            'c': torch.zeros((num_heads,n), device=S_T.device, dtype=S_T.dtype),
            'b_d': torch.zeros((num_heads), device=S_T.device, dtype=S_T.dtype)
        }
def reconstruct_attention_map(features, n, blocks_per_frame=4):
    """
    根据提取的特征重建attention map
    
    参数:
        features: dict，包含以下特征
            - d: 对角线特征，形状 (head, 2*n-1)
            - c: 垂直线特征，形状 (head, n)
            - b_d: 块对角线特征，形状 (head,)
        n: 序列长度
        blocks_per_frame: 块大小
    
    返回:
        reconstructed: 重建的attention map，形状 (head, n, n)
    """
    num_heads = features['d'].shape[0]
    device = features['d'].device
    dtype = features['d'].dtype
    
    # 初始化重建的attention map
    reconstructed = torch.zeros(num_heads, n, n, device=device, dtype=dtype)
    
    # 1. 添加对角线特征
    # d 对应 2*n-1 条对角线（从左上到右下的所有对角线）
    for k in range(2*n-1):
        offset = k - (n-1)  # 对角线偏移量，范围 [-(n-1), n-1]
        
        # 获取这条对角线的特征值
        diag_values = features['d'][:, k]  # 形状 (head,)
        
        # 遍历每个head
        for h in range(num_heads):
            value = diag_values[h]
            
            # 根据偏移量确定对角线的起始位置和长度
            if offset >= 0:
                # 主对角线及上方对角线
                start_row = 0
                start_col = offset
                length = n - offset
            else:
                # 主对角线下方对角线
                start_row = -offset
                start_col = 0
                length = n + offset
            
            # 将对角线特征值加到对应位置
            for i in range(length):
                row = start_row + i
                col = start_col + i
                reconstructed[h, row, col] += value
    
    # 2. 添加垂直线特征
    # c 对应 n 条垂直线（每一列）
    for col in range(n):
        # 获取这一列的特征值
        col_values = 2 * features['c'][:, col]  # 形状 (head,)
        
        # 遍历每个head
        for h in range(num_heads):
            value = col_values[h]
            
            # 将垂直线特征值加到这一列的所有行上
            reconstructed[h, :, col] += value
    
    # 3. 添加块对角线特征
    # b_d 对应对角线上的块
    num_blocks = n // blocks_per_frame
    
    for h in range(num_heads):
        value = features['b_d'][h]
        
        # 对每个块，在对角线位置加上特征值
        for block_idx in range(num_blocks):
            start = block_idx * blocks_per_frame
            end = start + blocks_per_frame
            
            # 块内的所有位置都加上特征值（不仅仅是块的对角线）
            # 这是基于原始代码中对块的定义：块内所有位置共享同一个亮度
            for i in range(blocks_per_frame):
                for j in range(blocks_per_frame):
                    row = start + i
                    col = start + j
                    if row < n and col < n:
                        reconstructed[h, row, col] += value
    
    return reconstructed
