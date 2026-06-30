"""
GPU加速的最小二乘求解器，用于attention map特征提取

该模块提供CUDA加速的M^T·M和M^T·S_T计算，避免显式构造大矩阵
"""

import torch
import os
import threading
from torch.utils.cpp_extension import load
_MTM = None
_MTM_lock = threading.Lock()

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
    """
    PyTorch实现的M^T·M计算（作为fallback）
    
    参数:
        S_T: 第t步的attention map, shape (head, n, n)
        regularization: 正则化系数
    
    返回:
        MTM: M^T·M矩阵, shape (total_features, total_features)
    """
    head_num = S_T.shape[0]
    n = S_T.shape[1]
    num_diags = 2 * n - 1
    total_features = 1 + num_diags + n
    
    device = S_T.device
    dtype = S_T.dtype
    
    MTM = torch.zeros(head_num, total_features, total_features, device=device, dtype=dtype)
    
    
    # 5. 对角线C_i与C_j的内积
    for i in range(num_diags):
        offset_i = i - (n - 1)
        for j in range(i, num_diags):
            offset_j = j - (n - 1)
            if offset_i == offset_j:
                # 同一条对角线
                diag_len = n - abs(offset_i)
                MTM[:, i, j] = float(diag_len)
                if i != j:
                    MTM[:, j, i] = float(diag_len)
    
    # 6. 垂直线D_i与D_j的内积
    for i in range(n):
        for j in range(i, n):
            val = float(n) if i == j else 0.0
            MTM[:, num_diags + i, num_diags + j] = val
            if i != j:
                MTM[:, num_diags + j, num_diags + i] = val

    # 7. 对角线C_i与垂直线D_j的内积
    for i in range(num_diags):
        offset = i - (n - 1)
        for j in range(n):
            # 对角线i与垂直线j的交点
            overlap = 0.0
            if offset >= 0:
                # 主对角线及上方
                if j >= offset and j < n:
                    overlap = 1.0
            else:
                # 主对角线下方
                if j < n + offset:
                    overlap = 1.0
            MTM[:, i, num_diags + j] = overlap
            MTM[:, num_diags + j, i] = overlap

    # 8. 块E_i与E_j的内积
    num_blocks = n // blocks_per_frame
    block_starts = torch.arange(0, n, blocks_per_frame, device=device)
    total = torch.zeros(head_num, device=device, dtype=dtype)
    MTM[:, num_diags + n, num_diags + n] = num_blocks * (blocks_per_frame ** 2)

    # 9. 对角线C_i与块E_j的内积
    for i in range(num_diags):
        offset = i - (n - 1)
        overlap = torch.zeros(head_num, device=device, dtype=dtype)
        for j in range(num_blocks):
            start = block_starts[j].item()
            # 计算对角线与块的重叠
            for k in range(blocks_per_frame):
                row = start + k
                if offset >= 0:
                    col = row + offset
                else:
                    col = row + offset  # offset是负数
                if col >= start and col < start + blocks_per_frame:
                    overlap += 1.0
        MTM[:, i, num_diags + n] = overlap
        MTM[:, num_diags + n, i] = overlap
    
    # 10. 垂直线D_i与块E_j的内积
    for i in range(n):
        MTM[:, num_diags + i, num_diags + n] = blocks_per_frame
        MTM[:, num_diags + n, num_diags + i] = blocks_per_frame

    # 添加正则化（对角线）以确保数值稳定性
    MTM += torch.eye(total_features, device=device, dtype=dtype).unsqueeze(0) * regularization
    
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

def solve_lstsq(warmup_state, S_T, step, blocks_per_frame, layer_idx=0, regularization=1e-5, use_cuda=True):
    """
    求解最小二乘问题: min ||S_T - MX||^2
    理论解: X = (M^T·M)^{-1} · M^T·S_T
    
    参数:
        S_T: 当前步的attention map, shape (n, n)
        block_starts: 块起始索引, shape (num_blocks,) 或 None
        block_sizes: 块大小, shape (num_blocks,) 或 None
        regularization: 正则化系数，用于数值稳定性（默认改为1e-4）
        use_cuda: 是否使用CUDA加速
    
    返回:
        特征字典 {
            'c': Tensor，对角线亮度值，shape (2n-1,)
            'd': Tensor，垂直线亮度值，shape (n,)
            'e': Tensor，块亮度值，shape (num_blocks,)
        }
    """
    num_heads = S_T.shape[0]
    n = S_T.shape[1]
    S_T = S_T.to(torch.float32)
    num_diags = 2 * n - 1
    
    # 尝试使用CUDA加速
    cuda_ext = _load_cuda_extension() if use_cuda else None
    global _MTM
    if _MTM is None:
        with _MTM_lock:
            if _MTM is None:
                try:
                # 使用CUDA实现
                    _MTM = cuda_ext.compute_mtm(S_T, blocks_per_frame, regularization)
                except Exception as e:
                    print(f"CUDA execution failed: {e}, falling back to PyTorch")
                    _MTM = compute_mtm_pytorch(S_T, blocks_per_frame, regularization)
    MTM = _MTM
    try:
        # 使用CUDA实现
        MTS = cuda_ext.compute_mts(S_T, blocks_per_frame)
    except Exception as e:
        print(f"CUDA execution failed: {e}, falling back to PyTorch")
        MTS = compute_mts_pytorch(S_T, blocks_per_frame)

    if step > warmup_state['warmup_steps'] - 2:
        # 求解线性系统 MTM · X = MTS
        
        # 安全的逐个头求解，避免数值稳定性问题
        X_list = []
        for head_idx in range(MTM.shape[0]):
            try:
                MTM_head = MTM[head_idx]  # 形状: [total_features, total_features]
                
                # 确保MTS_head形状正确
                if MTS.dim() == 2 and MTS.shape[0] == MTM.shape[0]:
                    MTS_head = MTS[head_idx].unsqueeze(1)  # 形状: [total_features, 1]
                else:
                    MTS_head = MTS.unsqueeze(1) if MTS.dim() == 2 else MTS
                    if MTS_head.shape[0] == MTM.shape[0]:
                        MTS_head = MTS_head[head_idx]
                    else:
                        MTS_head = MTS_head[0]  # 如果MTS只有一个头，使用第一个
                
                # 检查形状匹配
                if MTM_head.shape[0] != MTS_head.shape[0]:
                    # 调整到最小维度
                    min_dim = min(MTM_head.shape[0], MTS_head.shape[0])
                    MTM_head = MTM_head[:min_dim, :min_dim]
                    MTS_head = MTS_head[:min_dim].unsqueeze(1)
                
                # 使用稳定的求解方法
                try:
                    # 首先尝试Cholesky分解（最稳定）
                    L = torch.linalg.cholesky(MTM_head)
                    X_head = torch.cholesky_solve(MTS_head, L)
                except RuntimeError:
                    # Cholesky失败，使用LU分解
                    try:
                        X_head = torch.linalg.solve(MTM_head, MTS_head)
                    except RuntimeError:
                        # 如果都失败，使用伪逆
                        MTM_pinv = torch.linalg.pinv(MTM_head)
                        X_head = MTM_pinv @ MTS_head
                
                X_list.append(X_head.squeeze(-1))
                
            except Exception as e:
                print(f"头{head_idx}求解失败: {e}")
                # 返回零向量作为fallback
                X_head = torch.zeros(MTM_head.shape[0], device=MTM_head.device, dtype=MTM_head.dtype)
                X_list.append(X_head)
        
        X = torch.stack(X_list)
        
        # 安全的解包结果
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
