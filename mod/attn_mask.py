"""
自适应Attention Mask生成算法 - 函数式实现

该模块实现基于Warm up阶段学习的自适应attention mask生成，包括：
1. Warm up阶段：提取attention map的结构特征（对角线、垂直线、分块对角）
2. 预测阶段：基于学习到的特征动态生成sparse mask
3. Backend加速：支持 FlashInfer加速方式
"""

import torch
import time
import logging
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from einops import rearrange, repeat
from sageattention import sageattn
from spas_sage_attn import block_sparse_sage2_attn_cuda, spas_sage2_attn_meansim_topk_cuda

from MoD.mod.get_radial_mask import get_radial_mask
from MoD.mod.get_SVG_mask import get_svg_mask
from MoD.mod.state import is_warmup_complete

from MoD.mod.triton_flash_sparsity import flash_attn_with_sparsity_map

try:
    from MoD.mod.cuda_lstsq import solve_lstsq
except ImportError:
    print("Warning: cuda_lstsq not available, using PyTorch fallback")
    solve_lstsq = None


# ============================================================================
# Global Workspace for FlashInfer (to avoid per-layer allocation)
# ============================================================================
_GLOBAL_FLASHINFER_WORKSPACE: Optional[torch.Tensor] = None
# ============================================================================

# ============================================================================
# Utility Functions from attn_mask.py
# ============================================================================
def reset_peak_gpu_stats(device: Optional[torch.device] = None):
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(device.index)

def get_cuda_arch_versions():
    """获取CUDA架构版本"""
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

def print_peak_gpu_stats(prefix: str = "", device: Optional[torch.device] = None):
    if not torch.cuda.is_available():
        print(prefix + " GPU not available")
        return
    # 确保所有 kernel 完成
    torch.cuda.synchronize()
    if device is None:
        alloc = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
    else:
        alloc = torch.cuda.max_memory_allocated(device.index)
        reserved = torch.cuda.max_memory_reserved(device.index)
    def to_gib(x): return x / (1024**3)
    print(f"{prefix} peak_alloc={to_gib(alloc):.3f} GiB, peak_reserved={to_gib(reserved):.3f} GiB")


# ============================================================================
# Feature Extraction Functions (from AttentionMapDecomposer)
# ============================================================================

def extract_attention_features(
    warmup_state: Dict,
    S_T: torch.Tensor,
    blocks_per_frame: int,
    layer_idx: int = 0,
    regularization: float = 1,
    use_cuda: bool = True
) -> Dict[str, torch.Tensor]:
    """
    从attention map中提取特征
    
    参数:
        S_T: 当前步的attention map, shape (n, n)
        block_configs: 块配置列表，每个元素为(start_idx, size)
        regularization: 正则化系数
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
    assert S_T.shape == (num_heads, n, n), f"S_T shape mismatch: {S_T.shape} vs ({num_heads}, {n}, {n})"


    # 调用GPU加速的最小二乘求解（如果可用）
    features = solve_lstsq(
        warmup_state=warmup_state,
        S_T=S_T,
        step=warmup_state['current_steps'],
        layer_idx=layer_idx,
        blocks_per_frame=blocks_per_frame,
        regularization=regularization,
        use_cuda=use_cuda and S_T.is_cuda
    )
    return features

def update_warmup(
    warmup_state: Dict,
    step: int,
    block_sparse_map: torch.Tensor,
):
    """
    更新warmup阶段的数据
    
    参数:
        warmup_state: warmup状态字典
        head_idx: head索引
        step: 当前步数

    """
    if step == warmup_state['warmup_steps'] - 2:
        warmup_state['features_history'] = []
        warmup_state['current_steps'] = step
        warmup_state['prev_full_maps'] = block_sparse_map
    # 提取特征
    S_T = block_sparse_map
    features = extract_attention_features(
        warmup_state=warmup_state,
        S_T=S_T,
        blocks_per_frame=warmup_state['blocks_per_frame'],
        use_cuda=warmup_state['use_cuda']
    )
    # 存储特征
    warmup_state['features_history'].append(features)
    warmup_state['current_steps'] = step + 1
    warmup_state['prev_full_maps'] = block_sparse_map

# ============================================================================
# Attention Computation Functions
# ============================================================================
@torch.inference_mode()
def _compute_block_sparse_map(
    query: torch.Tensor,   # (B,H,S,D)
    key: torch.Tensor,     # (B,H,S,D)
    value: torch.Tensor,   # (B,H,S,D)
    *,
    block_size: int,
    video_token_num: int,
    text_token_num: int,
    threshold: float,
    model_type: str,       # "cogvideox" or "hunyuan"
    q_chunk_blocks: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = query.shape
    assert S % block_size == 0
    block_num = S // block_size
    video_block_num = video_token_num // block_size
    scale = D ** -0.5

    output = torch.empty_like(query)
    block_sparse_map = torch.empty(
        (B, H, video_block_num, video_block_num),
        device=query.device,
        dtype=torch.float16
    )

    key_t = key.transpose(-2, -1).contiguous()

    # 分支移出循环
    if model_type == "cogvideox":
        q_video_block_start = text_token_num // block_size
        q_video_block_end = q_video_block_start + video_block_num
        k_video_start = text_token_num
        k_video_end = text_token_num + video_block_num * block_size
    elif model_type == "hunyuan":
        q_video_block_start = 0
        q_video_block_end = video_block_num
        k_video_start = 0
        k_video_end = video_block_num * block_size
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    demo_inv = 1.0 / (block_size * block_size)

    # 按 chunk 处理 query blocks
    for qb in range(0, block_num, q_chunk_blocks):
        qe = min(block_num, qb + q_chunk_blocks)
        s = qb * block_size
        e = qe * block_size

        q_chunk = query[:, :, s:e, :]                               # (B,H,Qs,D)
        scores = torch.matmul(q_chunk, key_t) * scale               # (B,H,Qs,S)
        probs = torch.softmax(scores, dim=-1)                       # (B,H,Qs,S)
        output[:, :, s:e, :] = torch.matmul(probs, value)           # (B,H,Qs,D)

        # 只对 video query blocks 计算 sparse map
        inter_beg = max(qb, q_video_block_start)
        inter_end = min(qe, q_video_block_end)
        if inter_beg < inter_end:
            loc_s = (inter_beg - qb) * block_size
            loc_e = (inter_end - qb) * block_size

            p_video = probs[:, :, loc_s:loc_e, k_video_start:k_video_end]
            # (B,H,Qv*bs, V*bs) -> (B,H,Qv,bs,V,bs)
            qv_blocks = inter_end - inter_beg
            p_video = p_video.view(B, H, qv_blocks, block_size, video_block_num, block_size)

            # 比 int32 sum 再转换更省事：直接 float mean
            sparsity = p_video.lt(threshold).float().mean(dim=(3, 5))  # (B,H,Qv,V)
            block_sparse_map[:, :, inter_beg - q_video_block_start:inter_end - q_video_block_start, :] = sparsity.to(torch.float16)

    return output, block_sparse_map

# _compute_block_sparse_map = torch.compile(_compute_block_sparse_map, mode="max-autotune", dynamic=False)
# Disabled: torch.compile conflicts with @torch.inference_mode() - "Cannot set version_counter for inference tensor"

def compute_block_sparse_map(query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            warmup_state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算block sparse attention map

    参数:
        query: query tensor, shape (batch_size, head_num, seq_len, head_dim)
        key: key tensor, shape (batch_size, head_num, seq_len, head_dim)
        value: value tensor, shape (batch_size, head_num, seq_len, head_dim)
        warmup_state: warmup状态字典

    返回:
        output: output tensor, shape (batch_size, head_num, seq_len, head_dim)
        block_sparse_map, shape (batch_size, head_num, video_block_num, video_block_num)
    """
    output, block_sparse_map = _compute_block_sparse_map(query, key, value, 
                                                         block_size=warmup_state['block_size'],
                                                         video_token_num=warmup_state['video_token_num'],
                                                         text_token_num=warmup_state['text_token_num'],
                                                         threshold=warmup_state['threshold'],
                                                         model_type=warmup_state['model_type'])
    return output, block_sparse_map

def compute_block_sparse_map_triton(query, key, value, warmup_state):
    """
    使用Triton进行加速实现
    """
    return flash_attn_with_sparsity_map(
        query, key, value,
        block_size=warmup_state['block_size'],
        video_token_num=warmup_state['video_token_num'],
        text_token_num=warmup_state['text_token_num'],
        threshold=warmup_state['threshold'],
        model_type=warmup_state['model_type']
        )

import flashinfer
def AttentionSparseEngine(query, key, value, mask, pre_defined_mask=None,video_token_num=0,block_size=128):
    batch_size = query.shape[0]
    if mask.all():
        # dense case - sageattn needs [batch, seq, heads, dim], just slice and call
        kv_border = pre_defined_mask[0].sum() if pre_defined_mask is not None else key.shape[2]
        output_video = sageattn(
            query[:, :, :video_token_num, :],
            key[:, :, :kv_border, :],
            value[:, :, :kv_border, :],
            tensor_layout="HND",
        )

        if pre_defined_mask is not None:
            # flashinfer needs (seq, heads, dim), reshape only here
            q_flashinfer = rearrange(query[:, :, video_token_num:, :], "b h s d -> (b s) h d")
            k_flashinfer = rearrange(key[:, :, :pre_defined_mask[0].sum(), :], "b h s d -> (b s) h d")
            v_flashinfer = rearrange(value[:, :, :pre_defined_mask[0].sum(), :], "b h s d -> (b s) h d")
            output_text = flashinfer.single_prefill_with_kv_cache(
                q=q_flashinfer,
                k=k_flashinfer,
                v=v_flashinfer,
                causal=False,
                return_lse=False,
            )
            # Reshape back and concatenate
            output_text = rearrange(output_text, "(b s) h d -> b s (h d)", b=batch_size)
            output_video_flat = output_video.flatten(2, 3)
            return torch.cat([output_video_flat, output_text], dim=1)
        else:
            return output_video
    # sparse case - block_sparse_sage2_attn_cuda needs (b, h, s, d), reshape only here
    if block_size == 128:
        converted_mask = torch.repeat_interleave(mask, 2, dim=-1).unsqueeze(0)
    elif block_size == 64:
        num_head, num_row, num_col = mask.shape
        reshaped_mask = mask.view(num_head, num_row // 2, 2, num_col)
        converted_mask = torch.max(reshaped_mask, dim=2).values
        converted_mask = converted_mask.unsqueeze(0)
    converted_mask.to(torch.int8)
    if pre_defined_mask is None:
        output = block_sparse_sage2_attn_cuda(query, key, value, mask_id=converted_mask.contiguous(),tensor_layout="HND")

        return output

    kv_border = (pre_defined_mask[0].sum() + 63) // 64
    converted_mask[:, :, :, kv_border:] = False
    output_video = block_sparse_sage2_attn_cuda(
        query[:, :, :video_token_num, :],
        key,
        value,
        mask_id=converted_mask[:, :, :video_token_num // 128, :].contiguous(),
    )

    # flashinfer needs (seq, heads, dim), reshape from [batch, seq, heads, dim]
    q_flashinfer = rearrange(query[:, :, video_token_num:, :], "b h s d -> (b s) h d")
    k_flashinfer = rearrange(key[:, :, :pre_defined_mask[0].sum(), :], "b h s d -> (b s) h d")
    v_flashinfer = rearrange(value[:, :, :pre_defined_mask[0].sum(), :], "b h s d -> (b s) h d")
    output_text = flashinfer.single_prefill_with_kv_cache(
        q=q_flashinfer,
        k=k_flashinfer,
        v=v_flashinfer,
        causal=False,
        return_lse=False,
    )
    output_text = rearrange(output_text, "(b s) h d -> b h s d", b=batch_size)
    return torch.cat([output_video, output_text], dim=2)
# ============================================================================
# Mask Prediction Functions
# ============================================================================

def check_bd_values(values: torch.Tensor, mode: List[str], threshold: float = -7.5e-2) -> bool:
    """
    检查block diagonal的值是否满足阈值条件

    参数:
        values: block diagonal值序列, shape (num_heads, warmup_steps)
        threshold: 阈值
        mode: p值模式
    """
    for i in range(values.shape[0]):
        if mode[i] == 'sparse':
            if values[i, -1] <= threshold:
                mode[i] = 'block_diagonal'
    return mode

def choose_topk_lines(history: List[Dict[str, torch.Tensor]], top_k: int = 10, predict_T:int = None) -> torch.Tensor:
    last_features = history[-2]
    current_features = history[-1]
    last_c_values = last_features['c']
    current_c_values = current_features['c']
    last_d_values = last_features['d']
    current_d_values = current_features['d']
    predict_c_values_k = (current_c_values - last_c_values) / predict_T
    predict_d_values_k = (current_d_values - last_d_values) / predict_T
    predict_c_values = current_c_values + predict_c_values_k
    predict_d_values = current_d_values + predict_d_values_k
    predict_c_d_values = torch.cat([predict_c_values, predict_d_values], dim=1)
    top_k_values, selected_lines = torch.topk(
        predict_c_d_values,
        k=min(top_k, predict_c_d_values.shape[1]),
        dim=1,
        largest=False,
        sorted=False
    )
    return selected_lines
    

def predict_mask_from_warmup(
    warmup_state: Dict,
    n: int,
    top_k: int = 10,
    layer_idx: int = 0,
) -> Dict:
    """
    基于warmup历史预测mask
    
    参数:
        warmup_state: warmup状态
        head_idx: head索引
        n: attention map尺寸
        top_k: 选择最亮的K条线
        block_configs: 块配置
    
    返回:
        mask信息字典
    """

    num_heads = warmup_state['num_heads']
    blocks_per_frame = warmup_state['blocks_per_frame']
    if not is_warmup_complete(warmup_state):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        return {
            'mode': ['full'] * num_heads,
            'block_diagonal': [False] * num_heads,
            'selected_lines': [],
            'mask': torch.ones(num_heads, n, n, dtype=torch.bool, device=device)
        }
    
    # 获取特征历史
    history = warmup_state['features_history']
    device = history[0]['c'].device
    # 检查是否有分块对角模式
    mode = ['sparse'] * num_heads
    b_d_values = torch.stack([f['b_d'] for f in history],dim=0).to(device).permute(1, 0)
    mode = check_bd_values(b_d_values, threshold=1, mode=mode)

    # if layer_idx % 3 == 0:
    #     import os
    #     if not os.path.exists(f"results/layer_{layer_idx}"):
    #         os.makedirs(f"results/layer_{layer_idx}")
    #     with open(f"results/layer_{layer_idx}/c_features.txt", "w") as f:
    #         import json
    #         for t in range(len(history)):
    #             c_values = history[t]['c'].cpu().numpy().tolist()
    #             f.write(json.dumps({'step': t, 'c_values': c_values}) + "\n")
    #     with open(f"results/layer_{layer_idx}/d_features.txt", "w") as f:
    #         import json
    #         for t in range(len(history)):
    #             d_values = history[t]['d'].cpu().numpy().tolist()
    #             f.write(json.dumps({'step': t, 'd_values': d_values}) + "\n")

    # 选择最暗的K条线
    selected_lines = choose_topk_lines(history, top_k=top_k, predict_T=warmup_state.get('predict_T', None))

    mask = generate_mask_from_lines(warmup_state, num_heads, n, blocks_per_frame, selected_lines, mode, device)
    
    return {
        'mode': mode,
        'selected_lines': selected_lines,
        'mask': mask
    }



def generate_mask_from_lines(
    warmup_state: Dict,
    num_heads: int,
    n: int,
    blocks_per_frame: int,
    selected_lines: torch.Tensor,
    mode: List[str],
    device: torch.device
) -> torch.Tensor:
    """
    从选中的线条生成mask
    
    参数:
        n: mask尺寸
        selected_lines: 选中的线条列表
        mode: 每个head的模式列表
        device: 设备
    
    返回:
        boolean mask
    """
    block_size = warmup_state['block_size']
    block_num = n // block_size
    text_token_num = warmup_state['text_token_num']
    text_block_num = text_token_num // block_size
    video_block_num = warmup_state['video_token_num'] // block_size

    # 结果 mask（初始全 True 如原实现）
    mask = torch.ones(num_heads, block_num, block_num, dtype=torch.bool, device=device)

    # video_mask 要在 video 区域填充
    video_mask = torch.zeros(num_heads, video_block_num, video_block_num, dtype=torch.bool, device=device)

    # 规范 selected_lines 为 tensor (num_heads, k)
    if not torch.is_tensor(selected_lines):
        print(len(selected_lines))
        print(selected_lines[0].shape)
        selected_lines = torch.as_tensor(selected_lines, device=device)
    selected_lines = selected_lines.to(device=device)
    if selected_lines.dim() == 1:
        selected_lines = selected_lines.unsqueeze(1)
    selected_lines = selected_lines.long()  # (num_heads, k)

    # head 模式布尔向量
    mode_list = list(mode)
    full_heads = torch.tensor([m == 'full' for m in mode_list], device=device, dtype=torch.bool)
    block_diag_heads = torch.tensor([m == 'block_diagonal' for m in mode_list], device=device, dtype=torch.bool)

    # 直接对 full heads 设置 True
    if full_heads.any():
        video_mask[full_heads] = True

    # 预计算行列索引网格用于对角线 / block diag 判定
    if video_block_num > 0:
        row_idx = torch.arange(video_block_num, device=device).view(1, video_block_num, 1)   # (1, R, 1)
        col_idx = torch.arange(video_block_num, device=device).view(1, 1, video_block_num)   # (1, 1, R)
        diff = col_idx - row_idx  # (1, R, R)  col - row

        # 处理每个选中线条（top_k 通常较小，循环 top_k 但内部全张量化）
        k = selected_lines.size(1)
        for t in range(k):
            idx_k = selected_lines[:, t]  # (num_heads,)

            # 跳过已为 full 的 heads（已全 True）
            active_mask = ~full_heads
            if not active_mask.any():
                break

            idx_k = idx_k.to(device)

            # # case A: idx 指向列（ idx < video_block_num ）
            col_case = (idx_k < video_block_num) & active_mask

            if col_case.any():
                # 构造列掩码并按 head 广播
                # col_eq: (num_heads, R) 表示每 head 哪一列需要置真
                col_eq = (torch.arange(video_block_num, device=device).view(1, video_block_num) == idx_k.view(num_heads, 1))
                col_eq = col_eq & col_case.view(num_heads, 1)
                # expand 到 (num_heads, R, R) 表示整列为 True
                col_mask_2d = col_eq.unsqueeze(1).expand(-1, video_block_num, -1)
                video_mask |= col_mask_2d

            # case B: idx 指向对角线类（ idx >= video_block_num ）
            diag_case = (idx_k >= video_block_num) & active_mask
            if diag_case.any():
                offsets = idx_k - (video_block_num - 1) - video_block_num  # (num_heads,)
                # 比较 diff == offsets[head]，先扩展 offsets
                offsets_exp = offsets.view(num_heads, 1, 1)
                diff_exp = diff.expand(num_heads, video_block_num, video_block_num)  # (num_heads, R, R)
                diag_mask = (diff_exp == offsets_exp) & diag_case.view(num_heads, 1, 1)
                video_mask |= diag_mask

        # block_diagonal：按 blocks_per_frame 将相同 block_index 的方块置为 True
        if block_diag_heads.any():
            # 计算 block index 网格 (R, R)
            row_blk = (torch.arange(video_block_num, device=device) // blocks_per_frame).view(video_block_num, 1)
            col_blk = (torch.arange(video_block_num, device=device) // blocks_per_frame).view(1, video_block_num)
            block_diag_2d = (row_blk == col_blk)  # (R, R)
            # 按 head 应用
            video_mask |= block_diag_heads.view(num_heads, 1, 1) & block_diag_2d.unsqueeze(0)

    # 合并回全 mask（遵循原逻辑：cogvideox / hunyuan 两种布局）
    if warmup_state.get('model_type') == 'cogvideox':
        # text 区域全 True；video 区域用 video_mask
        if text_block_num > 0:
            mask[:, :text_block_num, :] = True
            mask[:, :, :text_block_num] = True
        mask[:, text_block_num:, text_block_num:] = video_mask
    elif warmup_state.get('model_type') == 'hunyuan':
        # video 区域全 True；text 区域用 video_mask
        mask[:, video_block_num:, :] = True
        mask[:, :, video_block_num:] = True
        mask[:, :video_block_num, :video_block_num] = video_mask
    elif warmup_state.get('model_type') == 'wan':
        mask = video_mask
    else:
        # 若未知 model_type，保守返回全 True
        mask = torch.ones_like(mask)
    print(f"Final Mask Shape: {mask.shape}")
    return mask


# ============================================================================
# Main Function: SparseAttentionWithMap
# ============================================================================


def SparseAttentionWithMap(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    warmup_state: Dict,
    logger: logging.Logger,
    pre_defined_mask: Optional[torch.Tensor] = None,
    current_step: int = 0,
    layer_idx: int = 0,
    use_dense: bool = False,
    in_test: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
    """
    执行自适应稀疏attention并返回结果    
    
    参数:
        query: query tensor, shape (1, num_heads, seq_len, head_dim)
        key: key tensor, shape (1, num_heads, seq_len, head_dim)
        value: value tensor, shape (1, num_heads, seq_len, head_dim)
        warmup_state: warmup状态字典（通过init_warmup_state创建）
        pre_defined_mask: 预定义的mask (Hunyuan模型需要)
        current_step: 当前步数
        top_k: 选择最亮的K条线
        use_cuda: 是否使用CUDA加速
        block_size: block大小（用于backend）
        **kwargs: 其他参数
    
    返回:
        output
    """
    if use_dense:
        mask = torch.ones((warmup_state['video_token_num'] // warmup_state['block_size'], warmup_state['video_token_num'] // warmup_state['block_size']), device=query.device, dtype=torch.bool)
        output = AttentionSparseEngine(
                    query,
                    key,
                    value,
                    mask,
                    pre_defined_mask=pre_defined_mask,
                    video_token_num=warmup_state.get('video_token_num',0),
                    block_size=warmup_state['block_size']
                )
        mask_info = {'mode': 'warmup', 'mask': None}
        return output, mask_info
    if warmup_state['sparse_type'] == 'spargeattn':
        output = spas_sage2_attn_meansim_topk_cuda(query, key, value, topk=0.5, is_causal=False)
        mask_info = {'mode': 'warmup', 'mask': None}
        return output, mask_info

    num_heads = query.size(1)
    seq_len = query.size(2)
    # 判断是否在warmup阶段
    in_warmup = current_step < warmup_state['warmup_steps']
    if in_warmup:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        if warmup_state['sparse_type'] == 'radial':
            if warmup_state.get('mask', None) is None:
                warmup_state['mask'] = get_radial_mask(query, warmup_state['video_token_num'], warmup_state['num_frames'], model_type=warmup_state['model_type'])
            logger.info(f"Layer {layer_idx}, Mask Sparsity: {1 - warmup_state['mask'].float().mean():.3%}")
            output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0,is_causal=False)
            mask_info = {'mode': 'warmup', 'mask': None}
            return output, mask_info
        elif warmup_state['sparse_type'] == 'mod-dit':
            if current_step >= warmup_state['warmup_steps'] - 2:
                output, block_sparse_map = compute_block_sparse_map_triton(query, key, value, warmup_state)
                block_sparse_map = block_sparse_map.squeeze(0)
                update_warmup(
                    warmup_state,
                    current_step,
                    block_sparse_map,
                )
                mask_info = {'mode': 'warmup', 'mask': None}
            
            # 当 Warmup 最后一步完成时，预先规划第一个掩码
            if current_step == warmup_state['warmup_steps'] - 1:
                mask_info = predict_mask_from_warmup(
                    warmup_state,
                    seq_len,
                    top_k=warmup_state['top_k'],
                    layer_idx=layer_idx
                )
                warmup_state['mask'] = mask_info['mask'] # 实际是下一步要使用的mask
                logger.info(f"Layer {layer_idx}, Mask Sparsity: {1 - warmup_state['mask'].float().mean():.3%}")
                warmup_state['mode'] = mask_info['mode']

            return output, warmup_state['mask']
    
    else:
        if warmup_state.get('mask', None) is not None:
            if warmup_state['sparse_type'] == 'mod-dit':
                if (current_step - warmup_state['warmup_steps']) % warmup_state['predict_T'] == warmup_state['predict_T'] - 1:
                    output, current_sparse_map = compute_block_sparse_map_triton(query, key, value, warmup_state)
                    current_sparse_map = current_sparse_map.squeeze(0)
                    
                    block_size = warmup_state['block_size']
                    if warmup_state['model_type']=='cogvideox':
                        current_mask = warmup_state['mask'][:, warmup_state['text_token_num']//block_size:, warmup_state['text_token_num']//block_size:]
                    if warmup_state['model_type']=='hunyuan':
                        current_mask = warmup_state['mask'][:, :warmup_state['video_token_num']//block_size, :warmup_state['video_token_num']//block_size]
                    if warmup_state['model_type']=='wan':
                        current_mask = warmup_state['mask']

                    warmup_state['prev_full_maps'] = torch.where(
                        current_mask==False,
                        warmup_state['prev_full_maps'],
                        current_sparse_map
                    )

                    features = extract_attention_features(
                        warmup_state=warmup_state,
                        S_T=warmup_state['prev_full_maps'],
                        layer_idx=layer_idx,
                        blocks_per_frame=warmup_state['blocks_per_frame'],
                        use_cuda=warmup_state['use_cuda']
                    )
                    
                    warmup_state['features_history'].append(features)
                    warmup_state['features_history'].pop(0) 

                    selected_lines = choose_topk_lines(
                        warmup_state['features_history'], 
                        top_k=warmup_state['top_k'], 
                        predict_T=warmup_state.get('predict_T', None)
                    )
                    next_mask = generate_mask_from_lines(
                        warmup_state, 
                        num_heads, 
                        seq_len, 
                        warmup_state['blocks_per_frame'], 
                        selected_lines, 
                        warmup_state['mode'], 
                        device=query.device
                    )
                    warmup_state['mask'] = next_mask
                else:
                    output = AttentionSparseEngine(
                        query,
                        key,
                        value,
                        warmup_state['mask'],
                        pre_defined_mask=pre_defined_mask,
                        video_token_num=warmup_state.get('video_token_num',0),
                        block_size=warmup_state['block_size']
                    )
            elif warmup_state['sparse_type'] == 'radial':
                output = AttentionSparseEngine(
                    query,
                    key,
                    value,
                    warmup_state['mask'],
                    pre_defined_mask=pre_defined_mask,
                    video_token_num=warmup_state.get('video_token_num',0),
                    block_size=warmup_state['block_size']
                )
            elif warmup_state['sparse_type'] == 'svg':
                temporal_mask = get_svg_mask(mask_name='temporal', context_length=warmup_state['text_token_num'], num_frame=warmup_state['num_frames'],
                                            frame_size=warmup_state['frame_size'], block_size=warmup_state['block_size'])
                spatial_mask = get_svg_mask(mask_name='spatial', context_length=warmup_state['text_token_num'], num_frame=warmup_state['num_frames'],
                                            frame_size=warmup_state['frame_size'], block_size=warmup_state['block_size'])
                temporal_num_heads = num_heads // 2
                spatial_num_heads = num_heads - temporal_num_heads
                warmup_state['mask'] = torch.cat([temporal_mask.repeat(temporal_num_heads,1,1), spatial_mask.repeat(spatial_num_heads,1,1)], dim=0)

        mask_info = {
            'mask': warmup_state['mask'], # 返回的是 *下一步* 的 mask
        }
        
        # 返回 (output, mask_info)
        return output, mask_info



