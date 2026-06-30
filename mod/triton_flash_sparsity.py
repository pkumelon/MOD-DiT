"""
FlashAttention + block-level sparsity map .
主要是flashattention用的onlinesoftmax不好改成能反馈中间结果的状态，这里相当于加了一个专门算sparsity attention map的triton
Architecture:
  Pass 1 – flash_attn_func (Dao's optimized CUDA kernel) → O, logsumexp L
  Pass 2 – Lightweight Triton kernel: re-scan Q,K with L to count sparsity in log domain

Inputs:  query, key, value (B, H, S, D), threshold (float), block_size (int: 64 or 128), model_type
Outputs: attention output (B, H, S, D), sparsity_map (B, H, video_block_num, video_block_num)
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple
from flash_attn import flash_attn_func


@triton.jit
def _sparsity_kernel(
    Q, K, L, Sparsity,
    sm_scale,
    ln_threshold,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_lz, stride_lh, stride_lm,
    stride_sz, stride_sh, stride_si, stride_sj,
    H: tl.int32,
    N_CTX: tl.int32,
    Q_VIDEO_OFFSET: tl.int32,
    K_VIDEO_OFFSET: tl.int32,
    VIDEO_BLOCK_NUM: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Only computes sparsity for video-video block pairs.
    Q_VIDEO_OFFSET: starting token index of video in Q (in tokens, not blocks)
    K_VIDEO_OFFSET: starting token index of video in K (in tokens, not blocks)
    VIDEO_BLOCK_NUM: number of video blocks
    """
    block_i = tl.program_id(0)   # which video Q block (0..VIDEO_BLOCK_NUM-1)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    l_off = off_z * stride_lz + off_h * stride_lh
    s_off = off_z * stride_sz + off_h * stride_sh

    # Q rows: offset to video region
    offs_m = Q_VIDEO_OFFSET + block_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block (video region) → stays in SRAM
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs)
    q = (q * sm_scale).to(q.dtype)

    # Load logsumexp L for this Q block (ln domain, from flash_attn)
    l_ptrs = L + l_off + offs_m * stride_lm
    L_i = tl.load(l_ptrs)  # (BLOCK_SIZE,), float32, ln domain

    # Threshold line: softmax(s_ij) < threshold ⟺ s_ij < L_i + ln(threshold)
    thresh_line = L_i + ln_threshold
    inv_area = 1.0 / (BLOCK_SIZE * BLOCK_SIZE)

    for block_j in range(VIDEO_BLOCK_NUM):
        # K columns: offset to video region
        offs_n = K_VIDEO_OFFSET + block_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Load K^T: (BLOCK_D, BLOCK_SIZE)
        k_ptrs = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k = tl.load(k_ptrs)

        # QK^T (in natural scale, since flash_attn L is in ln domain)
        qk = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Count entries below threshold
        below = (qk < thresh_line[:, None]).to(tl.float32)
        row_counts = tl.sum(below, axis=1)
        total_count = tl.sum(row_counts, axis=0)
        sparsity = total_count * inv_area

        # Store sparsity_map[b, h, block_i, block_j]
        s_ptr = Sparsity + s_off + block_i * stride_si + block_j * stride_sj
        tl.store(s_ptr, sparsity.to(Sparsity.dtype.element_ty))


@torch.no_grad()
def flash_attn_with_sparsity_map(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    threshold: float,
    block_size: int,
    model_type: str = "hunyuan",
    video_token_num: int = 0,
    text_token_num: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output and video-video block-level sparsity map.

    Args:
        query, key, value: (B, H, S, D) fp16/bf16
        threshold: sparsity threshold
        block_size: 64 or 128
        model_type: "hunyuan" (video first) or "cogvideox" (text first)
        video_token_num: number of video tokens
        text_token_num: number of text tokens

    Returns:
        output:       (B, H, S, D)
        sparsity_map: (B, H, video_block_num, video_block_num)
    """
    B, H, S, D = query.shape
    assert S % block_size == 0, f"S={S} must be divisible by block_size={block_size}"
    assert D in {64, 128}, f"head_dim={D}, expected 64 or 128"

    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

    # ── Pass 1: flash_attn → O, logsumexp ──
    q_fa = query.transpose(1, 2)   # (B, S, H, D)
    k_fa = key.transpose(1, 2)
    v_fa = value.transpose(1, 2)

    sm_scale = D ** -0.5
    o_fa, softmax_lse, _ = flash_attn_func(
        q_fa, k_fa, v_fa, softmax_scale=sm_scale, return_attn_probs=True)

    O = o_fa.transpose(1, 2)                    # (B, H, S, D)
    L = softmax_lse                              # (B, H, S), ln domain, float32

    # ── Determine video region offsets based on model_type ──
    if model_type == "wan":
        # wan: self-attention only has video tokens, no text tokens in sequence
        video_block_num = S // block_size
        q_video_offset = 0
        k_video_offset = 0
    elif model_type == "hunyuan":
        # hunyuan: video tokens first, text tokens after
        video_block_num = video_token_num // block_size
        q_video_offset = 0
        k_video_offset = 0
    elif model_type == "cogvideox":
        # cogvideox: text tokens first, video tokens after
        video_block_num = video_token_num // block_size
        q_video_offset = text_token_num
        k_video_offset = text_token_num
    else:
        raise ValueError(f"Unknown model_type: {model_type}, expected 'wan', 'hunyuan', or 'cogvideox'")

    # ── Pass 2: Triton sparsity kernel (video-video only) ──
    ln_thresh = math.log(threshold) if threshold > 0 else float("-inf")

    sparsity_map = torch.empty(B, H, video_block_num, video_block_num,
                               device=query.device, dtype=torch.float16)

    grid = (video_block_num, B * H)
    num_warps = 4 if D <= 64 else 8

    _sparsity_kernel[grid](
        query, key, L, sparsity_map,
        sm_scale, ln_thresh,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        sparsity_map.stride(0), sparsity_map.stride(1),
        sparsity_map.stride(2), sparsity_map.stride(3),
        H, S,
        q_video_offset, k_video_offset, video_block_num,
        BLOCK_SIZE=block_size,
        BLOCK_D=D,
        num_warps=num_warps,
        num_stages=2,
    )

    return O, sparsity_map


# ──────────────────────── Test & Benchmark ────────────────────────

def _naive_attn_with_sparsity(query, key, value, threshold, block_size,
                               model_type="hunyuan", video_token_num=0, text_token_num=0):
    B, H, S, D = query.shape
    scale = D ** -0.5
    video_block_num = video_token_num // block_size

    scores = query.float() @ key.float().transpose(-1, -2) * scale
    probs = scores.softmax(dim=-1)
    output = (probs @ value.float()).to(query.dtype)

    if model_type == "hunyuan":
        q_start, k_start = 0, 0
    else:
        q_start, k_start = text_token_num, text_token_num

    # Extract video-video attention probs
    p_video = probs[:, :, q_start:q_start + video_token_num,
                          k_start:k_start + video_token_num]
    p_video = p_video.view(B, H, video_block_num, block_size, video_block_num, block_size)
    sparsity_map = p_video.lt(threshold).float().mean(dim=(3, 5)).to(torch.float16)

    return output, sparsity_map


def test_correctness(B=1, H=4, S=640, D=128, block_size=64, threshold=0.01,
                     model_type="hunyuan"):
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # Ensure video_token_num is divisible by block_size
    text_token_num = 128
    video_token_num = S - text_token_num

    o_tri, sp_tri = flash_attn_with_sparsity_map(
        q, k, v, threshold, block_size,
        model_type=model_type,
        video_token_num=video_token_num,
        text_token_num=text_token_num)
    o_ref, sp_ref = _naive_attn_with_sparsity(
        q, k, v, threshold, block_size,
        model_type=model_type,
        video_token_num=video_token_num,
        text_token_num=text_token_num)

    o_err = (o_tri.float() - o_ref.float()).abs().max().item()
    sp_err = (sp_tri.float() - sp_ref.float()).abs().max().item()
    print(f"  O max error:        {o_err:.6f}")
    print(f"  Sparsity max error: {sp_err:.6f}")
    assert o_err < 5e-2, f"O error too large: {o_err}"
    assert sp_err < 5e-2, f"Sparsity error too large: {sp_err}"


def benchmark(B=1, H=24, S=4096, D=128, block_size=128, threshold=0.01,
              warmup=10, rep=50):
    import time
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    video_token_num = S - 128
    text_token_num = 128

    for _ in range(warmup):
        flash_attn_with_sparsity_map(q, k, v, threshold, block_size,
                                     video_token_num=video_token_num,
                                     text_token_num=text_token_num)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        flash_attn_with_sparsity_map(q, k, v, threshold, block_size,
                                     video_token_num=video_token_num,
                                     text_token_num=text_token_num)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) / rep * 1000

    for _ in range(warmup):
        _naive_attn_with_sparsity(q, k, v, threshold, block_size,
                                  video_token_num=video_token_num,
                                  text_token_num=text_token_num)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        _naive_attn_with_sparsity(q, k, v, threshold, block_size,
                                  video_token_num=video_token_num,
                                  text_token_num=text_token_num)
    torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - start) / rep * 1000

    print(f"  flash_attn + Triton sparsity:  {triton_ms:.2f} ms")
    print(f"  Naive PyTorch:                 {naive_ms:.2f} ms")
    print(f"  Speedup:                       {naive_ms / triton_ms:.1f}x")


if __name__ == "__main__":
    print("=== Correctness Tests ===")
    for model in ["hunyuan", "cogvideox"]:
        for bs in [64, 128]:
            print(f"model_type={model}, block_size={bs}, D=128")
            test_correctness(block_size=bs, D=128, model_type=model)
    print("\nAll correctness tests passed!\n")

    print("=== Benchmark (B=1, H=24, S=4096, D=128, block_size=128) ===")
    benchmark()
