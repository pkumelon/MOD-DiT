"""
FlashAttention plus an exact video-video block sparsity map.

Pass 1 uses FlashAttention to produce the attention output and final row LSE.
Pass 2 replays tiled QK products and counts probabilities below the threshold:
an SM80 CUDA Tensor Core kernel is preferred, with Triton as a fallback.

Inputs: query, key, value (B, H, S, D), threshold, block_size (64 or 128), model_type.
Outputs: attention output (B, H, S, D), sparsity map (B, H, video_blocks, video_blocks).
"""

import math
import os
import warnings
from typing import Tuple

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_func

from MoD.mod.cuda_flash_sparsity import cuda_backend_supported, cuda_sparsity_map


_AUTO_FALLBACK_WARNED = False


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


def _video_layout(
    model_type: str,
    sequence_length: int,
    video_token_num: int,
    text_token_num: int,
    block_size: int,
) -> Tuple[int, int, int]:
    if model_type == "wan":
        if video_token_num not in (0, sequence_length):
            raise ValueError(
                f"wan expects video_token_num to be 0 or S={sequence_length}, got {video_token_num}"
            )
        return sequence_length // block_size, 0, 0
    if model_type == "hunyuan":
        return video_token_num // block_size, 0, 0
    if model_type == "cogvideox":
        return video_token_num // block_size, text_token_num, text_token_num
    raise ValueError(
        f"Unknown model_type: {model_type}, expected 'wan', 'hunyuan', or 'cogvideox'"
    )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    threshold,
    block_size: int,
    model_type: str,
    video_token_num: int,
    text_token_num: int,
) -> Tuple[float, int, int, int]:
    if query.ndim != 4:
        raise ValueError(f"query must have shape [B, H, S, D], got {tuple(query.shape)}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must have identical [B, H, S, D] shapes")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("query, key, and value must be CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same CUDA device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"expected float16 or bfloat16 inputs, got {query.dtype}")
    if block_size not in (64, 128):
        raise ValueError(f"block_size must be 64 or 128, got {block_size}")

    _, _, sequence_length, head_dim = query.shape
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    if model_type == "wan":
        video_tokens = sequence_length
    else:
        if video_token_num <= 0:
            raise ValueError(f"video_token_num must be positive for {model_type}")
        video_tokens = video_token_num
    if video_tokens % block_size:
        raise ValueError(
            f"video_token_num={video_tokens} must be divisible by block_size={block_size}"
        )
    if text_token_num < 0:
        raise ValueError("text_token_num must be non-negative")
    if model_type in ("hunyuan", "cogvideox") and video_token_num + text_token_num > sequence_length:
        raise ValueError("video and text regions exceed the sequence length")

    if torch.is_tensor(threshold):
        if threshold.numel() != 1:
            raise ValueError("threshold tensor must contain one element")
        threshold_value = float(threshold.detach().item())
    else:
        threshold_value = float(threshold)
    if math.isnan(threshold_value):
        raise ValueError("threshold must not be NaN")

    video_blocks, q_offset, k_offset = _video_layout(
        model_type, sequence_length, video_token_num, text_token_num, block_size
    )
    return threshold_value, video_blocks, q_offset, k_offset


def _triton_sparsity_map(
    query: torch.Tensor,
    key: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float,
    log_threshold: float,
    q_video_offset: int,
    k_video_offset: int,
    video_blocks: int,
    block_size: int,
) -> torch.Tensor:
    batch, heads, sequence_length, head_dim = query.shape
    sparsity_map = torch.empty(
        batch,
        heads,
        video_blocks,
        video_blocks,
        device=query.device,
        dtype=torch.float16,
    )
    num_warps = 4 if head_dim <= 64 else 8
    _sparsity_kernel[(video_blocks, batch * heads)](
        query,
        key,
        lse,
        sparsity_map,
        softmax_scale,
        log_threshold,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        sparsity_map.stride(0), sparsity_map.stride(1),
        sparsity_map.stride(2), sparsity_map.stride(3),
        heads,
        sequence_length,
        q_video_offset,
        k_video_offset,
        video_blocks,
        BLOCK_SIZE=block_size,
        BLOCK_D=head_dim,
        num_warps=num_warps,
        num_stages=2,
    )
    return sparsity_map


def _compute_sparsity_map(
    query: torch.Tensor,
    key: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float,
    log_threshold: float,
    q_video_offset: int,
    k_video_offset: int,
    video_blocks: int,
    block_size: int,
) -> torch.Tensor:
    global _AUTO_FALLBACK_WARNED
    backend = os.environ.get("MOD_DIT_SPARSITY_BACKEND", "auto").lower()
    if backend not in {"auto", "cuda", "triton"}:
        raise ValueError(
            "MOD_DIT_SPARSITY_BACKEND must be one of: auto, cuda, triton"
        )

    kwargs = dict(
        softmax_scale=softmax_scale,
        log_threshold=log_threshold,
        q_video_offset=q_video_offset,
        k_video_offset=k_video_offset,
        video_blocks=video_blocks,
        block_size=block_size,
    )
    if backend in {"auto", "cuda"} and cuda_backend_supported(query.device):
        try:
            return cuda_sparsity_map(query, key, lse, **kwargs)
        except RuntimeError as error:
            if backend == "cuda":
                raise
            if not _AUTO_FALLBACK_WARNED:
                warnings.warn(
                    f"CUDA sparsity backend unavailable; falling back to Triton: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _AUTO_FALLBACK_WARNED = True
    elif backend == "cuda":
        raise RuntimeError("CUDA sparsity backend requires an SM80 or newer GPU")

    return _triton_sparsity_map(query, key, lse, **kwargs)


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
        model_type: "wan" (video only), "hunyuan" (video first), or
            "cogvideox" (text first)
        video_token_num: number of video tokens
        text_token_num: number of text tokens

    Returns:
        output:       (B, H, S, D)
        sparsity_map: (B, H, video_block_num, video_block_num)
    """
    threshold_value, video_block_num, q_video_offset, k_video_offset = _validate_inputs(
        query,
        key,
        value,
        threshold,
        block_size,
        model_type,
        video_token_num,
        text_token_num,
    )
    B, H, S, D = query.shape
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

    # ── Pass 1: flash_attn → O, logsumexp ──
    q_fa = query.transpose(1, 2)   # (B, S, H, D)
    k_fa = key.transpose(1, 2)
    v_fa = value.transpose(1, 2)

    sm_scale = D ** -0.5
    o_fa, softmax_lse, _ = flash_attn_func(
        q_fa, k_fa, v_fa, softmax_scale=sm_scale, return_attn_probs=True)

    O = o_fa.transpose(1, 2)                    # (B, H, S, D)
    L = softmax_lse.contiguous()                 # (B, H, S), ln domain, float32

    # Pass 2 replays video-video QK tiles using the final row LSE. The CUDA
    # backend uses SM80 Tensor Cores; Triton remains the portable fallback.
    ln_thresh = math.log(threshold_value) if threshold_value > 0 else float("-inf")
    sparsity_map = _compute_sparsity_map(
        query,
        key,
        L,
        softmax_scale=sm_scale,
        log_threshold=ln_thresh,
        q_video_offset=q_video_offset,
        k_video_offset=k_video_offset,
        video_blocks=video_block_num,
        block_size=block_size,
    )

    return O, sparsity_map


# ──────────────────────── Test & Benchmark ────────────────────────

def _naive_attn_with_sparsity(query, key, value, threshold, block_size,
                               model_type="hunyuan", video_token_num=0, text_token_num=0):
    B, H, S, D = query.shape
    scale = D ** -0.5
    effective_video_tokens = S if model_type == "wan" else video_token_num
    video_block_num = effective_video_tokens // block_size

    scores = query.float() @ key.float().transpose(-1, -2) * scale
    probs = scores.softmax(dim=-1)
    output = (probs @ value.float()).to(query.dtype)

    if model_type in {"hunyuan", "wan"}:
        q_start, k_start = 0, 0
    elif model_type == "cogvideox":
        q_start, k_start = text_token_num, text_token_num
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Extract video-video attention probs
    p_video = probs[:, :, q_start:q_start + effective_video_tokens,
                          k_start:k_start + effective_video_tokens]
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
