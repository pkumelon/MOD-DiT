#!/usr/bin/env python3
"""Benchmark the exact CUDA and Triton sparsity replay backends on NVIDIA GPUs."""

import argparse
import math
import os
from contextlib import contextmanager

import torch
from flash_attn import flash_attn_func

from MoD.mod.cuda_flash_sparsity import cuda_backend_supported, cuda_sparsity_map
from MoD.mod.triton_flash_sparsity import (
    _triton_sparsity_map,
    flash_attn_with_sparsity_map,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--block-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--model-type", choices=("hunyuan", "cogvideox", "wan"), default="hunyuan")
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def measure(operation, warmup, repetitions):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "p20": quantile(samples, 0.2),
        "median": quantile(samples, 0.5),
        "p80": quantile(samples, 0.8),
    }


@contextmanager
def forced_backend(name):
    previous = os.environ.get("MOD_DIT_SPARSITY_BACKEND")
    os.environ["MOD_DIT_SPARSITY_BACKEND"] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MOD_DIT_SPARSITY_BACKEND", None)
        else:
            os.environ["MOD_DIT_SPARSITY_BACKEND"] = previous


def print_result(label, result):
    print(
        f"{label:28s} median={result['median']:8.3f} ms  "
        f"p20={result['p20']:8.3f}  p80={result['p80']:8.3f}"
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not cuda_backend_supported():
        raise SystemExit("The CUDA backend requires SM80 or newer")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    video_tokens = args.sequence if args.model_type == "wan" else args.sequence - args.text_tokens
    if video_tokens % args.block_size:
        raise SystemExit("video token count must be divisible by block size")
    offset = args.text_tokens if args.model_type == "cogvideox" else 0
    video_blocks = video_tokens // args.block_size
    scale = args.head_dim ** -0.5
    log_threshold = math.log(args.threshold) if args.threshold > 0 else float("-inf")

    torch.manual_seed(123)
    shape = (args.batch, args.heads, args.sequence, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    q_fa = query.transpose(1, 2)
    k_fa = key.transpose(1, 2)
    v_fa = value.transpose(1, 2)
    _, lse, _ = flash_attn_func(
        q_fa,
        k_fa,
        v_fa,
        softmax_scale=scale,
        return_attn_probs=True,
    )
    lse = lse.contiguous()

    common = dict(
        softmax_scale=scale,
        log_threshold=log_threshold,
        q_video_offset=offset,
        k_video_offset=offset,
        video_blocks=video_blocks,
        block_size=args.block_size,
    )

    # Trigger extension compilation before measurement.
    cuda_sparsity_map(query, key, lse, **common)
    _triton_sparsity_map(query, key, lse, **common)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    cuda_pass = measure(
        lambda: cuda_sparsity_map(query, key, lse, **common),
        args.warmup,
        args.repetitions,
    )
    cuda_memory = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    triton_pass = measure(
        lambda: _triton_sparsity_map(query, key, lse, **common),
        args.warmup,
        args.repetitions,
    )
    triton_memory = torch.cuda.max_memory_allocated()

    public_args = dict(
        threshold=args.threshold,
        block_size=args.block_size,
        model_type=args.model_type,
        video_token_num=0 if args.model_type == "wan" else video_tokens,
        text_token_num=0 if args.model_type == "wan" else args.text_tokens,
    )
    with forced_backend("cuda"):
        cuda_total = measure(
            lambda: flash_attn_with_sparsity_map(query, key, value, **public_args),
            args.warmup,
            args.repetitions,
        )
    with forced_backend("triton"):
        triton_total = measure(
            lambda: flash_attn_with_sparsity_map(query, key, value, **public_args),
            args.warmup,
            args.repetitions,
        )

    print(f"device: {torch.cuda.get_device_name()} ({torch.cuda.get_device_capability()})")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence} D={args.head_dim}, "
        f"block={args.block_size}, dtype={args.dtype}, model={args.model_type}"
    )
    print_result("CUDA sparsity pass", cuda_pass)
    print_result("Triton sparsity pass", triton_pass)
    print_result("FlashAttn + CUDA", cuda_total)
    print_result("FlashAttn + Triton", triton_total)
    print(f"sparsity-pass speedup: {triton_pass['median'] / cuda_pass['median']:.3f}x")
    print(f"end-to-end speedup:    {triton_total['median'] / cuda_total['median']:.3f}x")
    print(f"peak allocated (CUDA/Triton): {cuda_memory / 2**20:.1f}/{triton_memory / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
