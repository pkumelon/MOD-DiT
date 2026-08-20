import math
import os

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("flash_attn")

from MoD.mod.cuda_flash_sparsity import cuda_backend_supported, cuda_sparsity_map
from MoD.mod.triton_flash_sparsity import (
    _compute_sparsity_map,
    _triton_sparsity_map,
    _validate_inputs,
    flash_attn_with_sparsity_map,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _layout(model_type, block_size):
    if model_type == "wan":
        return block_size * 2, block_size * 2, 0
    return block_size * 3, block_size * 2, block_size


def _reference_map(query, key, threshold, block_size, model_type, video_tokens, text_tokens):
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    probabilities = torch.softmax(scores, dim=-1)
    offset = text_tokens if model_type == "cogvideox" else 0
    video = probabilities[
        :, :, offset : offset + video_tokens, offset : offset + video_tokens
    ]
    blocks = video_tokens // block_size
    return (
        video.reshape(
            query.shape[0],
            query.shape[1],
            blocks,
            block_size,
            blocks,
            block_size,
        )
        .lt(threshold)
        .float()
        .mean(dim=(3, 5))
    )


def _replay_lse(query, key):
    # Match the production replay path: scale Q, round back to the input dtype,
    # then accumulate QK in FP32/Tensor Cores.
    scale = query.shape[-1] ** -0.5
    scaled_query = (query * scale).to(query.dtype)
    scores = torch.matmul(scaled_query, key.transpose(-1, -2)).float()
    return torch.logsumexp(scores, dim=-1), scale


@pytest.mark.parametrize("model_type", ["hunyuan", "cogvideox", "wan"])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_replay_matches_triton(model_type, block_size, head_dim, dtype):
    if not cuda_backend_supported():
        pytest.skip("SM80 or newer is required")
    torch.manual_seed(7)
    sequence, video_tokens, text_tokens = _layout(model_type, block_size)
    query = torch.randn(2, 2, sequence, head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    lse, scale = _replay_lse(query, key)
    video_blocks = video_tokens // block_size
    offset = text_tokens if model_type == "cogvideox" else 0
    log_threshold = math.log(1e-3)

    cuda_map = cuda_sparsity_map(
        query,
        key,
        lse.contiguous(),
        softmax_scale=scale,
        log_threshold=log_threshold,
        q_video_offset=offset,
        k_video_offset=offset,
        video_blocks=video_blocks,
        block_size=block_size,
    )
    triton_map = _triton_sparsity_map(
        query,
        key,
        lse.contiguous(),
        softmax_scale=scale,
        log_threshold=log_threshold,
        q_video_offset=offset,
        k_video_offset=offset,
        video_blocks=video_blocks,
        block_size=block_size,
    )

    count_error = (cuda_map.float() - triton_map.float()).abs() * block_size**2
    assert count_error.max().item() <= 2.1
    assert cuda_map.shape == (2, 2, video_blocks, video_blocks)
    assert cuda_map.dtype == torch.float16
    assert torch.all((cuda_map >= 0) & (cuda_map <= 1))


@pytest.mark.parametrize("model_type", ["hunyuan", "cogvideox", "wan"])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_api_matches_probability_reference(
    monkeypatch, model_type, block_size, dtype
):
    torch.manual_seed(11)
    monkeypatch.setenv("MOD_DIT_SPARSITY_BACKEND", "triton")
    sequence, video_tokens, text_tokens = _layout(model_type, block_size)
    query = torch.randn(1, 2, sequence, 64, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    threshold = 1e-3

    output, sparsity = flash_attn_with_sparsity_map(
        query,
        key,
        value,
        threshold,
        block_size,
        model_type=model_type,
        video_token_num=video_tokens if model_type != "wan" else 0,
        text_token_num=text_tokens,
    )
    reference = _reference_map(
        query, key, threshold, block_size, model_type, video_tokens, text_tokens
    )

    assert output.shape == query.shape
    assert sparsity.shape == reference.shape
    # FlashAttention and PyTorch can differ for scores exactly at the threshold.
    count_error = (sparsity.float() - reference).abs() * block_size**2
    assert count_error.max().item() <= 8.1


def test_non_positive_threshold_produces_zero_map(monkeypatch):
    monkeypatch.setenv("MOD_DIT_SPARSITY_BACKEND", "triton")
    query = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    output, sparsity = flash_attn_with_sparsity_map(
        query,
        query,
        query,
        0.0,
        64,
        model_type="wan",
    )
    assert output.shape == query.shape
    assert torch.count_nonzero(sparsity).item() == 0


def test_forced_backend_dispatch(monkeypatch):
    query = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    lse, scale = _replay_lse(query, query)
    kwargs = dict(
        softmax_scale=scale,
        log_threshold=math.log(1e-3),
        q_video_offset=0,
        k_video_offset=0,
        video_blocks=2,
        block_size=64,
    )

    monkeypatch.setenv("MOD_DIT_SPARSITY_BACKEND", "triton")
    triton_map = _compute_sparsity_map(query, query, lse.contiguous(), **kwargs)
    assert triton_map.shape == (1, 1, 2, 2)

    monkeypatch.setenv("MOD_DIT_SPARSITY_BACKEND", "invalid")
    with pytest.raises(ValueError, match="MOD_DIT_SPARSITY_BACKEND"):
        _compute_sparsity_map(query, query, lse.contiguous(), **kwargs)


@pytest.mark.parametrize(
    "mutation,error_type,match",
    [
        ({"block_size": 32}, ValueError, "block_size"),
        ({"video_token_num": 65}, ValueError, "divisible"),
        ({"model_type": "unknown"}, ValueError, "Unknown model_type"),
        ({"threshold": float("nan")}, ValueError, "NaN"),
    ],
)
def test_input_validation(mutation, error_type, match):
    query = torch.empty(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    arguments = dict(
        query=query,
        key=query,
        value=query,
        threshold=1e-3,
        block_size=64,
        model_type="hunyuan",
        video_token_num=64,
        text_token_num=64,
    )
    arguments.update(mutation)
    with pytest.raises(error_type, match=match):
        _validate_inputs(**arguments)


def test_cuda_kernel_uses_current_stream():
    if not cuda_backend_supported():
        pytest.skip("SM80 or newer is required")
    query = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    lse, scale = _replay_lse(query, query)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = cuda_sparsity_map(
            query,
            query,
            lse.contiguous(),
            softmax_scale=scale,
            log_threshold=math.log(1e-3),
            q_video_offset=0,
            k_video_offset=0,
            video_blocks=2,
            block_size=64,
        )
        marker = result.sum()
    stream.synchronize()
    assert torch.isfinite(marker)
