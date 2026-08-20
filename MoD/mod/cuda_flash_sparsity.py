"""Lazy loader for the exact CUDA sparsity-map replay kernel."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.cpp_extension import load

_EXTENSION = None
_EXTENSION_ERROR: Optional[BaseException] = None
_EXTENSION_LOCK = threading.Lock()


def _extension_name() -> str:
    # Keep a stable module name while allowing PyTorch's extension versioner to
    # rebuild whenever the source or compiler flags change.
    return "mod_dit_flash_sparsity_cuda"


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("CUDA sparsity extension is unavailable") from _EXTENSION_ERROR

    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            raise RuntimeError("CUDA sparsity extension is unavailable") from _EXTENSION_ERROR

        source = Path(__file__).with_name("cuda_flash_sparsity_kernel.cu")
        try:
            _EXTENSION = load(
                name=_extension_name(),
                sources=[str(source)],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                    "-lineinfo",
                ],
                verbose=os.environ.get("MOD_DIT_CUDA_VERBOSE", "0") == "1",
            )
        except BaseException as error:
            _EXTENSION_ERROR = error
            raise RuntimeError(f"Failed to build CUDA sparsity extension: {error}") from error
        return _EXTENSION


def cuda_backend_supported(
    device: Optional[Union[torch.device, int]] = None,
) -> bool:
    """Return whether the active device can execute the SM80 WMMA kernel."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    device = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def cuda_sparsity_map(
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
    """Run the exact Tensor Core replay pass and return an FP16 block map."""
    if not cuda_backend_supported(query.device):
        capability = torch.cuda.get_device_capability(query.device) if query.is_cuda else None
        raise RuntimeError(f"CUDA sparsity backend requires SM80 or newer, got {capability}")

    extension = _load_extension()
    return extension.sparsity_map(
        query,
        key,
        lse,
        float(softmax_scale),
        float(log_threshold),
        int(q_video_offset),
        int(k_video_offset),
        int(video_blocks),
        int(block_size),
    )


def extension_error() -> Optional[BaseException]:
    """Expose a cached build error for diagnostics without rebuilding."""
    return _EXTENSION_ERROR
