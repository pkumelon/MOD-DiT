from typing import Optional
import torch
import logging

def _to_gib(x: int) -> float:
    return x / (1024 ** 3)

def reset_peak_gpu_stats(device: Optional[torch.device] = None):
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(device.index)


def print_gpu_stats(tag: str = "", device: Optional[torch.device] = None, logger: Optional[logging.Logger] = None, do_sync = True):
    if not torch.cuda.is_available():
        logger.info(f"[{tag}] CUDA not available")
        return

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    if do_sync:
        torch.cuda.synchronize(device)

    alloc = torch.cuda.memory_allocated(device)
    reserv = torch.cuda.memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserv = torch.cuda.max_memory_reserved(device)

    logger.info(
        f"[{tag}] "
        f"alloc={_to_gib(alloc):.3f} GiB | "
        f"reserved={_to_gib(reserv):.3f} GiB | "
        f"peak_alloc={_to_gib(peak_alloc):.3f} GiB | "
        f"peak_reserved={_to_gib(peak_reserv):.3f} GiB | "
        f"cache_gap(reserved-alloc)={_to_gib(reserv-alloc):.3f} GiB"
    )