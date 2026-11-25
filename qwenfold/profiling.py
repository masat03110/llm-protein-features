"""Wall-clock and peak-memory measurement for Apple Silicon (unified memory).

On Apple Silicon CPU and GPU share one memory pool, so we record
  * peak process RSS (CPU side, sampled), and
  * peak MPS driver-allocated memory (GPU side, sampled).
Their sum is an upper bound of the unified memory footprint of a run.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import psutil
import torch


def sync(device: str | torch.device):
    dev = str(device)
    if dev.startswith("mps"):
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


def _gpu_bytes(device: str) -> int:
    if device.startswith("mps"):
        return torch.mps.driver_allocated_memory()
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated()
    return 0


@contextmanager
def measure(device: str = "mps", interval: float = 0.02):
    """with measure(dev) as m: ...; afterwards m has seconds / peak_rss_gb / peak_gpu_gb."""
    device = str(device)
    proc = psutil.Process()
    stats = {"peak_rss": proc.memory_info().rss, "peak_gpu": _gpu_bytes(device)}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            stats["peak_rss"] = max(stats["peak_rss"], proc.memory_info().rss)
            stats["peak_gpu"] = max(stats["peak_gpu"], _gpu_bytes(device))
            time.sleep(interval)

    class Result:
        pass

    res = Result()
    th = threading.Thread(target=sampler, daemon=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    th.start()
    t0 = time.perf_counter()
    try:
        yield res
    finally:
        sync(device)
        res.seconds = time.perf_counter() - t0
        stop.set()
        th.join()
        stats["peak_rss"] = max(stats["peak_rss"], proc.memory_info().rss)
        stats["peak_gpu"] = max(stats["peak_gpu"], _gpu_bytes(device))
        res.peak_rss_gb = stats["peak_rss"] / 1024**3
        res.peak_gpu_gb = stats["peak_gpu"] / 1024**3
        # CUDA: exact peaks from the allocator (the sampled value above can miss short spikes)
        cuda = device.startswith("cuda")
        res.peak_gpu_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3 if cuda else None
        res.peak_gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if cuda else None


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
