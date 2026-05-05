"""Shared timing and output logic for the per-model micro-benchmark."""
import time
import json
import gc
import os
import torch
import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")

WARMUP = int(os.environ.get("BENCH_WARMUP", 10))
TOTAL = int(os.environ.get("BENCH_TOTAL", 60))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_bench_loop(model_name, model, step_fn, num_batches, total=TOTAL, warmup=WARMUP):
    """Run benchmark with explicit step function.

    Args:
        model_name: string identifier
        model: nn.Module (for param counting)
        step_fn: callable(batch_idx) that performs one training step.
        num_batches: total number of batches available in the dataloader.
    """
    reset_gpu()
    times = []
    use_cuda = torch.cuda.is_available()

    for i in range(total):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        step_fn(i)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000
        if i >= warmup:
            times.append(ms)
            print(f"  iter {i}: {ms:.1f} ms")

    return save_results(model_name, model, times)


def save_results(model_name, model, times):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sorted_t = sorted(times)
    n = len(times)
    mean_t = sum(times) / n
    use_cuda = torch.cuda.is_available()
    result = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if use_cuda else "N/A",
        "ms_per_iter_median": round(sorted_t[n // 2], 2),
        "ms_per_iter_mean": round(mean_t, 2),
        "ms_per_iter_std": round(
            (sum((t - mean_t) ** 2 for t in times) / n) ** 0.5, 2
        ),
        "ms_per_iter_all": [round(t, 2) for t in times],
        "peak_gpu_memory_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1) if use_cuda else 0,
        "peak_gpu_reserved_MB": round(torch.cuda.max_memory_reserved() / 1e6, 1) if use_cuda else 0,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 50}")
    print(
        f"  {model_name}: {result['ms_per_iter_median']:.1f} ms/iter, "
        f"{result['peak_gpu_memory_MB']:.0f} MB peak"
    )
    print(f"{'=' * 50}")
    return result
