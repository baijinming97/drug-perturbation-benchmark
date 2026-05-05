# Internal: per-model GPU memory & throughput micro-benchmark

Stand-alone; not used by any other experiment. Measures peak GPU memory and
ms/iter for each vendored model under a controlled training step (10 warmup
+ 60 timed iters, paper-default batch size).

Prereq: `bash reference/fetch_upstream.sh` (and `data/XPert/processed_data/`
extracted for `bench_mlp.py`).

Run: `bash run_all_bench.sh` (one SLURM job, ~2 h on a100).

Output: `results/<MODEL>.json` — fields `ms_per_iter_*`,
`peak_gpu_memory_MB`, `peak_gpu_reserved_MB`, `total_params`, `gpu`,
`pytorch`, `cuda`. GPU-specific; compare only within the same GPU.
