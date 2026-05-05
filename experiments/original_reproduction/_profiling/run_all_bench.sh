#!/bin/bash
# Submit ONE SLURM job that runs all 8 micro-benchmarks sequentially on a
# single GPU. Times each model's training step + records peak GPU memory.
#
# Usage:
#   bash experiments/original_reproduction/_profiling/run_all_bench.sh
#
# Output: experiments/original_reproduction/_profiling/results/<MODEL>.json
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BENCH_DIR/results"
mkdir -p "$LOG_DIR"

sbatch --partition=a100 --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=02:00:00 \
    --job-name="profiling_all" \
    --output="${LOG_DIR}/profiling_all_%j.log" \
    --wrap="eval \"\$(conda shell.bash hook)\" && conda activate benchmark && cd $BENCH_DIR && \
python bench_mlp.py && \
python bench_transigen.py && \
python bench_ciger.py && \
python bench_deepce.py && \
python bench_multidcp.py && \
python bench_pertdit.py && \
python bench_prnet.py && \
python bench_xpert.py && \
echo 'ALL 8 MODELS COMPLETE'"

echo "Submitted. Monitor:  squeue -u \$USER | grep profiling"
echo "Results:             ls $LOG_DIR/"
