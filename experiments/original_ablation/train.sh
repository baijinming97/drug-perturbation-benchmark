#!/usr/bin/env bash
# original_ablation — bash entry with sanity checks.
# Default: 6 models × {none, zero} × 5 folds (= 60 runs).
#
# Sanity (1 epoch, fold 0 only, ~20-40 min on a100):
#   conda activate benchmark
#   bash experiments/original_ablation/train.sh --epochs 1 --folds 0
#
# Full reproduction (paper epochs × all 5 folds):
#   bash experiments/original_ablation/train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$SCRIPT_DIR/train_all.py"

if [ "$(basename "${CONDA_DEFAULT_ENV:-}")" != "benchmark" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected 'benchmark'." >&2
    exit 1
fi

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
    echo "ERROR: CUDA not available — training requires a GPU." >&2; exit 1; }

# original_ablation needs converted data for all 6 models.
# PertDiT by design shares prnet/prnet_original.h5ad and only produces its own
# embedding pickles (see convert_pertdit.py); probe those instead of an h5ad.
for M in ciger deepce multidcp pertdit prnet transigen; do
    if [ "$M" = "pertdit" ]; then
        H="$REPO_ROOT/data/_converted/pertdit/p6_drug_emb.pkl"
    else
        H="$REPO_ROOT/data/_converted/$M/${M}_original.h5ad"
    fi
    if [ ! -e "$H" ]; then
        echo "ERROR: $H missing." >&2
        echo "  Run: python experiments/original_ablation/prepare.py" >&2
        exit 1
    fi
done

echo "[OK] env=benchmark, CUDA available, 6 models converted data ready"
echo "[OK] dispatching to: python $PY $*"
echo
exec python "$PY" "$@"
