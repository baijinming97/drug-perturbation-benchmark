#!/usr/bin/env bash
# mlp_baseline — bash entry with environment sanity checks.
# Default: dispatch to train_default.py (5-fold default L3_H2048 reproduction).
# For the 100-run robustness sweep use train_default.py's neighbour, sweep.py.
#
# Quick sanity:
#   conda activate benchmark
#   bash experiments/mlp_baseline/train.sh --epochs 1 --folds 0
#
# Full reproduction (paper default):
#   bash experiments/mlp_baseline/train.sh
#
# Robustness sweep (4 × 5 × 5 = 100 runs, ~12 h):
#   python experiments/mlp_baseline/sweep.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$SCRIPT_DIR/train_default.py"

if [ "$(basename "${CONDA_DEFAULT_ENV:-}")" != "benchmark" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected 'benchmark'." >&2
    echo "  conda env create -f environment/environment.yml && conda activate benchmark" >&2
    exit 1
fi

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
    echo "ERROR: CUDA not available — MLP training requires a GPU." >&2
    exit 1
}

H5AD="$REPO_ROOT/data/XPert/processed_data/l1000_sdst_78453.h5ad"
if [ ! -e "$H5AD" ]; then
    echo "ERROR: $H5AD not found." >&2
    echo "  Run: python experiments/mlp_baseline/prepare.py" >&2
    exit 1
fi

echo "[OK] env=benchmark, CUDA available, h5ad ready"
echo "[OK] dispatching to: python $PY $*"
echo
exec python "$PY" "$@"
