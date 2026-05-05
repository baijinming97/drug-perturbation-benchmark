#!/usr/bin/env bash
# main_benchmark — bash entry point with environment sanity checks.
#
# Use this if you prefer bash and want pre-flight checks; otherwise call
# `python train_all.py ...` directly.
#
# Quick verification:
#   conda activate benchmark
#   bash experiments/main_benchmark/train.sh --epochs 1 --folds 0
#
# Full reproduction:
#   bash experiments/main_benchmark/train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$SCRIPT_DIR/train_all.py"

# 1. conda env (this benchmark requires the 'benchmark' env)
if [ "$(basename "${CONDA_DEFAULT_ENV:-}")" != "benchmark" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected 'benchmark'." >&2
    echo "  Build it once via:  conda env create -f environment/environment.yml && bash environment/install_pip.sh" >&2
    echo "  Then activate:      conda activate benchmark" >&2
    exit 1
fi

# 2. CUDA available
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
    echo "ERROR: CUDA not available — these training scripts require a GPU." >&2
    echo "  Verify with:  python -c 'import torch; print(torch.cuda.is_available())'" >&2
    exit 1
}

# 3. data/XPert/processed_data/ accessible (XPert l1000_sdst h5ad with drug-blind split columns)
H5AD="$REPO_ROOT/data/XPert/processed_data/l1000_sdst_78453.h5ad"
if [ ! -e "$H5AD" ]; then
    echo "ERROR: $H5AD not found." >&2
    echo "  Either:" >&2
    echo "    (a) extract data/XPert/processed_data.zip into data/XPert/processed_data/, or" >&2
    echo "    (b) symlink: ln -s <abs path to extracted dir> $REPO_ROOT/data/XPert/processed_data" >&2
    exit 1
fi

# All checks passed — dispatch
echo "[OK] env=benchmark, CUDA available, data found at $H5AD"
echo "[OK] dispatching to: python $PY $*"
echo
exec python "$PY" "$@"
