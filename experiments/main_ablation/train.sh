#!/usr/bin/env bash
# main_ablation — bash entry point with environment sanity checks.
#
# Use this if you prefer bash and want pre-flight checks; otherwise call
# `python train_all.py ...` directly.
#
# Quick verification (1 epoch, fold 0, both ablations):
#   conda activate benchmark
#   bash experiments/main_ablation/train.sh --epochs 1 --folds 0
#
# Full reproduction:
#   bash experiments/main_ablation/train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$SCRIPT_DIR/train_all.py"

# 1. conda env
if [ "$(basename "${CONDA_DEFAULT_ENV:-}")" != "benchmark" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected 'benchmark'." >&2
    echo "  Build it once via:  conda env create -f environment/environment.yml && bash environment/install_pip.sh" >&2
    echo "  Then activate:      conda activate benchmark" >&2
    exit 1
fi

# 2. CUDA
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
    echo "ERROR: CUDA not available — these training scripts require a GPU." >&2
    exit 1
}

# 3. data
H5AD="$REPO_ROOT/data/XPert/processed_data/l1000_sdst_78453.h5ad"
if [ ! -e "$H5AD" ]; then
    echo "ERROR: $H5AD not found." >&2
    echo "  Run: python experiments/main_ablation/prepare.py" >&2
    exit 1
fi

# 4. CIGER gene_feature_nm.csv (ablation also uses CIGER → needs this)
GENE_NM="$REPO_ROOT/data/CIGER/gene_feature_nm.csv"
if [ ! -e "$GENE_NM" ]; then
    echo "ERROR: $GENE_NM not found." >&2
    echo "  Run: python experiments/main_ablation/prepare.py" >&2
    exit 1
fi

echo "[OK] env=benchmark, CUDA available, data + gene_feature_nm.csv ready"
echo "[OK] dispatching to: python $PY $*"
echo
exec python "$PY" "$@"
