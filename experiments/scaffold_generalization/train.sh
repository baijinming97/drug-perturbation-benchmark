#!/usr/bin/env bash
# scaffold_generalization — bash entry with sanity checks.
# Default: 7 models × {none, zero} × 5 folds = 70 runs.
#
# Sanity (1 epoch, fold 0, both ablations = 14 runs):
#   bash experiments/scaffold_generalization/train.sh --epochs 1 --folds 0
#
# Full reproduction (paper epochs × all 5 folds):
#   bash experiments/scaffold_generalization/train.sh

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

H5AD="$REPO_ROOT/data/XPert/processed_data/l1000_sdst_78453.h5ad"
if [ ! -e "$H5AD" ]; then
    echo "ERROR: $H5AD not found." >&2
    echo "  Run: python experiments/scaffold_generalization/prepare.py" >&2
    exit 1
fi

GENE_NM="$REPO_ROOT/data/CIGER/gene_feature_nm.csv"
if [ ! -e "$GENE_NM" ]; then
    echo "ERROR: $GENE_NM not found. Run prepare.py" >&2; exit 1; fi

DRUG_EMB="$REPO_ROOT/data/PertDiT/bench_drug_emb/bench_drug_emb.pkl"
if [ ! -e "$DRUG_EMB" ]; then
    echo "ERROR: $DRUG_EMB not found. Run prepare.py (~25 min step E)" >&2; exit 1; fi

XPERT_ZERO="$REPO_ROOT/data/_xpert_ablation/drugs_unimol_zero_all.npy"
if [ ! -e "$XPERT_ZERO" ]; then
    echo "ERROR: $XPERT_ZERO not found (required for --ablation_modes zero). Run prepare.py step G" >&2; exit 1; fi

echo "[OK] env=benchmark, CUDA available, all data ready"
echo "[OK] dispatching to: python $PY $*"
echo
exec python "$PY" "$@"
