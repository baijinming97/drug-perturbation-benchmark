#!/bin/bash
# Pip install steps for the `benchmark` conda env.
#
# Usage (from repo root):
#   conda env create -f environment/environment.yml      # creates the env
#   bash environment/install_pip.sh                       # this script
#
# This script self-activates the `benchmark` env via the env's own pip; it
# does NOT depend on `conda activate` having been run beforehand. That makes
# it safe to invoke from a non-interactive shell.
#
# Order matters: torch must precede flash-attn / torch-geometric extensions /
# unimol-tools (they are CUDA extensions or downstream of torch).
#
# Each invocation truncates and rewrites environment/INSTALL_LOG.md so the
# transcript reflects only this run (avoids confusion on retry / re-runs).
# The log is in .gitignore — it's a local audit artifact, not shipped.

set -euo pipefail

ENV_NAME="benchmark"

# Locate the env via conda's own machinery — works for any envs_dirs prefix
# (~/.conda/envs/, ~/miniconda3/envs/, /opt/anaconda3/envs/, custom condarc, …)
# instead of hardcoding $HOME/.conda/envs/.
if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: 'conda' not on PATH" >&2
    echo "Run first:  conda env create -f environment/environment.yml" >&2
    exit 1
fi
ENV_PREFIX="$(conda env list | awk -v name="$ENV_NAME" '$1 == name { print $NF }')"
if [ -z "$ENV_PREFIX" ] || [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "ERROR: env '$ENV_NAME' not found via 'conda env list'." >&2
    echo "Run first:  conda env create -f environment/environment.yml" >&2
    exit 1
fi
PY="$ENV_PREFIX/bin/python"
PIP="$PY -m pip"

LOG="environment/INSTALL_LOG.md"
mkdir -p "$(dirname "$LOG")"
echo "# Install log — $(date -Iseconds)" > "$LOG"   # truncate on each run
echo "" >> "$LOG"
echo "Python: $($PY --version 2>&1)" >> "$LOG"
echo "Pip:    $($PIP --version | head -1)" >> "$LOG"

run() {
    echo "" >> "$LOG"
    echo "## \`$*\`" >> "$LOG"
    echo "Started $(date +%H:%M:%S)" >> "$LOG"
    echo "[$(date +%H:%M:%S)] $*"
    "$@" 2>&1 | tee -a "$LOG"
    echo "Finished $(date +%H:%M:%S)" >> "$LOG"
}

echo "=== Stage 1/5: PyTorch 2.1.0 + cu121 ==="
run $PIP install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Stage 2/5: torch-geometric extensions (must match torch+cu version) ==="
run $PIP install \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    pyg-lib==0.4.0 \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
run $PIP install torch-geometric==2.6.1

echo "=== Stage 3/5: scientific / bio core ==="
run $PIP install \
    "numpy==1.26.4" \
    "pandas==2.3.0" \
    "scipy==1.13.1" \
    "scikit-learn==1.4.0" \
    "matplotlib==3.8.2" \
    "seaborn==0.13.2" \
    "h5py==3.10.0" \
    "pyyaml==6.0.1" \
    "tqdm==4.66.1" \
    "requests==2.28.1"
run $PIP install \
    "scanpy==1.9.8" \
    "anndata==0.10.9" \
    "rdkit==2024.3.2"   # last manylinux_2_17 wheel — works on glibc >= 2.17 (CentOS 7+, RHEL 7+, Ubuntu 14.04+); rdkit >= 2024.3.5 requires glibc >= 2.28

echo "=== Stage 4/5: deep-learning add-ons ==="
run $PIP install \
    "einops==0.8.2" \
    "torchmetrics==1.6.0" \
    "pytorch-lightning==1.9.5" \
    "diffusers==0.30.2"
# flash-attn: install the prebuilt wheel from the upstream GitHub release.
# Avoids needing nvcc / a CUDA toolkit on the install host — PyPI ships only
# the sdist for 2.6.0.post1, whose setup.py requires nvcc to compile and is
# RAM-hungry (OOMs on small nodes).
# The cu122 wheel is ABI-compatible with our torch 2.1.0+cu121 install (both
# cxx11abiFALSE; the CUDA 12.x runtime is driver-forward-compatible).
run $PIP install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0.post1/flash_attn-2.6.0.post1+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
# unimol-tools is a CUDA-using package by XPert.
run $PIP install unimol-tools==0.1.4.post1
# unimol-tools downgrades pandas; restore the pinned version.
run $PIP install "pandas==2.3.0"

echo "=== Stage 5/5: required helper deps + optional sub-script deps ==="

# 5a. REQUIRED — transformers + sentencepiece for prepare.py
#     (experiments/main_benchmark/data_prep/prepare_pertdit_embeddings.py
#      uses MolT5 + BioLinkBERT to compute PertDiT drug embeddings).
# Pin transformers to 4.36.x: newer (>=4.40) require torch >= 2.4 for
# torch.utils._pytree.register_pytree_node, but this env is torch 2.1.0.
run $PIP install "transformers==4.36.2" "sentencepiece==0.2.0"   # sentencepiece >= 0.2.1 only ships manylinux_2_28 wheels; 0.2.0 still has manylinux_2_17 (works on glibc >= 2.17)

# 5b. REQUIRED — zenodo_get for reference/fetch_upstream.sh
#     (downloads MultiDCP / PRnet / XPert datasets from Zenodo).
run $PIP install "zenodo_get>=1.5"

# 5c. OPTIONAL — used by per-model sub-scripts that the dpb training path does
# not trigger. Made non-blocking: failure here doesn't abort install.
#   wandb pinned <0.18 because wandb >=0.21 only ships manylinux_2_28 wheels;
#   on glibc < 2.28 systems pip falls back to the sdist, which needs Go to
#   compile wandb-core. wandb 0.17.x predates the Go-required split and ships
#   plain py3 wheels that work on any glibc >= 2.17 (CentOS 7+, RHEL 7+).
echo "[$(date +%H:%M:%S)] (optional) wandb / apscheduler / cmapPy"
$PIP install "wandb<0.18" apscheduler cmapPy 2>&1 | tee -a "$LOG" \
    || echo "WARNING: optional sub-script packages failed to install — training path unaffected (see $LOG)."

# === HDF5 file-lock workaround for lustre / NFS / network filesystems ===
# Default HDF5 file locking is not supported by lustre and some NFS mounts.
# Without HDF5_USE_FILE_LOCKING=FALSE, h5ad writes either (a) silently produce
# 0-byte files (anndata swallows the error and returns) or (b) raise
# BlockingIOError mid-write. Both have been observed on UCT CHPC and other
# academic HPC clusters. We install a conda activate.d hook so the env var is
# set automatically every time the env is activated.
ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/hdf5_lock.sh" << 'EOF'
# Disable HDF5 file locking — required on lustre, NFS, and some network FS.
# Safe everywhere else; this codebase does not run concurrent writers against
# the same h5ad file.
export HDF5_USE_FILE_LOCKING=FALSE
EOF
echo "[$(date +%H:%M:%S)] HDF5_USE_FILE_LOCKING=FALSE → $ACTIVATE_DIR/hdf5_lock.sh (auto-loaded on env activation)"

echo "" >> "$LOG"
echo "## Final pip freeze" >> "$LOG"
echo "\`\`\`" >> "$LOG"
$PIP freeze >> "$LOG"
echo "\`\`\`" >> "$LOG"
echo ""
echo "Install complete. See environment/INSTALL_LOG.md for full transcript."
echo ""
echo "IMPORTANT: to pick up the HDF5_USE_FILE_LOCKING=FALSE hook in your CURRENT"
echo "shell, run:    conda deactivate && conda activate benchmark"
echo "(future shells will pick it up automatically.)"
echo ""
echo "Quick verification:"
echo "  for M in mlp ciger deepce multidcp pertdit prnet transigen; do"
echo "    $PY experiments/_shared/training/train_\$M.py --help > /dev/null 2>&1 && echo \"\$M ✓\" || echo \"\$M ✗\""
echo "  done"
