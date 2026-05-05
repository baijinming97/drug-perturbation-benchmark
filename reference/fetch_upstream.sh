#!/usr/bin/env bash
# reference/fetch_upstream.sh
#
# Idempotently re-fetch every upstream code repo and original data file
# referenced in reference/PROVENANCE.md, populating:
#   reference/<M>/   pristine git clone @ pinned commit (.git stripped)
#   data/<M>/        original data (Zenodo / Tsinghua / Figshare / repo-bundled)
#
# DESIGN
#   - Idempotent: anything already present is skipped (won't overwrite).
#   - Network: HTTPS only (port 443). SSH (22) NOT required.
#   - Heavy downloads (PertDiT 8.9 GB, MultiDCP 11 GB, XPert 14 GB) — run on a
#     machine with enough disk; on this HPC we used ada-partition sbatch jobs.
#
# REQUIREMENTS
#   git, curl, python (with `zenodo_get` package), wget
#   For post-processing (NOT done here, see README §3): unrar, tar, unzip
#
# USAGE
#   bash reference/fetch_upstream.sh           # do everything missing
#   bash reference/fetch_upstream.sh CIGER     # only one model
#   DRY_RUN=1 bash reference/fetch_upstream.sh # print actions, do nothing
#
# AFTER THIS SCRIPT RUNS
#   See README §3 for required post-processing:
#     - PertDiT: rename .h5ad → .rar, unrar
#     - MultiDCP/XPert: untar / unzip the archives
#

set -euo pipefail

# Resolve repo root regardless of caller's CWD ($script_dir/..).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Provenance table (single source of truth, mirrors reference/PROVENANCE.md) ---
declare -A REPO_URL=(
    [CIGER]="https://github.com/pth1993/CIGER.git"
    [DeepCE]="https://github.com/pth1993/DeepCE.git"
    [MultiDCP]="https://github.com/XieResearchGroup/MultiDCP.git"
    [PertDiT]="https://github.com/wangkekekeke/PertDiT.git"
    [PRnet]="https://github.com/Perturbation-Response-Prediction/PRnet.git"
    [TranSiGen]="https://github.com/myzhengSIMM/TranSiGen.git"
    [XPert]="https://github.com/GSanShui/XPert.git"
)
declare -A REPO_COMMIT=(
    [CIGER]="81c16f107cf957ca73840e6fab41de174c85ee8f"
    [DeepCE]="9b0d04fa920ef73df578b070cfcc3982effdde42"
    [MultiDCP]="36ecdef9598de703b61fe0d1b91bcfeeba844274"
    [PertDiT]="596d681f816d3184d7a0a63a133ce68d5838fae0"
    [PRnet]="f19174bde3ed2633f54c7831799cc38c4ffc7a0d"
    [TranSiGen]="8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a"
    [XPert]="d53ff497e465692c80fac7b899e3cc5cc8e7bfed"
)
declare -A DATA_PROBE=(
    # If this file/dir already exists under data/<M>/, the data step is skipped.
    [CIGER]="drug_smiles.csv"
    [DeepCE]="drugs_smiles.csv"
    [MultiDCP]="data.tar.gz"
    [PertDiT]="lincs_l1000.h5ad"   # actually a RAR; see README §3.1
    [PRnet]="Lincs_L1000.h5ad"
    [TranSiGen]="LINCS2020/KPGT_emb2304.pickle"
    [XPert]="processed_data.zip"
)

DRY_RUN="${DRY_RUN:-0}"
WANT="${1:-ALL}"

run() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $*"
    else
        echo "  + $*"
        eval "$@"
    fi
}

# --- 1. Clone (idempotent) ---------------------------------------------------
clone_pinned() {
    local M="$1" url="${REPO_URL[$1]}" commit="${REPO_COMMIT[$1]}"
    local dir="reference/$M"
    if [ -d "$dir" ] && [ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -not -name '.git' -print -quit 2>/dev/null)" ]; then
        echo "[SKIP clone]  $M  ← reference/$M already populated"
        return 0
    fi
    echo "[CLONE]       $M  @ $commit"
    run "rm -rf '$dir'"
    run "git clone --quiet '$url' '$dir'"
    run "git -C '$dir' checkout --quiet --detach '$commit'"
    run "rm -rf '$dir/.git'"
}

# --- 2. Data fetch (idempotent) ---------------------------------------------
fetch_data() {
    local M="$1"
    local probe="data/$M/${DATA_PROBE[$M]}"
    if [ -s "$probe" ]; then
        echo "[SKIP data]   $M  ← $probe already present"
        return 0
    fi
    echo "[FETCH data]  $M"
    run "mkdir -p 'data/$M'"
    case "$M" in
    CIGER)
        # bundled: copy from upstream clone
        run "cp -r 'reference/CIGER/CIGER/data/.' 'data/CIGER/'"
        ;;
    DeepCE)
        # bundled: copy from upstream clone
        run "cp -r 'reference/DeepCE/DeepCE/data/.' 'data/DeepCE/'"
        ;;
    MultiDCP)
        # Zenodo 10.5281/zenodo.5172809 (~11 GB → data.tar.gz)
        run "cd data/MultiDCP && python -m zenodo_get -d 10.5281/zenodo.5172809 && cd '$REPO_ROOT'"
        ;;
    PertDiT)
        # Tsinghua Cloud direct link (~8.9 GB; arrives as .h5ad named, actually .rar)
        # --no-check-certificate: some HPC CA bundles (e.g. CentOS 7) lack the Let's Encrypt R12/R13 intermediate certs used by cloud.tsinghua.edu.cn and release-assets.githubusercontent.com. Integrity is still verifiable via the SHA-256 in reference/PROVENANCE.md.
        run "wget -nv --no-check-certificate -O 'data/PertDiT/lincs_l1000.h5ad' 'https://cloud.tsinghua.edu.cn/f/7bca2e22c1f14c4db7db/?dl=1'"
        ;;
    PRnet)
        # Zenodo 10.5281/zenodo.14230870 (~4 GB: Lincs_L1000.h5ad + Sci_Plex.h5ad)
        run "cd data/PRnet && python -m zenodo_get -d 10.5281/zenodo.14230870 && cd '$REPO_ROOT'"
        ;;
    TranSiGen)
        # Two-part: (1) demo subset bundled in upstream `data/`, plus
        #           (2) full-training assets from upstream Release v1.0:
        #               processed_data.h5     652 MB  sha256 4701099...4b354e
        #               KPGT_emb2304.pickle    78 MB  sha256 61b2a13...62f4
        # Each step is independently idempotent.
        run "mkdir -p 'data/TranSiGen/LINCS2020'"
        if [ ! -s 'data/TranSiGen/LINCS2020/idx2smi.pickle' ]; then
            run "cp -r 'reference/TranSiGen/data/.' 'data/TranSiGen/'"
        else
            echo "  [skip] LINCS2020 demo bundle already copied"
        fi
        local rel_url="https://github.com/myzhengSIMM/TranSiGen/releases/download/v1.0"
        if [ ! -s 'data/TranSiGen/LINCS2020/processed_data.h5' ]; then
            run "wget -nv --no-check-certificate -O 'data/TranSiGen/LINCS2020/processed_data.h5' '$rel_url/processed_data.h5'"
        else
            echo "  [skip] processed_data.h5 already present"
        fi
        if [ ! -s 'data/TranSiGen/LINCS2020/KPGT_emb2304.pickle' ]; then
            run "wget -nv --no-check-certificate -O 'data/TranSiGen/LINCS2020/KPGT_emb2304.pickle' '$rel_url/KPGT_emb2304.pickle'"
        else
            echo "  [skip] KPGT_emb2304.pickle already present"
        fi
        ;;
    XPert)
        # Zenodo 10.5281/zenodo.15357711 (~14 GB) + Figshare 28955141 (~5 zips)
        run "cd data/XPert && python -m zenodo_get -d 10.5281/zenodo.15357711 && cd '$REPO_ROOT'"
        run "mkdir -p data/XPert/figshare_28955141"
        # Figshare API: list files, then download each by id
        local fs_json
        fs_json="$(curl -sfk https://api.figshare.com/v2/articles/28955141)" || {
            echo "  [warn] figshare API unreachable; skipping XPert figshare files" >&2
            return 0
        }
        echo "$fs_json" | python -c "
import json, sys
for f in json.load(sys.stdin)['files']:
    print(f\"{f['id']}\t{f['name']}\")" | while IFS=$'\t' read -r fid fname; do
            local out="data/XPert/figshare_28955141/$fname"
            if [ -s "$out" ]; then
                echo "  [skip] $fname already present"
            else
                run "curl -sLfk -o '$out' 'https://ndownloader.figshare.com/files/$fid'"
            fi
        done
        ;;
    *)
        echo "  [error] unknown model: $M" >&2
        return 1
        ;;
    esac
}

# --- Main loop ---------------------------------------------------------------
MODELS=(CIGER DeepCE MultiDCP PertDiT PRnet TranSiGen XPert)
if [ "$WANT" != "ALL" ]; then
    MODELS=("$WANT")
fi

for M in "${MODELS[@]}"; do
    if [ -z "${REPO_URL[$M]+x}" ]; then
        echo "[error] unknown model: $M (valid: CIGER DeepCE MultiDCP PertDiT PRnet TranSiGen XPert)" >&2
        exit 1
    fi
    echo
    echo "========== $M =========="
    clone_pinned "$M"
    fetch_data "$M"
done

echo
echo "All requested models processed."
echo "Post-processing (extraction) is documented in README §3."
