"""Shared pre-flight setup steps. Each `step_*` is idempotent and returns a
short status string. Task-specific `experiments/<task>/prepare.py` files import
the subset of steps they need.

Conventions:
  - Status strings start with one of: SKIP / DONE / OK / NEEDS-* / FAIL
  - Steps that rely on external data (Zenodo / Tsinghua) tell the user to run
    `reference/fetch_upstream.sh` rather than re-downloading.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]


# ────────────────────────────────────────────────────────────────────────────
# Core paths (defined once)
# ────────────────────────────────────────────────────────────────────────────

# A. L1000_sdst h5ad + auxiliary npy (used by all 6 non-XPert models)
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "XPert" / "processed_data"
PROCESSED_DATA_ZIP = REPO_ROOT / "data" / "XPert" / "processed_data.zip"
PROCESSED_DATA_PROBE = PROCESSED_DATA_DIR / "l1000_sdst_78453.h5ad"

# A2. XPert UniMol drug array (Figshare-distributed separately from processed_data.zip;
#     ~4.2 GB single npy used by config_l1000.yaml's drug_unimol_path).
XPERT_UNIMOL_ARR_ZIP = REPO_ROOT / "data" / "XPert" / "figshare_28955141" / "all_drugs_unimol_arr.zip"
XPERT_UNIMOL_ARR_NPY = PROCESSED_DATA_DIR / "all_drugs_unimol_arr.npy"

# B. CIGER gene_feature_nm.csv (5 gene names re-mapped to modern HGNC)
CIGER_GENE_UPSTREAM = REPO_ROOT / "data" / "CIGER" / "gene_feature.csv"
CIGER_GENE_NM = REPO_ROOT / "data" / "CIGER" / "gene_feature_nm.csv"
GENE_RENAMES = {
    "KIAA0907": "KHDC4",
    "PAPD7":    "TENT4A",
    "IKBKAP":   "ELP1",
    "TMEM5":    "RXYLT1",
    "HDGFRP3":  "HDGFL3",
}

# C. TranSiGen pretrained VAE (shipped by upstream)
TRANSIGEN_VAE_X1 = REPO_ROOT / "reference" / "TranSiGen" / "results" / "trained_model_shRNA_vae_x1" / "best_model.pt"
TRANSIGEN_VAE_X2 = REPO_ROOT / "reference" / "TranSiGen" / "results" / "trained_model_shRNA_vae_x2" / "best_model.pt"


# ────────────────────────────────────────────────────────────────────────────
# Step functions — each idempotent, all share the same signature:
#   step(force: bool, check_only: bool) -> str
# ────────────────────────────────────────────────────────────────────────────

def step_extract_processed_data(*, force: bool = False, check_only: bool = False) -> str:
    """Unzip data/XPert/processed_data.zip into data/XPert/processed_data/."""
    if PROCESSED_DATA_PROBE.exists() and not force:
        return f"SKIP — {PROCESSED_DATA_PROBE.relative_to(REPO_ROOT)} already exists"
    if check_only:
        return f"NEEDS-EXTRACT — {PROCESSED_DATA_ZIP.relative_to(REPO_ROOT)}"
    if not PROCESSED_DATA_ZIP.exists():
        return f"FAIL — zip missing: {PROCESSED_DATA_ZIP.relative_to(REPO_ROOT)} (run `reference/fetch_upstream.sh XPert` first)"
    if PROCESSED_DATA_DIR.is_symlink():
        PROCESSED_DATA_DIR.unlink()
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(PROCESSED_DATA_ZIP) as z:
        z.extractall(PROCESSED_DATA_DIR.parent)
    return f"DONE — extracted to {PROCESSED_DATA_DIR.relative_to(REPO_ROOT)}/"


def step_extract_xpert_unimol_arr(*, force: bool = False, check_only: bool = False) -> str:
    """Unzip data/XPert/figshare_28955141/all_drugs_unimol_arr.zip into data/XPert/processed_data/.

    XPert's config_l1000.yaml expects ``drug_unimol_path: data/XPert/processed_data/all_drugs_unimol_arr.npy``.
    The 4.2 GB array is Figshare-distributed in its own zip (NOT inside processed_data.zip)
    and must be extracted explicitly.
    """
    if XPERT_UNIMOL_ARR_NPY.exists() and not force:
        return f"SKIP — {XPERT_UNIMOL_ARR_NPY.relative_to(REPO_ROOT)} already exists"
    if check_only:
        return f"NEEDS-EXTRACT — {XPERT_UNIMOL_ARR_ZIP.relative_to(REPO_ROOT)}"
    if not XPERT_UNIMOL_ARR_ZIP.exists():
        return f"FAIL — zip missing: {XPERT_UNIMOL_ARR_ZIP.relative_to(REPO_ROOT)} (run `reference/fetch_upstream.sh XPert` first)"
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(XPERT_UNIMOL_ARR_ZIP) as z:
        z.extractall(PROCESSED_DATA_DIR)
    return f"DONE — extracted {XPERT_UNIMOL_ARR_NPY.relative_to(REPO_ROOT)}"


def step_generate_gene_feature_nm(*, force: bool = False, check_only: bool = False) -> str:
    """CIGER needs gene_feature_nm.csv: same data as upstream gene_feature.csv,
    but 5 outdated gene names (KIAA0907 etc.) renamed to modern HGNC symbols."""
    if CIGER_GENE_NM.exists() and not force:
        return f"SKIP — {CIGER_GENE_NM.relative_to(REPO_ROOT)} already exists"
    if check_only:
        return f"NEEDS-GEN — rename {len(GENE_RENAMES)} genes in {CIGER_GENE_UPSTREAM.name}"
    if not CIGER_GENE_UPSTREAM.exists():
        return f"FAIL — upstream missing: {CIGER_GENE_UPSTREAM.relative_to(REPO_ROOT)} (run `reference/fetch_upstream.sh CIGER` first)"
    n_lines = n_renamed = 0
    with CIGER_GENE_UPSTREAM.open() as fin, CIGER_GENE_NM.open("w") as fout:
        for line in fin:
            n_lines += 1
            head, _, rest = line.partition(",")
            new_head = GENE_RENAMES.get(head, head)
            if new_head != head:
                n_renamed += 1
            fout.write(f"{new_head},{rest}" if rest else line)
    return f"DONE — wrote {n_lines} rows ({n_renamed} renamed) to {CIGER_GENE_NM.relative_to(REPO_ROOT)}"


def step_verify_transigen_vae(**_) -> str:
    """TranSiGen training requires two pretrained VAE checkpoints from upstream."""
    missing = [p for p in (TRANSIGEN_VAE_X1, TRANSIGEN_VAE_X2) if not p.exists()]
    if not missing:
        return "OK — both TranSiGen VAE checkpoints present"
    return f"FAIL — missing: {[str(p.relative_to(REPO_ROOT)) for p in missing]} (run `reference/fetch_upstream.sh TranSiGen` first)"


def step_generate_pertdit_embeddings(*, force: bool = False, check_only: bool = False) -> str:
    """PertDiT needs MolT5 + BioLinkBERT embeddings of all 8981 SMILES (~3.7 GB pkl).
    Generates `data/PertDiT/bench_drug_emb/bench_drug_emb.pkl` + `bench_dose_emb.pt`.
    Heavy step: ~25 min on a100 + ~2 GB HuggingFace model download (first time)."""
    out = REPO_ROOT / "data" / "PertDiT" / "bench_drug_emb" / "bench_drug_emb.pkl"
    if out.exists() and not force:
        return f"SKIP — {out.relative_to(REPO_ROOT)} already exists"
    if check_only:
        return f"NEEDS-GEN — run prepare_pertdit_embeddings.py (~25 min on GPU)"
    script = REPO_ROOT / "experiments" / "main_benchmark" / "data_prep" / "prepare_pertdit_embeddings.py"
    if not script.exists():
        return f"FAIL — generator missing: {script.relative_to(REPO_ROOT)}"
    import subprocess, sys
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        return f"FAIL — generator exited rc={e.returncode}"
    return f"DONE — wrote {out.relative_to(REPO_ROOT)}"


def step_extract_multidcp_data(*, force: bool = False, check_only: bool = False) -> str:
    """Untar data/MultiDCP/data.tar.gz (Zenodo, ~11 GB → ~30 GB extracted)."""
    extracted = REPO_ROOT / "data" / "MultiDCP" / "extracted"
    probe = extracted / "data"
    if probe.exists() and not force:
        return f"SKIP — {probe.relative_to(REPO_ROOT)}/ already extracted"
    if check_only:
        return f"NEEDS-EXTRACT — data/MultiDCP/data.tar.gz (~11 GB → 30 GB, ~5 min)"
    tar_path = REPO_ROOT / "data" / "MultiDCP" / "data.tar.gz"
    if not tar_path.exists():
        return f"FAIL — {tar_path.relative_to(REPO_ROOT)} missing (run reference/fetch_upstream.sh MultiDCP)"
    extracted.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(extracted)], check=True)
    return f"DONE — extracted to {extracted.relative_to(REPO_ROOT)}/"


def step_extract_pertdit_data(*, force: bool = False, check_only: bool = False) -> str:
    """Unrar data/PertDiT/lincs_l1000.h5ad (named .h5ad but actually a RAR archive)."""
    extracted = REPO_ROOT / "data" / "PertDiT" / "extracted"
    probe = extracted / "lincs_adata.h5ad"
    if probe.exists() and not force:
        return f"SKIP — {probe.relative_to(REPO_ROOT)} already extracted"
    if check_only:
        return f"NEEDS-EXTRACT — data/PertDiT/lincs_l1000.h5ad (RAR ~8.9 GB → 13 GB)"
    rar_path = REPO_ROOT / "data" / "PertDiT" / "lincs_l1000.h5ad"
    if not rar_path.exists():
        return f"FAIL — {rar_path.relative_to(REPO_ROOT)} missing (run reference/fetch_upstream.sh PertDiT)"
    extracted.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run(["unrar", "x", "-o+", str(rar_path), str(extracted) + "/"], check=True)
    return f"DONE — extracted to {extracted.relative_to(REPO_ROOT)}/"


# Each convert_<M>.py produces multiple files; SKIP must verify ALL of them
# exist AND are non-empty. (a) Mid-run interruption can leave .h5ad written
# but idx2smi.npy missing — checking only one file leads to a false SKIP.
# (b) On lustre/NFS, HDF5 file-locking failures can leave .h5ad as a 0-byte
# file with the script silently exiting 0 — checking existence alone is not
# enough; size must also be > 0.
CONVERT_OUTPUTS = {
    "ciger":     ["ciger_original.h5ad",     "idx2smi.npy", "gene_feature_p6.csv"],
    "deepce":    ["deepce_original.h5ad",    "idx2smi.npy"],
    "multidcp":  ["multidcp_original.h5ad",  "idx2smi.npy"],
    "pertdit":   ["p6_drug_emb.pkl",         "p6_dose_emb.pkl"],
    "prnet":     ["prnet_original.h5ad",     "idx2smi.npy"],
    "transigen": ["transigen_original.h5ad", "idx2smi.npy", "idx2kpgt.npy"],
}


def step_convert_per_model(model: str, *, force: bool = False, check_only: bool = False) -> str:
    """Run convert_<model>.py to produce per-model adapted h5ad in data/_converted/<model>/."""
    out_dir = REPO_ROOT / "data" / "_converted" / model
    expected = [out_dir / name for name in CONVERT_OUTPUTS[model]]
    missing = [p.name for p in expected if not p.exists() or p.stat().st_size == 0]
    if not missing and not force:
        return f"SKIP — {len(expected)} outputs already present in {out_dir.relative_to(REPO_ROOT)}/"
    if check_only:
        return f"NEEDS-CONV — convert_{model}.py → data/_converted/{model}/ (missing/empty: {missing})"
    script = REPO_ROOT / "experiments" / "original_reproduction" / "data_prep" / f"convert_{model}.py"
    if not script.exists():
        return f"FAIL — {script.relative_to(REPO_ROOT)} missing"
    out_dir.mkdir(parents=True, exist_ok=True)
    import subprocess, sys
    # Each convert_<M>.py has its own --output_dir + model-specific input flags.
    cmd = [sys.executable, str(script), "--output_dir", str(out_dir)]
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        return f"FAIL — convert exited rc={e.returncode}"
    # Post-run verification: catch silent HDF5-lock failures on lustre/NFS.
    still_missing = [p.name for p in expected if not p.exists() or p.stat().st_size == 0]
    if still_missing:
        return (f"FAIL — convert_{model}.py finished rc=0 but output incomplete: {still_missing}. "
                f"On lustre/NFS, set HDF5_USE_FILE_LOCKING=FALSE (or reactivate the conda env).")
    return f"DONE — wrote {len(expected)} outputs to {out_dir.relative_to(REPO_ROOT)}/"


def step_generate_original_drug_splits(*, force: bool = False, check_only: bool = False) -> str:
    """In-place add drug_split_0..4 to deepce/multidcp/transigen converted h5ad.
    KFold(5, shuffle=True, random_state=42) over unique drugs.
    CIGER, PertDiT, PRnet already have these columns from their convert_<M>.py."""
    targets = ["deepce", "multidcp", "transigen"]
    missing = []
    for m in targets:
        h5 = REPO_ROOT / "data" / "_converted" / m / f"{m}_original.h5ad"
        if not h5.exists():
            return f"FAIL — {h5.relative_to(REPO_ROOT)} missing (run convert_{m}.py first)"
        try:
            import anndata as ad
            a = ad.read_h5ad(h5, backed="r")
            if not all(f"drug_split_{k}" in a.obs.columns for k in range(5)):
                missing.append(m)
        except Exception as e:
            return f"FAIL — could not read {h5.relative_to(REPO_ROOT)}: {e}"
    if not missing and not force:
        return f"SKIP — drug_split_0..4 already present in all 3 h5ad"
    if check_only:
        return f"NEEDS-GEN — KFold split for {missing}"
    script = REPO_ROOT / "experiments" / "original_ablation" / "data_prep" / "prepare_splits.py"
    import subprocess, sys
    cmd = [sys.executable, str(script)] + (["--force"] if force else []) + ["--models"] + missing
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        return f"FAIL — prepare_splits.py rc={e.returncode}"
    return f"DONE — added drug_split_0..4 to {missing}"


def step_convert_ciger(**kw):     return step_convert_per_model("ciger", **kw)
def step_convert_deepce(**kw):    return step_convert_per_model("deepce", **kw)
def step_convert_multidcp(**kw):  return step_convert_per_model("multidcp", **kw)
def step_convert_pertdit(**kw):   return step_convert_per_model("pertdit", **kw)
def step_convert_prnet(**kw):     return step_convert_per_model("prnet", **kw)
def step_convert_transigen(**kw): return step_convert_per_model("transigen", **kw)


def step_verify_split_cols_drug_blind(**_) -> str:
    """h5ad obs must contain nm_drug_blind_1..5 (used by main_benchmark, main_ablation, mlp_baseline, original_ablation)."""
    return _verify_h5ad_cols([f"nm_drug_blind_{i}" for i in range(1, 6)])


def step_generate_drug_blind_splits(*, force: bool = False, check_only: bool = False) -> str:
    """In-place add nm_drug_blind_1..5 to data/XPert/processed_data/l1000_sdst_78453.h5ad.

    5-fold rotation over unique drugs (seed 131419). Subprocess-runs
    experiments/main_benchmark/data_prep/create_splits.py.
    """
    h5 = PROCESSED_DATA_PROBE
    if not h5.exists():
        return f"FAIL — {h5.relative_to(REPO_ROOT)} missing (run step_extract_processed_data first)"
    needed = [f"nm_drug_blind_{i}" for i in range(1, 6)]
    # Peek at obs column names via h5py (released before subprocess).
    # NB: do NOT use anndata backed-mode here — it keeps an open file handle
    # which collides with the subprocess's exclusive write lock on NFS.
    try:
        import h5py
        with h5py.File(h5, "r") as f:
            obs_cols = set(f["obs"].keys()) if "obs" in f else set()
    except Exception as e:
        return f"FAIL — could not read {h5.relative_to(REPO_ROOT)}: {e}"
    missing = [c for c in needed if c not in obs_cols]
    if not missing and not force:
        return "SKIP — nm_drug_blind_1..5 already present"
    if check_only:
        return f"NEEDS-GEN — {len(missing)}/5 drug-blind split columns missing"
    script = REPO_ROOT / "experiments" / "main_benchmark" / "data_prep" / "create_splits.py"
    import subprocess, sys
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        return f"FAIL — create_splits.py rc={e.returncode}"
    return "DONE — added nm_drug_blind_1..5 to h5ad"


# ────────────────────────────────────────────────────────────────────────────
# XPert ablation arrays — pre-compute zero/shuffle versions of UniMol +
# HG drug embeddings to match XPert's existing config_l1000_ablation_*.yaml
# convention. Lets the unified-pipeline shim run XPert with --ablation_mode
# zero/shuffle without modifying XPert source.
# ────────────────────────────────────────────────────────────────────────────

XPERT_UNIMOL_SRC = REPO_ROOT / "data" / "XPert" / "processed_data" / "all_drugs_unimol_arr.npy"
XPERT_HG_REFERENCE_DIR = REPO_ROOT / "reference" / "XPert" / "HG_data"
XPERT_HG_SRC = XPERT_HG_REFERENCE_DIR / "saved_embedding" / "HG_drug_embeddings.npy"
XPERT_ABL_DIR = REPO_ROOT / "data" / "_xpert_ablation"
XPERT_ABL_SEED = 131419  # matches the seed convention used by the other 6 train_<M>.py


def step_verify_xpert_hg_data(*, force: bool = False, check_only: bool = False) -> str:
    """Verify reference/XPert/HG_data/ is populated.

    XPert YAML configs reference this path (hg_path: reference/XPert/HG_data/).
    The directory ships with the XPert upstream repo and is fetched by
    `reference/fetch_upstream.sh XPert`. force/check_only are accepted for
    STEPS interface compatibility but ignored.
    """
    if XPERT_HG_SRC.exists():
        return f"OK — {XPERT_HG_SRC.relative_to(REPO_ROOT)} present"
    return f"FAIL — {XPERT_HG_SRC.relative_to(REPO_ROOT)} missing (run reference/fetch_upstream.sh XPert)"


def step_generate_xpert_ablation_data(*, force: bool = False, check_only: bool = False) -> str:
    """Pre-compute zero/shuffle versions of XPert UniMol + HG drug arrays.

    Outputs (in data/_xpert_ablation/, used by config_l1000_ablation_*.yaml):
      drugs_unimol_zero_all.npy        zeros_like(unimol_arr)
      drugs_unimol_shuffle.npy         unimol_arr permuted along drug axis
      HG_drug_embeddings_zero.npy      zeros_like(hg_emb)
      HG_drug_embeddings_shuffle.npy   hg_emb permuted along drug axis
    Shuffle uses seed=131419 to match other models' ablation_mode convention.
    """
    targets = {
        "drugs_unimol_zero_all.npy": (XPERT_UNIMOL_SRC, "zero"),
        "drugs_unimol_shuffle.npy": (XPERT_UNIMOL_SRC, "shuffle"),
        "HG_drug_embeddings_zero.npy": (XPERT_HG_SRC, "zero"),
        "HG_drug_embeddings_shuffle.npy": (XPERT_HG_SRC, "shuffle"),
    }
    out_paths = [XPERT_ABL_DIR / name for name in targets]
    if not force and all(p.exists() for p in out_paths):
        return "SKIP — all 4 XPert ablation arrays present"
    if check_only:
        missing = [p.name for p in out_paths if not p.exists()]
        return f"NEEDS-WORK — would generate: {missing}"

    if not XPERT_UNIMOL_SRC.exists():
        return f"FAIL — missing source: {XPERT_UNIMOL_SRC}"
    if not XPERT_HG_SRC.exists():
        return f"FAIL — missing source: {XPERT_HG_SRC}"

    try:
        import numpy as np
    except ImportError:
        return "SKIP — numpy not in current env"

    XPERT_ABL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(XPERT_ABL_SEED)

    # unimol_arr shape: (n_drugs, max_atoms=122, feat_dim) — large file, load once
    unimol = np.load(XPERT_UNIMOL_SRC, allow_pickle=True)
    perm_unimol = rng.permutation(unimol.shape[0])
    np.save(XPERT_ABL_DIR / "drugs_unimol_zero_all.npy", np.zeros_like(unimol))
    np.save(XPERT_ABL_DIR / "drugs_unimol_shuffle.npy", unimol[perm_unimol])
    del unimol

    rng = np.random.RandomState(XPERT_ABL_SEED)  # reset for HG
    hg = np.load(XPERT_HG_SRC, allow_pickle=True)
    perm_hg = rng.permutation(hg.shape[0])
    np.save(XPERT_ABL_DIR / "HG_drug_embeddings_zero.npy", np.zeros_like(hg))
    np.save(XPERT_ABL_DIR / "HG_drug_embeddings_shuffle.npy", hg[perm_hg])

    return f"DONE — wrote 4 arrays in {XPERT_ABL_DIR.relative_to(REPO_ROOT)}/ (seed={XPERT_ABL_SEED})"


def step_verify_split_cols_scaffold(**_) -> str:
    """h5ad obs must contain nm_scaffold_1..5 (used by scaffold_generalization)."""
    return _verify_h5ad_cols([f"nm_scaffold_{i}" for i in range(1, 6)])


def step_generate_scaffold_splits(*, force: bool = False, check_only: bool = False) -> str:
    """In-place add nm_scaffold_1..5 to data/XPert/processed_data/l1000_sdst_78453.h5ad.

    Bemis-Murcko scaffold-disjoint 5-fold (seed 131419). Subprocess-runs
    experiments/scaffold_generalization/data_prep/create_scaffold_splits.py.
    Requires data/XPert/processed_data/all_drugs_idx2smi_8981.npy (extracted by step A).
    """
    h5 = PROCESSED_DATA_PROBE
    if not h5.exists():
        return f"FAIL — {h5.relative_to(REPO_ROOT)} missing (run step_extract_processed_data first)"
    idx2smi = REPO_ROOT / "data" / "XPert" / "processed_data" / "all_drugs_idx2smi_8981.npy"
    if not idx2smi.exists():
        return f"FAIL — {idx2smi.relative_to(REPO_ROOT)} missing (extracted by step_extract_processed_data)"
    needed = [f"nm_scaffold_{i}" for i in range(1, 6)]
    # See step_generate_drug_blind_splits: h5py-only peek so the parent
    # releases the file handle before the subprocess takes the write lock.
    try:
        import h5py
        with h5py.File(h5, "r") as f:
            obs_cols = set(f["obs"].keys()) if "obs" in f else set()
    except Exception as e:
        return f"FAIL — could not read {h5.relative_to(REPO_ROOT)}: {e}"
    missing = [c for c in needed if c not in obs_cols]
    if not missing and not force:
        return "SKIP — nm_scaffold_1..5 already present"
    if check_only:
        return f"NEEDS-GEN — {len(missing)}/5 scaffold split columns missing"
    script = REPO_ROOT / "experiments" / "scaffold_generalization" / "data_prep" / "create_scaffold_splits.py"
    import subprocess, sys
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        return f"FAIL — create_scaffold_splits.py rc={e.returncode}"
    return "DONE — added nm_scaffold_1..5 to h5ad"


def _verify_h5ad_cols(needed: list[str]) -> str:
    if not PROCESSED_DATA_PROBE.exists():
        return "SKIP — h5ad not extracted yet (run step_extract_processed_data first)"
    try:
        import anndata as ad
    except ImportError:
        return "SKIP — anndata not in current env (need 'benchmark' env)"
    a = ad.read_h5ad(PROCESSED_DATA_PROBE, backed="r")
    missing = [c for c in needed if c not in a.obs.columns]
    if not missing:
        return f"OK — all {len(needed)} required cols present"
    return f"FAIL — missing cols: {missing}"


# ────────────────────────────────────────────────────────────────────────────
# Driver — each per-task prepare.py calls run_steps() with its subset
# ────────────────────────────────────────────────────────────────────────────

def run_steps(steps: list[tuple[str, Callable]], force: bool = False, check_only: bool = False) -> bool:
    """Execute a list of (label, step_fn) pairs, print results, return False if any FAIL."""
    print(f"  repo root: {REPO_ROOT}")
    print(f"  mode     : {'check-only' if check_only else ('force' if force else 'normal')}")
    print()
    fail = False
    for label, fn in steps:
        msg = fn(force=force, check_only=check_only)
        status = msg.split(" — ", 1)[0]
        detail = msg.split(" — ", 1)[1] if " — " in msg else ""
        print(f"  [{status:>13}]  {label}")
        if detail:
            print(f"                  {detail}")
        if status == "FAIL":
            fail = True
    print()
    return not fail


def add_argparse_options(parser):
    """Add --force / --check-only to a task's prepare.py argparser."""
    parser.add_argument("--force", action="store_true", help="Re-run all steps regardless of state")
    parser.add_argument("--check-only", action="store_true", help="Print status but make no changes")
