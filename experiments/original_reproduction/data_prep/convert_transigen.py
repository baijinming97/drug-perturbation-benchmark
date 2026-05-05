"""
Convert TranSiGen original HDF5 to the unified h5ad schema.

Input:  TranSiGen processed_data.h5 (78,569 samples) + KPGT_emb2304.pickle
Output: unified-schema h5ad + idx2smi.npy + idx2kpgt.npy
        + 3 split columns (smiles_split_{seed} for seeds 364039/364040/364041)

Replicates TranSiGen's getSplitsByGroupKFold(canonical_smiles, 5, shuffle=True, random_state=seed).
"""

import argparse
import os
import pickle
import sys

from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import sklearn.utils
from sklearn.model_selection import GroupKFold


REPO_ROOT = Path(__file__).resolve().parents[3]
TRANSIGEN_DATA = str(REPO_ROOT / "data" / "TranSiGen" / "LINCS2020")
H5_PATH = os.path.join(TRANSIGEN_DATA, "processed_data.h5")
KPGT_PATH = os.path.join(TRANSIGEN_DATA, "KPGT_emb2304.pickle")
OUT_DIR = str(REPO_ROOT / "data" / "_converted" / "transigen")

SEEDS = [364039, 364040, 364041]


def get_splits_by_group_kfold(groups, n_splits, random_state):
    """Exact replica of TranSiGen utils.getSplitsByGroupKFold."""
    kf = GroupKFold(n_splits=n_splits)
    unique_groups = np.unique(groups)
    rnd_renames = sklearn.utils.shuffle(
        np.arange(len(unique_groups)), random_state=random_state)
    groups_renamed = np.array(
        [rnd_renames[np.argwhere(unique_groups == g)[0][0]] for g in groups])
    kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups_renamed)
    folds = [list(x[1]) for x in kfsplit]
    tr_fold_nums = list(range(len(folds)))[:-2]
    ind_tr = sum([folds[i] for i in tr_fold_nums], [])
    ind_va = folds[-2]
    ind_te = folds[-1]
    return ind_tr, ind_va, ind_te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", type=str, default=H5_PATH)
    p.add_argument("--kpgt_path", type=str, default=KPGT_PATH)
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TranSiGen original → unified-schema h5ad conversion")
    print("=" * 60)

    # ── Load original HDF5 ──────────────────────────────────────────────────
    print("\n[1/5] Loading original HDF5 ...")
    with h5py.File(args.h5_path, "r") as f:
        x1 = f["x1"][:]  # control expression (N, 978)
        x2 = f["x2"][:]  # treated expression (N, 978)
        smiles_raw = f["canonical_smiles"][:]
        cid_raw = f["cid"][:]
        sig_raw = f["sig"][:]

    smiles_all = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in smiles_raw])
    cid_all = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cid_raw])
    sig_all = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in sig_raw])
    n_samples = len(smiles_all)
    print(f"  Samples: {n_samples}")
    print(f"  x1 (control): {x1.shape}, x2 (treated): {x2.shape}")
    print(f"  Unique SMILES: {len(set(smiles_all))}")
    print(f"  Unique cells: {len(set(cid_all))}")

    # ── Build pert_idx + idx2smi ────────────────────────────────────────────
    print("\n[2/5] Building pert_idx and idx2smi ...")
    unique_smiles = sorted(set(smiles_all))
    smi2idx = {s: i for i, s in enumerate(unique_smiles)}
    idx2smi = {i: s for s, i in smi2idx.items()}
    pert_idx = np.array([smi2idx[s] for s in smiles_all], dtype=np.int64)
    print(f"  Unique drugs: {len(unique_smiles)}")

    # Cell index
    unique_cells = sorted(set(cid_all))
    cell2idx = {c: i for i, c in enumerate(unique_cells)}
    cell_idx = np.array([cell2idx[c] for c in cid_all], dtype=np.int64)
    print(f"  Unique cell types: {len(unique_cells)}")

    # ── Build KPGT dict keyed by pert_idx ───────────────────────────────────
    print("\n[3/5] Building KPGT dict ...")
    with open(args.kpgt_path, "rb") as f:
        kpgt_smi = pickle.load(f)
    print(f"  Original KPGT keys: {len(kpgt_smi)}")

    idx2kpgt = {}
    missing_kpgt = []
    for idx, smi in idx2smi.items():
        if smi in kpgt_smi:
            idx2kpgt[idx] = np.array(kpgt_smi[smi], dtype=np.float32)
        else:
            missing_kpgt.append(smi)
            idx2kpgt[idx] = np.zeros(2304, dtype=np.float32)

    if missing_kpgt:
        print(f"  WARNING: {len(missing_kpgt)} drugs missing KPGT embeddings (zeroed)")
    else:
        print(f"  All {len(idx2kpgt)} drugs have KPGT embeddings")

    # ── Apply GroupKFold splits for each seed ───────────────────────────────
    print("\n[4/5] Computing splits ...")
    split_cols = {}
    for seed in SEEDS:
        col_name = f"smiles_split_{seed}"
        labels = np.full(n_samples, "", dtype=object)
        ind_tr, ind_va, ind_te = get_splits_by_group_kfold(
            smiles_all, n_splits=5, random_state=seed)
        labels[ind_tr] = "train"
        labels[ind_va] = "valid"
        labels[ind_te] = "test"
        split_cols[col_name] = labels
        n_tr = len(ind_tr)
        n_va = len(ind_va)
        n_te = len(ind_te)
        n_drugs_te = len(set(smiles_all[ind_te]))
        print(f"  {col_name}: train={n_tr}, valid={n_va}, test={n_te} "
              f"(test drugs={n_drugs_te})")

    # ── Assemble unified-schema h5ad ─────────────────────────────────────────
    print("\n[5/5] Assembling unified h5ad ...")
    obs_dict = {
        "pert_idx": pert_idx,
        "cell_idx": pd.Categorical(cell_idx),
        "cell_mfc_name": cid_all,
        "SMILES": smiles_all,
    }
    for col_name, labels in split_cols.items():
        obs_dict[col_name] = labels

    obs_df = pd.DataFrame(obs_dict)
    obs_df.index = [str(i) for i in range(len(obs_df))]

    adata = anndata.AnnData(
        X=x2.astype(np.float32),
        obs=obs_df,
        obsm={"X_ctl": x1.astype(np.float32)},
    )

    h5ad_path = os.path.join(args.output_dir, "transigen_original.h5ad")
    smi_path = os.path.join(args.output_dir, "idx2smi.npy")
    kpgt_path = os.path.join(args.output_dir, "idx2kpgt.npy")

    adata.write_h5ad(h5ad_path)
    # Defensive check: lustre/NFS HDF5 file-locking failures can leave a 0-byte
    # file with anndata silently swallowing the error. Verify the write took.
    if not os.path.exists(h5ad_path) or os.path.getsize(h5ad_path) == 0:
        raise RuntimeError(
            f"h5ad write produced empty/missing file: {h5ad_path}. "
            "On lustre/NFS, set HDF5_USE_FILE_LOCKING=FALSE."
        )
    np.save(smi_path, idx2smi)
    np.save(kpgt_path, idx2kpgt)

    print(f"\n  Saved: {h5ad_path} ({adata.shape})")
    print(f"  Saved: {smi_path} ({len(idx2smi)} drugs)")
    print(f"  Saved: {kpgt_path} ({len(idx2kpgt)} drugs)")
    print(f"  x2 range: [{x2.min():.3f}, {x2.max():.3f}]")
    print(f"  x1 range: [{x1.min():.3f}, {x1.max():.3f}]")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
