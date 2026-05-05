"""
Convert PRNet original h5ad to the unified h5ad schema.

Input:  PRNet Lincs_L1000.h5ad (883K rows, incl. controls, multi-dose)
Output: unified-schema h5ad (drug-treated only, X_ctl in obsm, split cols)
        + idx2smi.npy

Does NOT normalize — the training shim handles normalize_total + log1p.
"""

import argparse
import os
import sys

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[3]
PRNET_H5AD = str(REPO_ROOT / "data" / "PRnet" / "Lincs_L1000.h5ad")
OUT_DIR = str(REPO_ROOT / "data" / "_converted" / "prnet")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=PRNET_H5AD)
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PRNet original → unified-schema h5ad conversion")
    print("=" * 60)

    # ── Load original h5ad ───────────────────────────────────────────────────
    print("\n[1/5] Loading original h5ad ...")
    adata_full = sc.read_h5ad(args.input)
    print(f"  Full shape: {adata_full.shape}")
    print(f"  Controls (control==1): {(adata_full.obs['control'] == 1).sum()}")

    # ── Filter to drug-treated samples ───────────────────────────────────────
    print("\n[2/5] Filtering to drug-treated samples ...")
    treated_mask = adata_full.obs['control'].values == 0
    adata = adata_full[treated_mask].copy()
    print(f"  Drug-treated: {adata.shape[0]}")

    # ── Build X_ctl from paired_control_index ────────────────────────────────
    print("\n[3/5] Building X_ctl from paired_control_index ...")
    idx2pos = {name: pos for pos, name in enumerate(adata_full.obs.index)}
    ctrl_names = adata.obs['paired_control_index'].values.astype(str)
    ctrl_positions = np.array([idx2pos[n] for n in ctrl_names], dtype=np.int64)

    X_full = adata_full.X
    if sparse.issparse(X_full):
        X_full = X_full.toarray()

    X_ctl = X_full[ctrl_positions]
    print(f"  X_ctl shape: {X_ctl.shape}")

    # ── Build pert_idx + idx2smi ─────────────────────────────────────────────
    print("\n[4/5] Building pert_idx and idx2smi ...")
    smiles_all = adata.obs['SMILES'].values.astype(str)
    unique_smiles = sorted(set(smiles_all))
    smi2idx = {s: i for i, s in enumerate(unique_smiles)}
    idx2smi = {i: s for s, i in smi2idx.items()}

    pert_idx = np.array([smi2idx[s] for s in smiles_all], dtype=np.int64)
    print(f"  Unique drugs: {len(unique_smiles)}")

    cell_types = adata.obs['cell_type'].values.astype(str)
    unique_cells = sorted(set(cell_types))
    cell2idx = {c: i for i, c in enumerate(unique_cells)}
    cell_idx = np.array([cell2idx[c] for c in cell_types], dtype=np.int64)
    print(f"  Unique cell types: {len(unique_cells)}")

    # ── Assemble unified-schema h5ad ──────────────────────────────────────────
    print("\n[5/5] Assembling unified h5ad ...")
    X_pert = adata.X
    if sparse.issparse(X_pert):
        X_pert = X_pert.toarray()

    obs_dict = {
        'pert_idx': pert_idx,
        'cell_idx': pd.Categorical(cell_idx),
        'dose': adata.obs['dose'].values.astype(float),
        'cell_type': cell_types,
        'SMILES': smiles_all,
    }

    split_cols = [f'drug_split_{i}' for i in range(5)]
    for col in split_cols:
        vals = adata.obs[col].values.astype(str)
        vals[vals == ''] = 'exclude'
        obs_dict[col] = vals

    obs_df = pd.DataFrame(obs_dict)
    obs_df.index = [str(i) for i in range(len(obs_df))]

    adata_nm = anndata.AnnData(
        X=X_pert.astype(np.float32),
        obs=obs_df,
        obsm={'X_ctl': X_ctl.astype(np.float32)},
    )
    adata_nm.var_names = adata.var_names

    for col in split_cols:
        counts = adata_nm.obs[col].value_counts().to_dict()
        print(f"  {col}: {counts}")

    h5ad_path = os.path.join(args.output_dir, "prnet_original.h5ad")
    smi_path = os.path.join(args.output_dir, "idx2smi.npy")

    adata_nm.write_h5ad(h5ad_path)
    if not os.path.exists(h5ad_path) or os.path.getsize(h5ad_path) == 0:
        raise RuntimeError(
            f"h5ad write produced empty/missing file: {h5ad_path}. "
            "On lustre/NFS, set HDF5_USE_FILE_LOCKING=FALSE."
        )
    np.save(smi_path, idx2smi)

    print(f"\n  Saved: {h5ad_path} ({adata_nm.shape})")
    print(f"  Saved: {smi_path} ({len(idx2smi)} drugs)")
    print("=" * 60)
    print("Done! Next: python experiments/original_reproduction/train_all.py --models prnet --folds <fold>")


if __name__ == "__main__":
    main()
