"""
Convert MultiDCP original cold-cell 3-fold CV data to the unified h5ad schema.

Input:  signature_{train,dev,test}_cell_{1,2,3}.csv  (9 files)
        adjusted_ccle_tcga_ad_tpm_log2.csv           (cell basal expression)
        all_drugs_l1000.csv                           (drug SMILES)
Output: multidcp_original.h5ad  + idx2smi.npy
        3 split columns: cell_1, cell_2, cell_3  (values: train/valid/test)

MultiDCP's DEG main task is leave-new-cells-out 3-fold CV over 15 cell lines.
All 3 folds share the same 6087 samples (503 drugs × 15 cells); only the
train/dev/test cell assignment changes across folds.
"""

import argparse
import os
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
# dpb extracts data/MultiDCP/data.tar.gz → data/MultiDCP/extracted/data/ first
# (handled by step_extract_multidcp_data in prepare_steps.py).
MULTIDCP_DATA = str(REPO_ROOT / "data" / "MultiDCP" / "extracted" / "data")
OUT_DIR = str(REPO_ROOT / "data" / "_converted" / "multidcp")

# 17 gene-symbol aliases (old signature name → new CCLE/HGNC name)
GENE_ALIAS = {
    "ADCK3":    "COQ8A",
    "FAM63A":   "MINDY1",
    "HDGFRP3":  "HDGFL3",
    "HN1L":     "JPT2",
    "IKBKAP":   "ELP1",
    "KIAA0196": "WASHC5",
    "KIAA0907": "KHDC4",
    "KIAA1033": "WASHC4",
    "LRRC16A":  "CARMIL1",
    "NARFL":    "CIAO3",
    "PAPD7":    "TENT4A",
    "PRUNE":    "PRUNE1",
    "SQRDL":    "SQOR",
    "TMEM110":  "STIMATE",
    "TMEM2":    "CEMIP2",
    "TMEM5":    "RXYLT1",
    "TOMM70A":  "TOMM70",
}

EXPECTED_COUNTS = {
    "cell_1": {"train": 3918, "valid": 1072, "test": 1097},
    "cell_2": {"train": 4468, "valid": 944,  "test": 675},
    "cell_3": {"train": 3788, "valid": 850,  "test": 1449},
}

EXPECTED_CELLS = {
    "cell_1": {
        "train": {"A375","BT20","HA1E","HCC515","HEPG2","HS578T","HT29","JURKAT","PC3","YAPC"},
        "valid": {"MCF7","SKBR3"},
        "test":  {"A549","HELA","MDAMB231"},
    },
    "cell_2": {
        "train": {"A375","A549","BT20","HA1E","HELA","HT29","JURKAT","MCF7","MDAMB231","SKBR3"},
        "valid": {"HEPG2","PC3"},
        "test":  {"HCC515","HS578T","YAPC"},
    },
    "cell_3": {
        "train": {"A549","HCC515","HELA","HEPG2","HS578T","MCF7","MDAMB231","PC3","SKBR3","YAPC"},
        "valid": {"BT20","HA1E"},
        "test":  {"A375","HT29","JURKAT"},
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=MULTIDCP_DATA)
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("MultiDCP original cold-cell 3-fold → unified-schema h5ad")
    print("=" * 70)

    # ── 1. Load all 9 signature CSVs and merge by sig_id ────────────────────
    print("\n[1/6] Loading 9 signature CSVs ...")
    sig_dir = os.path.join(args.data_dir, "pert_transcriptom")
    meta_cols = ["sig_id", "pert_id", "pert_type", "cell_id", "pert_idose"]

    # Use cell_1/train as the base for gene columns
    base_df = pd.read_csv(os.path.join(sig_dir, "signature_train_cell_1.csv"), nrows=0)
    gene_cols = [c for c in base_df.columns if c not in meta_cols]
    assert len(gene_cols) == 978, f"Expected 978 genes, got {len(gene_cols)}"
    print(f"  Gene columns: {len(gene_cols)}")

    # Build per-fold split assignments keyed by sig_id
    fold_splits = {}  # fold_name → {sig_id → split_label}
    for fold_num in [1, 2, 3]:
        fold_name = f"cell_{fold_num}"
        sig2split = {}
        for orig_split, bench_split in [("train", "train"), ("dev", "valid"), ("test", "test")]:
            fname = f"signature_{orig_split}_cell_{fold_num}.csv"
            df = pd.read_csv(os.path.join(sig_dir, fname), usecols=["sig_id"])
            for sid in df["sig_id"]:
                assert sid not in sig2split, f"Duplicate sig_id {sid} in fold {fold_name}"
                sig2split[sid] = bench_split
            print(f"  {fname}: {len(df)} samples")
        fold_splits[fold_name] = sig2split

    # Load full data from cell_1 (arbitrary — all folds share same samples)
    dfs = []
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(sig_dir, f"signature_{split}_cell_1.csv"))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    assert len(df_all) == 6087, f"Expected 6087 samples, got {len(df_all)}"
    assert df_all["sig_id"].nunique() == 6087, "Duplicate sig_ids found"
    print(f"  Total unique samples: {len(df_all)}")

    # ── 2. Extract perturbed expression (X) ─────────────────────────────────
    print("\n[2/6] Extracting perturbed expression ...")
    X_pert = df_all[gene_cols].values.astype(np.float32)
    print(f"  X_pert shape: {X_pert.shape}, range: [{X_pert.min():.3f}, {X_pert.max():.3f}]")

    # ── 3. Load CCLE basal expression and align to signature gene order ─────
    print("\n[3/6] Loading and aligning CCLE basal expression ...")
    ccle_path = os.path.join(args.data_dir, "adjusted_ccle_tcga_ad_tpm_log2.csv")
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    print(f"  CCLE shape: {ccle_df.shape}")

    # Build reverse alias: new_name → old_name
    new_to_old = {v: k for k, v in GENE_ALIAS.items()}

    # Reorder CCLE columns to match signature gene order
    ccle_cols_reordered = []
    missing_genes = []
    for g in gene_cols:
        if g in ccle_df.columns:
            ccle_cols_reordered.append(g)
        elif g in GENE_ALIAS and GENE_ALIAS[g] in ccle_df.columns:
            ccle_cols_reordered.append(GENE_ALIAS[g])
        else:
            missing_genes.append(g)

    if missing_genes:
        print(f"  ERROR: {len(missing_genes)} genes not found in CCLE: {missing_genes}")
        sys.exit(1)
    assert len(ccle_cols_reordered) == 978

    # Verify all 15 cell lines exist
    all_cells = sorted(df_all["cell_id"].unique())
    assert len(all_cells) == 15, f"Expected 15 cell lines, got {len(all_cells)}"
    for c in all_cells:
        assert c in ccle_df.index, f"Cell {c} not in CCLE"
    print(f"  All 15 cell lines found in CCLE")

    # Build X_ctl: per-sample basal expression from CCLE, aligned to signature gene order
    ccle_aligned = ccle_df[ccle_cols_reordered].copy()
    ccle_aligned.columns = gene_cols  # rename back to signature names
    X_ctl = np.zeros_like(X_pert)
    for i, cell in enumerate(df_all["cell_id"]):
        X_ctl[i] = ccle_aligned.loc[cell].values.astype(np.float32)
    print(f"  X_ctl shape: {X_ctl.shape}, range: [{X_ctl.min():.3f}, {X_ctl.max():.3f}]")

    # ── 4. Build drug and cell indices ──────────────────────────────────────
    print("\n[4/6] Building drug/cell indices and SMILES ...")
    drug_smi_path = os.path.join(args.data_dir, "all_drugs_l1000.csv")
    drug_smi = {}
    with open(drug_smi_path) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2 and parts[0] != "broad_cpd_id":
                drug_smi[parts[0]] = parts[1]
    print(f"  Drug SMILES lookup: {len(drug_smi)} entries")

    unique_drugs = sorted(df_all["pert_id"].unique())
    drug2idx = {d: i for i, d in enumerate(unique_drugs)}
    idx2smi = {}
    missing_smi = 0
    for d in unique_drugs:
        idx = drug2idx[d]
        smi = drug_smi.get(d, None)
        if smi is None:
            print(f"  WARNING: No SMILES for {d}")
            missing_smi += 1
            idx2smi[idx] = "MISSING"
        else:
            idx2smi[idx] = smi
    pert_idx = np.array([drug2idx[d] for d in df_all["pert_id"]], dtype=np.int64)
    print(f"  Unique drugs: {len(unique_drugs)}, missing SMILES: {missing_smi}")

    cell2idx = {c: i for i, c in enumerate(all_cells)}
    cell_idx = np.array([cell2idx[c] for c in df_all["cell_id"]], dtype=np.int64)
    print(f"  Unique cells: {len(all_cells)}: {all_cells}")

    # ── 5. Assemble obs DataFrame ───────────────────────────────────────────
    print("\n[5/6] Assembling AnnData ...")
    obs_dict = {
        "pert_idx": pert_idx,
        "cell_idx": pd.Categorical(cell_idx),
        "cell_mfc_name": df_all["cell_id"].values.astype(str),
        "pert_id": df_all["pert_id"].values.astype(str),
        "SMILES": np.array([idx2smi[drug2idx[d]] for d in df_all["pert_id"]]),
        "sig_id": df_all["sig_id"].values.astype(str),
        "pert_idose": df_all["pert_idose"].values.astype(str),
    }
    # Add 3 fold split columns
    for fold_name, sig2split in fold_splits.items():
        obs_dict[fold_name] = np.array([sig2split[sid] for sid in df_all["sig_id"]])

    obs_df = pd.DataFrame(obs_dict)
    obs_df.index = [str(i) for i in range(len(obs_df))]

    adata = anndata.AnnData(
        X=X_pert,
        obs=obs_df,
        obsm={"X_ctl": X_ctl},
    )
    adata.var_names = gene_cols

    # ── 6. Validation ───────────────────────────────────────────────────────
    print("\n[6/6] Validation ...")
    all_ok = True
    for fold_name in ["cell_1", "cell_2", "cell_3"]:
        splits = adata.obs[fold_name].values
        for split_label, expected_n in EXPECTED_COUNTS[fold_name].items():
            actual_n = (splits == split_label).sum()
            ok = actual_n == expected_n
            status = "OK" if ok else "FAIL"
            print(f"  {fold_name}/{split_label}: expected={expected_n}, actual={actual_n} [{status}]")
            if not ok:
                all_ok = False

        # Check cell sets
        for split_label, expected_cells in EXPECTED_CELLS[fold_name].items():
            mask = splits == split_label
            actual_cells = set(adata.obs.loc[mask, "cell_mfc_name"].unique())
            ok = actual_cells == expected_cells
            status = "OK" if ok else "FAIL"
            if not ok:
                print(f"  {fold_name}/{split_label} cells: expected={sorted(expected_cells)}, actual={sorted(actual_cells)} [{status}]")
                all_ok = False

    # Cross-fold: each sample appears exactly once per split assignment
    for fold_name in ["cell_1", "cell_2", "cell_3"]:
        splits = adata.obs[fold_name].values
        for label in ["train", "valid", "test"]:
            assert (splits == label).sum() > 0, f"No {label} in {fold_name}"
        assert len(splits) == sum(EXPECTED_COUNTS[fold_name].values())

    if not all_ok:
        print("\n  *** VALIDATION FAILED ***")
        sys.exit(1)
    print("  All validation checks passed!")

    # ── Save ────────────────────────────────────────────────────────────────
    h5ad_path = os.path.join(args.output_dir, "multidcp_original.h5ad")
    smi_path_out = os.path.join(args.output_dir, "idx2smi.npy")

    adata.write_h5ad(h5ad_path)
    if not os.path.exists(h5ad_path) or os.path.getsize(h5ad_path) == 0:
        raise RuntimeError(
            f"h5ad write produced empty/missing file: {h5ad_path}. "
            "On lustre/NFS, set HDF5_USE_FILE_LOCKING=FALSE."
        )
    np.save(smi_path_out, idx2smi)

    print(f"\n  Saved: {h5ad_path} ({adata.shape})")
    print(f"  Saved: {smi_path_out} ({len(idx2smi)} drugs)")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
