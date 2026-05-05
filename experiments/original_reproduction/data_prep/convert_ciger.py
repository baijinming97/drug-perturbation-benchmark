"""
Convert CIGER original CSV to the unified h5ad schema.

Input:  chemical_signature.csv (4165 samples × 978 genes, differential expr)
        drug_smiles.csv (pert_id → SMILES)
        drug_id.csv (single-line comma-sep list for splitting)
        gene_feature.csv (978 genes × 1107 features)

Output: unified-schema h5ad + idx2smi.npy + gene_feature_p6.csv
        5 split columns (drug_split_0..4)

Faithfulness notes:
- Replicates choose_mean_example() dedup: groups by (pert_id, pert_type,
  cell_id, pert_idose), picks the sample closest to median rank (4165→3332).
- Replicates split_data_by_pert_id_cv(): drug_id.csv order preserved
  (original shuffle() discards return value).
- Iterates sorted(data.items()) to match original row order.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
CIGER_DATA = str(REPO_ROOT / "data" / "CIGER")
OUT_DIR = str(REPO_ROOT / "data" / "_converted" / "ciger")


def replicate_ciger_split(drug_id_file):
    """Replicate CIGER split_data_by_pert_id_cv exactly.

    The original code does shuffle(pert_id, random_state=87) but discards
    the return value, so pert_id stays in file order. We replicate this.
    """
    with open(drug_id_file) as f:
        pert_ids = f.readline().strip().split(",")

    num = len(pert_ids)
    fold_size = int(num / 10)

    folds = [
        pert_ids[:fold_size * 2],
        pert_ids[fold_size * 2:fold_size * 4],
        pert_ids[fold_size * 4:fold_size * 6],
        pert_ids[fold_size * 6:fold_size * 8],
        pert_ids[fold_size * 8:],
    ]

    splits = {}
    for fold in range(5):
        test_ids = set(folds[fold])
        dev_ids = set(folds[(fold + 1) % 5])
        train_ids = set()
        for j in range(5):
            if j != fold and j != (fold + 1) % 5:
                train_ids.update(folds[j])
        splits[fold] = (train_ids, dev_ids, test_ids)

    return splits


def choose_mean_example(examples):
    """Pick the replicate closest to median rank — exact copy of original."""
    examples = np.array(examples)
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value) ** 2
    index = np.argmin(distance)
    return examples[index]


def dedup_like_original(df, gene_cols):
    """Replicate original read_data() grouping and dedup.

    Groups by (pert_id, pert_type, cell_id, pert_idose), applies
    choose_mean_example for groups with >1 replicate, iterates in
    sorted key order to match original row ordering.
    """
    data = {}
    meta = {}
    for _, row in df.iterrows():
        key = f"{row['pert_id']},{row['pert_type']},{row['cell_id']},{row['pert_idose']}"
        lb = row[gene_cols].values.astype(np.float64)
        if key in data:
            data[key].append(lb)
        else:
            data[key] = [lb]
            meta[key] = row

    deduped_labels = []
    deduped_meta = []
    for key in sorted(data.keys()):
        lbs = data[key]
        if len(lbs) == 1:
            deduped_labels.append(lbs[0])
        else:
            deduped_labels.append(choose_mean_example(lbs))
        deduped_meta.append(meta[key])

    labels = np.array(deduped_labels, dtype=np.float32)
    meta_df = pd.DataFrame(deduped_meta).reset_index(drop=True)
    return meta_df, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ciger_data", type=str, default=CIGER_DATA)
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CIGER original → unified-schema h5ad conversion (with dedup)")
    print("=" * 60)

    # ── 1. Read chemical_signature.csv ──────────────────────────────────────
    print("\n[1/6] Loading chemical_signature.csv ...")
    sig_path = os.path.join(args.ciger_data, "chemical_signature.csv")
    df = pd.read_csv(sig_path)
    print(f"  Raw shape: {df.shape}")

    meta_cols = ["sig_id", "pert_id", "pert_type", "cell_id", "pert_idose"]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Metadata cols: {len(meta_cols)}, Gene cols: {len(gene_cols)}")

    # ── 2. Dedup via choose_mean_example ────────────────────────────────────
    print("\n[2/6] Deduplicating (choose_mean_example) ...")
    df_dedup, x_deg = dedup_like_original(df, gene_cols)
    print(f"  Before: {len(df)} rows → After: {len(df_dedup)} unique groups")
    print(f"  x_deg shape: {x_deg.shape}, range: [{x_deg.min():.3f}, {x_deg.max():.3f}]")

    # ── 3. Read drug_smiles.csv ─────────────────────────────────────────────
    print("\n[3/6] Loading drug_smiles.csv ...")
    smi_path = os.path.join(args.ciger_data, "drug_smiles.csv")
    drug_smi = {}
    with open(smi_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                drug_smi[row[0]] = row[1]
    print(f"  Drug SMILES: {len(drug_smi)}")

    # Build pert_idx mapping (from deduped data)
    unique_drugs = sorted(set(df_dedup["pert_id"]))
    drug2idx = {d: i for i, d in enumerate(unique_drugs)}
    idx2smi = {}
    missing_smi = []
    for d in unique_drugs:
        idx = drug2idx[d]
        if d in drug_smi:
            idx2smi[idx] = drug_smi[d]
        else:
            missing_smi.append(d)
            idx2smi[idx] = "MISSING"

    pert_idx = np.array([drug2idx[d] for d in df_dedup["pert_id"]], dtype=np.int64)
    print(f"  Unique drugs in data: {len(unique_drugs)}")
    if missing_smi:
        print(f"  WARNING: {len(missing_smi)} drugs missing SMILES")

    # Cell index
    unique_cells = sorted(set(df_dedup["cell_id"]))
    cell2idx = {c: i for i, c in enumerate(unique_cells)}
    cell_idx = np.array([cell2idx[c] for c in df_dedup["cell_id"]], dtype=np.int64)
    print(f"  Unique cells: {len(unique_cells)}: {unique_cells}")

    # ── 4. Apply CIGER 5-fold split ─────────────────────────────────────────
    print("\n[4/6] Computing 5-fold cold-drug split ...")
    drug_id_file = os.path.join(args.ciger_data, "drug_id.csv")
    all_splits = replicate_ciger_split(drug_id_file)

    split_arrays = {}
    for fold in range(5):
        train_ids, dev_ids, test_ids = all_splits[fold]
        col_name = f"drug_split_{fold}"
        labels = np.full(len(df_dedup), "", dtype=object)
        for i, pid in enumerate(df_dedup["pert_id"]):
            if pid in train_ids:
                labels[i] = "train"
            elif pid in dev_ids:
                labels[i] = "valid"
            elif pid in test_ids:
                labels[i] = "test"
            else:
                labels[i] = "exclude"
        split_arrays[col_name] = labels
        n_tr = np.sum(labels == "train")
        n_va = np.sum(labels == "valid")
        n_te = np.sum(labels == "test")
        n_ex = np.sum(labels == "exclude")
        print(f"  {col_name}: train={n_tr}, valid={n_va}, test={n_te}, exclude={n_ex}")

    # ── 5. Assemble unified-schema h5ad ─────────────────────────────────────
    print("\n[5/6] Assembling unified h5ad ...")
    obs_dict = {
        "pert_idx": pert_idx,
        "cell_idx": pd.Categorical(cell_idx),
        "cell_mfc_name": df_dedup["cell_id"].values.astype(str),
        "pert_id": df_dedup["pert_id"].values.astype(str),
        "SMILES": np.array([idx2smi[drug2idx[d]] for d in df_dedup["pert_id"]]),
    }
    for col_name, labels in split_arrays.items():
        obs_dict[col_name] = labels

    obs_df = pd.DataFrame(obs_dict)
    obs_df.index = [str(i) for i in range(len(obs_df))]

    adata = anndata.AnnData(
        X=x_deg,
        obs=obs_df,
        obsm={"X_ctl": np.zeros_like(x_deg)},
    )
    adata.var_names = gene_cols

    h5ad_path = os.path.join(args.output_dir, "ciger_original.h5ad")
    smi_path_out = os.path.join(args.output_dir, "idx2smi.npy")

    adata.write_h5ad(h5ad_path)
    if not os.path.exists(h5ad_path) or os.path.getsize(h5ad_path) == 0:
        raise RuntimeError(
            f"h5ad write produced empty/missing file: {h5ad_path}. "
            "On lustre/NFS, set HDF5_USE_FILE_LOCKING=FALSE."
        )
    np.save(smi_path_out, idx2smi)

    print(f"  Saved: {h5ad_path} ({adata.shape})")
    print(f"  Saved: {smi_path_out} ({len(idx2smi)} drugs)")

    # ── 6. Copy gene feature file ───────────────────────────────────────────
    print("\n[6/6] Copying gene features ...")
    gene_src = os.path.join(args.ciger_data, "gene_feature.csv")
    gene_dst = os.path.join(args.output_dir, "gene_feature_p6.csv")
    shutil.copy2(gene_src, gene_dst)
    print(f"  Copied: {gene_dst}")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
