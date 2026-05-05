"""
Convert DeepCE original CSV to the unified h5ad schema.

Faithfully reproduces the original read_data() pipeline:
  1. Filter by time=24H, pert_type=trt_cp, 7 cell lines, 6 doses, exclude 2 pert_ids
  2. Deduplicate with choose_mean_example (same drug-cell-dose → keep median-rank sample)
  3. Sort by (pert_id, pert_type, cell_id, pert_idose) key

Preserves pert_type, cell_id_str, pert_idose as obs columns for auto-detection.

Input:  signature_train.csv + signature_dev.csv + signature_test.csv
        drugs_smiles.csv
Output: unified-schema h5ad + idx2smi.npy
        Single split column 'fixed_split' (train/valid/test from original files)

DeepCE uses a fixed train/dev/test split (not cross-validation).
3 seeds (343/344/345) only affect model initialization, not data split.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEEPCE_DATA = str(REPO_ROOT / "data" / "DeepCE")
OUT_DIR = str(REPO_ROOT / "data" / "_converted" / "deepce")

FILTER = {
    "time": "24H",
    "pert_id_exclude": {'BRD-U41416256', 'BRD-U60236422'},
    "pert_type": {"trt_cp"},
    "cell_id": {'A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'},
    "pert_idose": {"0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"},
}


def choose_mean_example(examples):
    """Select the sample closest to median rank across genes.

    Exact copy of original data_utils.py choose_mean_example().
    """
    examples = np.array(examples)
    mean_value = (len(examples) - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value) ** 2
    index = np.argmin(distance)
    return examples[index]


def read_and_filter_csv(path, split_name):
    """Read CSV and apply original filter + dedup. Returns (features, labels, split_labels)."""
    data = defaultdict(list)
    raw_count = 0

    with open(path) as f:
        header = f.readline()
        for line in f:
            raw_count += 1
            parts = line.strip().split(',')
            sig_id, pert_id, pert_type, cell_id, pert_idose = parts[0], parts[1], parts[2], parts[3], parts[4]
            if (FILTER["time"] in sig_id
                    and pert_id not in FILTER["pert_id_exclude"]
                    and pert_type in FILTER["pert_type"]
                    and cell_id in FILTER["cell_id"]
                    and pert_idose in FILTER["pert_idose"]):
                key = f"{pert_id},{pert_type},{cell_id},{pert_idose}"
                lb = [float(x) for x in parts[5:]]
                data[key].append(lb)

    features = []
    labels = []
    for key, lbs in sorted(data.items()):
        ft = key.split(',')
        features.append(ft)
        if len(lbs) == 1:
            labels.append(lbs[0])
        else:
            labels.append(choose_mean_example(lbs))

    filtered_count = sum(len(v) for v in data.values())
    dedup_count = len(features)
    print(f"  {split_name}: raw={raw_count}, after_filter={filtered_count}, after_dedup={dedup_count}")

    return features, np.array(labels, dtype=np.float64), [split_name] * dedup_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deepce_data", type=str, default=DEEPCE_DATA)
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DeepCE original → unified-schema h5ad conversion (faithful)")
    print("=" * 60)

    # ── 1. Load, filter, dedup CSVs (matches original read_data) ───────────
    print("\n[1/4] Loading + filtering + dedup ...")
    all_features = []
    all_labels = []
    all_splits = []

    for split_name, file_name in [("train", "signature_train.csv"),
                                   ("valid", "signature_dev.csv"),
                                   ("test", "signature_test.csv")]:
        path = os.path.join(args.deepce_data, file_name)
        features, labels, splits = read_and_filter_csv(path, split_name)
        all_features.extend(features)
        all_labels.append(labels)
        all_splits.extend(splits)

    all_labels = np.concatenate(all_labels, axis=0)
    print(f"  Total after filter+dedup: {len(all_features)}")
    print(f"  Labels shape: {all_labels.shape}")

    # ── 2. Drug SMILES ──────────────────────────────────────────────────────
    print("\n[2/4] Loading drugs_smiles.csv ...")
    smi_path = os.path.join(args.deepce_data, "drugs_smiles.csv")
    drug_smi = {}
    with open(smi_path) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                drug_smi[parts[0]] = parts[1]
    print(f"  Drug SMILES dict: {len(drug_smi)}")

    pert_ids = [ft[0] for ft in all_features]
    pert_types = [ft[1] for ft in all_features]
    cell_ids = [ft[2] for ft in all_features]
    pert_idoses = [ft[3] for ft in all_features]

    unique_drugs = sorted(set(pert_ids))
    drug2idx = {d: i for i, d in enumerate(unique_drugs)}
    idx2smi = {drug2idx[d]: drug_smi.get(d, "MISSING") for d in unique_drugs}
    pert_idx = np.array([drug2idx[d] for d in pert_ids], dtype=np.int64)
    print(f"  Unique drugs: {len(unique_drugs)}")

    unique_cells = sorted(set(cell_ids))
    cell2idx = {c: i for i, c in enumerate(unique_cells)}
    cell_idx = np.array([cell2idx[c] for c in cell_ids], dtype=np.int64)
    print(f"  Unique cells: {len(unique_cells)}: {unique_cells}")

    unique_doses = sorted(set(pert_idoses))
    print(f"  Unique doses: {len(unique_doses)}: {unique_doses}")

    unique_pert_types = sorted(set(pert_types))
    print(f"  Unique pert_types: {len(unique_pert_types)}: {unique_pert_types}")

    # ── 3. Verify cold-drug split ───────────────────────────────────────────
    print("\n[3/4] Verifying cold-drug property ...")
    splits_arr = np.array(all_splits)
    for s in ["train", "valid", "test"]:
        mask = splits_arr == s
        drugs_in_split = set(np.array(pert_ids)[mask])
        print(f"  {s}: {mask.sum()} samples, {len(drugs_in_split)} drugs")

    train_drugs = set(np.array(pert_ids)[splits_arr == "train"])
    valid_drugs = set(np.array(pert_ids)[splits_arr == "valid"])
    test_drugs = set(np.array(pert_ids)[splits_arr == "test"])
    print(f"  Train∩Valid: {len(train_drugs & valid_drugs)}, "
          f"Train∩Test: {len(train_drugs & test_drugs)}, "
          f"Valid∩Test: {len(valid_drugs & test_drugs)}")

    # ── 4. Assemble unified-schema h5ad ─────────────────────────────────────
    print("\n[4/4] Assembling unified h5ad ...")
    obs_dict = {
        "pert_idx": pert_idx,
        "cell_idx": pd.Categorical(cell_idx),
        "cell_mfc_name": np.array(cell_ids, dtype=str),
        "pert_id": np.array(pert_ids, dtype=str),
        "pert_type": np.array(pert_types, dtype=str),
        "pert_idose": np.array(pert_idoses, dtype=str),
        "SMILES": np.array([idx2smi[drug2idx[d]] for d in pert_ids]),
        "fixed_split": np.array(all_splits),
    }

    obs_df = pd.DataFrame(obs_dict)
    obs_df.index = [str(i) for i in range(len(obs_df))]

    adata = anndata.AnnData(
        X=all_labels.astype(np.float32),
        obs=obs_df,
        obsm={"X_ctl": np.zeros((len(all_labels), all_labels.shape[1]), dtype=np.float32)},
    )

    gene_cols_path = os.path.join(args.deepce_data, "signature_train.csv")
    with open(gene_cols_path) as f:
        header = f.readline().strip().split(',')
    meta_cols = {"sig_id", "pert_id", "pert_type", "cell_id", "pert_idose"}
    gene_cols = [c for c in header if c not in meta_cols]
    adata.var_names = gene_cols

    h5ad_path = os.path.join(args.output_dir, "deepce_original.h5ad")
    smi_path_out = os.path.join(args.output_dir, "idx2smi.npy")

    adata.write_h5ad(h5ad_path)
    if not os.path.exists(h5ad_path) or os.path.getsize(h5ad_path) == 0:
        raise RuntimeError(
            f"h5ad write produced empty/missing file: {h5ad_path}. "
            "On lustre/NFS, set HDF5_USE_FILE_LOCKING=FALSE."
        )
    np.save(smi_path_out, idx2smi)

    n_tr = (splits_arr == "train").sum()
    n_va = (splits_arr == "valid").sum()
    n_te = (splits_arr == "test").sum()
    print(f"  fixed_split: train={n_tr}, valid={n_va}, test={n_te}")
    print(f"  Saved: {h5ad_path} ({adata.shape})")
    print(f"  Saved: {smi_path_out} ({len(idx2smi)} drugs)")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print("=" * 60)
    print("Done! Filter+dedup matches original read_data() exactly.")


if __name__ == "__main__":
    main()
