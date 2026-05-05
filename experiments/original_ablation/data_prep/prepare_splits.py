"""
Generate 5-fold cold-drug `drug_split_0..4` columns for models whose
original dataset doesn't already have them (DeepCE, MultiDCP, TranSiGen).

CIGER, PertDiT, PRnet are skipped because their converted h5ad
already contains drug_split_0..4 (produced by their own convert_<M>.py).

In-place idempotent: if all 5 columns are present, skip; otherwise
KFold(5, shuffle=True, random_state=42) over unique drugs and write back.

Usage:
    python experiments/original_ablation/data_prep/prepare_splits.py
    python experiments/original_ablation/data_prep/prepare_splits.py --models deepce multidcp
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[3]
CONV_BASE = REPO_ROOT / "data" / "_converted"

# Only these 3 models need drug_split_0..4 generated; the others (CIGER,
# PertDiT, PRnet) already have them from their convert_<M>.py.
MODELS_NEEDING_SPLIT = ["deepce", "multidcp", "transigen"]

SEED = 42
N_SPLITS = 5


def has_all_splits(adata) -> bool:
    return all(f"drug_split_{k}" in adata.obs.columns for k in range(N_SPLITS))


def create_drug_splits(adata):
    """Add drug_split_0..4 columns (in-place) using KFold over unique drugs."""
    drug_ids = adata.obs["pert_idx"].values
    unique_drugs = np.unique(drug_ids)
    n_drugs = len(unique_drugs)
    print(f"  {adata.n_obs} samples, {n_drugs} unique drugs")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_drugs))

    drug_to_sample: dict = {}
    for i, d in enumerate(drug_ids):
        drug_to_sample.setdefault(d, []).append(i)

    for fold_k in range(N_SPLITS):
        col_name = f"drug_split_{fold_k}"
        labels = np.full(adata.n_obs, "", dtype=object)

        test_drug_idx = folds[fold_k][1]
        valid_drug_idx = folds[(fold_k + 1) % N_SPLITS][1]
        train_drug_idx = set(range(n_drugs)) - set(test_drug_idx) - set(valid_drug_idx)

        test_drugs = set(unique_drugs[test_drug_idx])
        valid_drugs = set(unique_drugs[valid_drug_idx])
        train_drugs = set(unique_drugs[list(train_drug_idx)])

        assert not (test_drugs & valid_drugs), "Drug leak: test ∩ valid"
        assert not (test_drugs & train_drugs), "Drug leak: test ∩ train"
        assert not (valid_drugs & train_drugs), "Drug leak: valid ∩ train"

        for d in unique_drugs:
            if d in test_drugs:
                tag = "test"
            elif d in valid_drugs:
                tag = "valid"
            else:
                tag = "train"
            for idx in drug_to_sample[d]:
                labels[idx] = tag

        assert "" not in labels, "Unlabeled samples found"
        adata.obs[col_name] = pd.Categorical(labels, categories=["train", "valid", "test"])
        nt = (labels == "train").sum()
        nv = (labels == "valid").sum()
        ntst = (labels == "test").sum()
        print(f"  {col_name}: train={nt}({len(train_drugs)}d) valid={nv}({len(valid_drugs)}d) test={ntst}({len(test_drugs)}d)")

    return adata


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=MODELS_NEEDING_SPLIT,
                   help=f"Which models to add splits for (default: {MODELS_NEEDING_SPLIT})")
    p.add_argument("--force", action="store_true",
                   help="Re-generate even if drug_split_0..4 already present")
    args = p.parse_args()

    for model in args.models:
        h5ad_path = CONV_BASE / model / f"{model}_original.h5ad"
        print(f"\n{'='*60}")
        print(f"Processing: {model}")
        print(f"  Path: {h5ad_path}")

        if not h5ad_path.exists():
            print(f"  SKIP — h5ad missing (run convert_{model}.py first)")
            continue

        adata = anndata.read_h5ad(h5ad_path)

        if has_all_splits(adata) and not args.force:
            print(f"  SKIP — drug_split_0..4 already present")
            continue

        adata = create_drug_splits(adata)
        adata.write_h5ad(h5ad_path)
        print(f"  Saved (in-place): {h5ad_path} ({os.path.getsize(h5ad_path) / 1e6:.1f} MB)")

    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
