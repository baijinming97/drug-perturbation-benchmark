"""
Create 3:1:1 drug-blind 5-fold cross-validation splits.

Splits all unique drugs into 5 groups (seed=131419).
For each fold i:
  - test  = group i
  - valid = group (i+1) % 5
  - train = remaining 3 groups

Writes columns nm_drug_blind_1 ~ nm_drug_blind_5 into h5ad obs.
"""

import numpy as np
import scanpy as sc
import pandas as pd
import os

SEED = 131419
H5AD_PATH = "data/XPert/processed_data/l1000_sdst_78453.h5ad"
SPLITS_DIR = "data/splits/cold_drug"

print(f"Loading {H5AD_PATH} ...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {adata.shape}")

# Get unique drugs and shuffle
drugs = adata.obs['pert_idx'].unique()
rng = np.random.RandomState(SEED)
rng.shuffle(drugs)
print(f"Total unique drugs: {len(drugs)}")

# Split into 5 roughly equal groups
groups = np.array_split(drugs, 5)
for i, g in enumerate(groups):
    print(f"  Group {i}: {len(g)} drugs")

# Create drug-to-group mapping
drug_to_group = {}
for gi, g in enumerate(groups):
    for d in g:
        drug_to_group[d] = gi

# For each fold, assign train/valid/test
for fold_idx in range(5):
    col_name = f"nm_drug_blind_{fold_idx + 1}"
    test_group = fold_idx
    val_group = (fold_idx + 1) % 5
    train_groups = [g for g in range(5) if g != test_group and g != val_group]

    labels = []
    for d in adata.obs['pert_idx']:
        g = drug_to_group[d]
        if g == test_group:
            labels.append('test')
        elif g == val_group:
            labels.append('valid')
        else:
            labels.append('train')

    adata.obs[col_name] = pd.Categorical(labels, categories=['train', 'valid', 'test'])

    n_train = sum(1 for l in labels if l == 'train')
    n_valid = sum(1 for l in labels if l == 'valid')
    n_test = sum(1 for l in labels if l == 'test')
    n_train_drugs = sum(len(groups[g]) for g in train_groups)
    n_val_drugs = len(groups[val_group])
    n_test_drugs = len(groups[test_group])
    print(f"\n{col_name}:")
    print(f"  train: {n_train} samples ({n_train_drugs} drugs)")
    print(f"  valid: {n_valid} samples ({n_val_drugs} drugs)")
    print(f"  test:  {n_test} samples ({n_test_drugs} drugs)")

    # Save split info to CSV backup
    fold_dir = os.path.join(SPLITS_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    split_df = pd.DataFrame({
        'sample_idx': range(len(labels)),
        'split': labels,
        'drug_id': adata.obs['pert_idx'].values
    })
    split_df.to_csv(os.path.join(fold_dir, 'split_info.csv'), index=False)

# Save h5ad
print(f"\nSaving to {H5AD_PATH} ...")
adata.write_h5ad(H5AD_PATH)
print("Done!")
