"""
Create 3:1:1 drug-blind 5-fold cross-validation splits using
Bemis-Murcko scaffold grouping (LPT-balanced).

Drop-in replacement for create_splits.py:
  - Same seed (131419), same 5-group rotation, same drug-level granularity
  - ONLY change: group assignment = scaffold-aware LPT instead of random shuffle
  - Acyclic drugs (Murcko scaffold == "") are pooled into a single "__ACYCLIC__" bucket

Writes columns nm_scaffold_1 ~ nm_scaffold_5 into h5ad obs.
Writes backup CSVs to data/splits/scaffold/fold_{0..4}/split_info.csv
"""

import numpy as np
import pandas as pd
import scanpy as sc
import os
import json
import random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ===================== CONFIG (mirrors create_splits.py: same seed, same 5-group rotation) =====================
SEED = 131419
H5AD_PATH = "data/XPert/processed_data/l1000_sdst_78453.h5ad"
IDX2SMI_PATH = "data/XPert/processed_data/all_drugs_idx2smi_8981.npy"
SPLITS_DIR = "data/splits/scaffold"
META_OUT = "data/splits/scaffold/scaffold_split_meta.json"

INCLUDE_CHIRALITY = False
ACYCLIC_BUCKET_KEY = "__ACYCLIC__"
INVALID_BUCKET_KEY = "__INVALID__"

N_GROUPS = 5
# ==============================================================================


def assert_cwd():
    """Refuse to run unless CWD = dpb repository root."""
    anchors = ['experiments/scaffold_generalization', 'data/XPert/processed_data']
    missing = [a for a in anchors if not os.path.exists(a)]
    if missing:
        raise RuntimeError(
            f"CWD must be the dpb repository root. Missing anchors: {missing}. "
            f"Current CWD: {os.getcwd()}. "
            f"Run from the dpb repository root."
        )


def get_scaffold_key(smiles: str) -> str:
    """Exact Bemis-Murcko scaffold, acyclic → shared bucket."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return INVALID_BUCKET_KEY
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=INCLUDE_CHIRALITY
    )
    return scaffold if scaffold != "" else ACYCLIC_BUCKET_KEY


def lpt_assign_to_groups(scaffold_groups, n_groups, seed):
    """
    LPT (Longest Processing Time) balanced greedy assignment:
      - Sort scaffold groups by n_samples desc (tie: n_drugs desc, then key lex)
      - For each group, assign to the currently least-loaded fold
      - Tie-break among equally-loaded folds uses `seed` (deterministic)
    Returns: (list of 5 lists of pert_idx, list of 5 sample loads)
    """
    sorted_groups = sorted(
        scaffold_groups,
        key=lambda g: (-g['n_samples'], -g['n_drugs'], g['key'])
    )

    rng = random.Random(seed)
    group_loads = [0] * n_groups
    group_drugs = [[] for _ in range(n_groups)]

    for g in sorted_groups:
        min_load = min(group_loads)
        candidates = [i for i, l in enumerate(group_loads) if l == min_load]
        chosen = rng.choice(candidates) if len(candidates) > 1 else candidates[0]
        group_drugs[chosen].extend(g['drugs'])
        group_loads[chosen] += g['n_samples']

    return group_drugs, group_loads


def main():
    assert_cwd()

    print(f"[1/5] Loading {H5AD_PATH} ...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"      Shape: {adata.shape}")

    print(f"\n[2/5] Loading SMILES from {IDX2SMI_PATH} ...")
    idx2smi = np.load(IDX2SMI_PATH, allow_pickle=True).item()
    print(f"      Loaded {len(idx2smi)} drug SMILES")

    drugs = adata.obs['pert_idx'].unique()
    print(f"      Unique drugs in h5ad: {len(drugs)}")

    n_samples_per_drug = adata.obs['pert_idx'].value_counts().to_dict()

    # --------------- [3/5] Scaffold aggregation ---------------
    print(f"\n[3/5] Computing Bemis-Murcko scaffolds (includeChirality={INCLUDE_CHIRALITY}) ...")
    pert_to_scaffold = {}
    scaffold_to_perts = defaultdict(list)
    n_acyclic = 0
    n_invalid = 0

    for pert_idx in drugs:
        smi = idx2smi.get(int(pert_idx))
        if smi is None:
            print(f"      ⚠  WARNING: pert_idx={pert_idx} not found in idx2smi, using INVALID bucket")
            key = INVALID_BUCKET_KEY
        else:
            key = get_scaffold_key(smi)

        pert_to_scaffold[pert_idx] = key
        scaffold_to_perts[key].append(pert_idx)
        if key == ACYCLIC_BUCKET_KEY:
            n_acyclic += 1
        elif key == INVALID_BUCKET_KEY:
            n_invalid += 1

    scaffold_groups = []
    for key, perts in scaffold_to_perts.items():
        scaffold_groups.append({
            'key': key,
            'drugs': sorted(perts),
            'n_drugs': len(perts),
            'n_samples': sum(n_samples_per_drug[p] for p in perts),
        })

    total_scaffolds = len(scaffold_groups)
    non_bucket_scaffolds = [g for g in scaffold_groups
                            if g['key'] not in (ACYCLIC_BUCKET_KEY, INVALID_BUCKET_KEY)]
    singletons = [g for g in non_bucket_scaffolds if g['n_drugs'] == 1]

    print(f"\n      --- Scaffold Statistics ---")
    print(f"      Total unique scaffold groups: {total_scaffolds}")
    print(f"        Bemis-Murcko (with rings): {len(non_bucket_scaffolds)}")
    print(f"        Acyclic bucket: 1 group with {n_acyclic} drugs")
    print(f"        Invalid bucket: {n_invalid} drugs")
    print(f"      Singleton scaffolds (1 drug): {len(singletons)}")

    top10 = sorted(scaffold_groups, key=lambda g: -g['n_samples'])[:10]
    print(f"\n      Top-10 largest scaffold groups (by n_samples):")
    for i, g in enumerate(top10):
        key_short = g['key'][:60] + "..." if len(g['key']) > 60 else g['key']
        print(f"        #{i+1:2d}: n_drugs={g['n_drugs']:4d}, n_samples={g['n_samples']:5d}, "
              f"key={key_short}")

    target_per_group_drugs = len(drugs) / N_GROUPS
    target_per_group_samples = adata.n_obs / N_GROUPS
    top1 = top10[0]
    if top1['n_drugs'] > target_per_group_drugs:
        print(f"\n      ❌ HARD STOP: Top-1 scaffold has {top1['n_drugs']} drugs, "
              f"exceeds single-fold capacity ({target_per_group_drugs:.0f}). "
              f"LPT cannot balance. Please report.")
        return
    if top1['n_samples'] > target_per_group_samples:
        print(f"\n      ❌ HARD STOP: Top-1 scaffold has {top1['n_samples']} samples, "
              f"exceeds single-fold capacity ({target_per_group_samples:.0f}). "
              f"LPT cannot balance. Please report.")
        return
    print(f"\n      ✓ Top-1 scaffold fits within single-fold capacity.")

    # --------------- [4/5] LPT assignment ---------------
    print(f"\n[4/5] LPT-balanced assignment to {N_GROUPS} groups (seed={SEED}) ...")
    groups, group_loads = lpt_assign_to_groups(scaffold_groups, N_GROUPS, SEED)

    print(f"\n      Group balance after LPT:")
    mean_samples = np.mean(group_loads)
    max_dev_pct = 0.0
    for i, (g, load) in enumerate(zip(groups, group_loads)):
        dev_pct = (load - mean_samples) / mean_samples * 100
        max_dev_pct = max(max_dev_pct, abs(dev_pct))
        print(f"        Group {i}: {len(g):4d} drugs, {load:5d} samples, Δ={dev_pct:+.2f}%")
    print(f"      Max |Δ% n_samples|: {max_dev_pct:.2f}%")

    if max_dev_pct > 20:
        print(f"\n      ❌ HARD STOP: Max deviation {max_dev_pct:.2f}% > 20%. "
              f"Scaffold distribution too skewed. Please report.")
        return
    elif max_dev_pct > 15:
        print(f"\n      ⚠  WARNING: Max deviation {max_dev_pct:.2f}% in [15%, 20%]. "
              f"Continuing but flag in paper.")
    else:
        print(f"      ✓ Balance acceptable (< 15%).")

    # --------------- [5/5] 5-fold rotation (mirrors create_splits.py) ---------------
    drug_to_group = {}
    for gi, g in enumerate(groups):
        for d in g:
            drug_to_group[d] = gi

    print(f"\n[5/5] Generating 5-fold rotations and writing outputs ...")
    per_fold_stats = []

    for fold_idx in range(5):
        col_name = f"nm_scaffold_{fold_idx + 1}"
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

        print(f"\n      {col_name}:")
        print(f"        train: {n_train} samples ({n_train_drugs} drugs)")
        print(f"        valid: {n_valid} samples ({n_val_drugs} drugs)")
        print(f"        test:  {n_test} samples ({n_test_drugs} drugs)")

        per_fold_stats.append({
            'fold': fold_idx,
            'col': col_name,
            'n_train_samples': n_train, 'n_valid_samples': n_valid, 'n_test_samples': n_test,
            'n_train_drugs': n_train_drugs, 'n_valid_drugs': n_val_drugs, 'n_test_drugs': n_test_drugs,
        })

        fold_dir = os.path.join(SPLITS_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        split_df = pd.DataFrame({
            'sample_idx': range(len(labels)),
            'split': labels,
            'drug_id': adata.obs['pert_idx'].values,
        })
        split_df.to_csv(os.path.join(fold_dir, 'split_info.csv'), index=False)

    # --------------- Scaffold-disjoint validation ---------------
    print(f"\n[VALIDATE] Running scaffold-disjoint checks ...")
    for fold_idx in range(5):
        col = f'nm_scaffold_{fold_idx + 1}'
        train_drugs = set(adata.obs.loc[adata.obs[col] == 'train', 'pert_idx'].unique())
        valid_drugs = set(adata.obs.loc[adata.obs[col] == 'valid', 'pert_idx'].unique())
        test_drugs  = set(adata.obs.loc[adata.obs[col] == 'test',  'pert_idx'].unique())

        assert not (train_drugs & valid_drugs), f"fold {fold_idx}: train∩valid drug overlap"
        assert not (train_drugs & test_drugs),  f"fold {fold_idx}: train∩test drug overlap"
        assert not (valid_drugs & test_drugs),  f"fold {fold_idx}: valid∩test drug overlap"

        train_scafs = {pert_to_scaffold[d] for d in train_drugs}
        valid_scafs = {pert_to_scaffold[d] for d in valid_drugs}
        test_scafs  = {pert_to_scaffold[d] for d in test_drugs}
        assert not (train_scafs & valid_scafs), f"fold {fold_idx}: train-valid scaffold leak"
        assert not (train_scafs & test_scafs),  f"fold {fold_idx}: train-test scaffold leak"
        assert not (valid_scafs & test_scafs),  f"fold {fold_idx}: valid-test scaffold leak"

    all_drugs = adata.obs['pert_idx'].unique()
    for d in all_drugs:
        mask = adata.obs['pert_idx'] == d
        roles = [adata.obs.loc[mask, f'nm_scaffold_{i+1}'].iloc[0] for i in range(5)]
        assert roles.count('test') == 1, f"drug {d}: test count != 1"
        assert roles.count('valid') == 1, f"drug {d}: valid count != 1"
        assert roles.count('train') == 3, f"drug {d}: train count != 3"

    print(f"      ✓ All scaffold-disjoint and rotation invariants verified.")

    # --------------- Meta JSON ---------------
    meta = {
        'split_seed': SEED,
        'n_groups': N_GROUPS,
        'scaffold_definition': 'Bemis-Murcko (exact)',
        'include_chirality': INCLUDE_CHIRALITY,
        'acyclic_strategy': f'pooled into single "{ACYCLIC_BUCKET_KEY}" bucket',
        'n_acyclic_drugs': n_acyclic,
        'n_invalid_drugs': n_invalid,
        'algorithm': 'LPT greedy on scaffold groups, sorted by (n_samples desc, n_drugs desc, key lex)',
        'total_drugs': int(len(drugs)),
        'total_samples': int(adata.n_obs),
        'group_loads': [int(l) for l in group_loads],
        'max_deviation_pct': float(max_dev_pct),
        'per_fold_stats': per_fold_stats,
    }
    os.makedirs(os.path.dirname(META_OUT), exist_ok=True)
    with open(META_OUT, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"      Meta written to {META_OUT}")

    print(f"\n[SAVE] Writing h5ad back to {H5AD_PATH} ...")
    adata.write_h5ad(H5AD_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
