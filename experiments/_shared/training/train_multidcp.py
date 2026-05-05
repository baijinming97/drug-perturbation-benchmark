"""
MultiDCP training on the unified dataset (l1000_sdst).

Two engineering optimizations over a naive port of upstream main_deepce.py:
  1. Suppress RDKit deprecation warnings (eliminates 12GB+ stderr I/O)
  2. Pre-compute drug features for all unique SMILES once at startup
     (avoids repeated RDKit calls per batch per epoch)

Numerically equivalent to the naive port: same SMILES → same features →
same model output.

Original: Wu et al., PLoS Computational Biology 2022
Code: github.com/XieResearchGroup/MultiDCP

Training loop structure (per epoch, matching multidcp_ae.py L35-198):
  1. AE train → 2. AE validation → 3. Perturbed train →
  4. Perturbed validation (save best) → 5. AE test → 6. Perturbed test

Usage:
    python train_multidcp.py --fold 0
    python train_multidcp.py --fold 0 --ablation_mode zero
    python train_multidcp.py --fold 0 --ablation_mode shuffle
"""

# ── Optimization 1: suppress RDKit warnings BEFORE any import ────────────────
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
RDLogger.DisableLog('rdApp.*')

import os
from pathlib import Path
import sys
import re
import argparse
import json
import time
import random
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
import scanpy as sc
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

# ── Import original MultiDCP code via sys.path (zero modification) ──────────
# Add models/ and utils/ separately (original uses sys.path.append for both).
# Do NOT import datareader (requires pytorch_lightning) or
# multidcp_ae_utils (requires wandb).
MULTIDCP_ROOT = str(Path(__file__).resolve().parents[3] / "models" / "MultiDCP" / "MultiDCP")
sys.path.insert(0, os.path.join(MULTIDCP_ROOT, 'models'))
sys.path.insert(0, os.path.join(MULTIDCP_ROOT, 'utils'))

import multidcp
from data_utils import create_mask_feature
from molecules import Molecules, Node, node_id, degrees
from molecule_utils import atom_features, bond_features

# Original MultiDCP DATA_FILTER excludes these perturbagens before replicate
# aggregation (reference/MultiDCP/MultiDCP/multidcp_ae.py).
ORIGINAL_EXCLUDED_PERT_IDS = {
    'BRD-U41416256', 'BRD-U60236422', 'BRD-U01690642', 'BRD-U08759356',
    'BRD-U25771771', 'BRD-U33728988', 'BRD-U37049823', 'BRD-U44618005',
    'BRD-U44700465', 'BRD-U51951544', 'BRD-U66370498', 'BRD-U68942961',
    'BRD-U73238814', 'BRD-U82589721', 'BRD-U86922168', 'BRD-U97083655',
}

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[3]))
H5AD_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/l1000_sdst_78453.h5ad")
SMI_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/all_drugs_idx2smi_8981.npy")
GENE_VEC_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/PPI_gene_vector_128d.npy")
RESULTS_BASE = os.path.join(REPO_ROOT, "results")


def parse_args():
    p = argparse.ArgumentParser(description="MultiDCP cold-drug training")
    p.add_argument("--fold", type=int, required=True, help="Fold index 0-4")
    p.add_argument("--split_col", type=str, default=None,
                   help="Override split column name (default: nm_drug_blind_<fold+1>; pass e.g. nm_scaffold_1 to use scaffold-disjoint splits)")
    p.add_argument("--ablation_mode", type=str, default="none",
                   choices=["none", "zero", "shuffle"],
                   help="Drug ablation: none=original, zero=zero atom features, "
                        "shuffle=permute drug identity")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--h5ad_path", type=str, default=H5AD_DEFAULT)
    p.add_argument("--smi_path", type=str, default=SMI_DEFAULT)
    p.add_argument("--gene_vec_path", type=str, default=GENE_VEC_DEFAULT)
    p.add_argument("--ae_data_prefix", type=str, default=None,
                   help="Prefix for external AE CSV files (e.g. .../gene_expression_combat_norm_978_split4). "
                        "If not set, uses X_ctl from h5ad.")
    p.add_argument("--cell_ge_file", type=str, default=None,
                   help="Original MultiDCP CCLE/TCGA basal-expression CSV. Required when --gene_order_mode raw.")
    p.add_argument("--gene_order_mode", choices=["aligned", "raw"], default="aligned",
                   help="aligned keeps the existing h5ad/signature gene order; raw keeps original MultiDCP "
                        "AE and CCLE CSV column order for the cell/AE branches.")
    p.add_argument("--original_data_filter", action="store_true",
                   help="Apply original MultiDCP 24H + excluded-pert_id filter before replicate aggregation.")
    p.add_argument("--dedup_strategy", choices=["norm_rank", "original"], default="norm_rank",
                   help="Representative selection when --dedup is set. original matches data_utils.choose_mean_example.")
    p.add_argument("--original_sort", action="store_true",
                   help="Sort each split by original feature key before batching, matching data_utils.read_data().")
    p.add_argument("--dose_vocab", type=str,
                   default="0.04 um,0.12 um,0.37 um,1.11 um,3.33 um,10.0 um",
                   help="Comma-separated canonical dose levels (default: original MultiDCP 6 levels)")
    p.add_argument("--dev", type=str, default="cuda:0")
    # Original MultiDCP defaults (from multidcp_ae.py + train_multidcp_ae.sh)
    p.add_argument("--seed", type=int, default=343)
    p.add_argument("--max_epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="Stop if dev Pearson doesn't improve for this many epochs (0=disable)")
    p.add_argument("--min_epochs", type=int, default=50,
                   help="Minimum epochs before early stopping activates")
    p.add_argument("--dedup", action="store_true", default=False,
                   help="Deduplicate samples by (pert_id, cell_id, pert_idose), keeping median-rank representative")
    p.add_argument(
        "--output_space",
        choices=["deg", "native"],
        default="deg",
        help=(
            "Space written to test_predictions/test_ground_truth. "
            "Use 'deg' for unified XPert-style runs; use 'native' for original "
            "MultiDCP reproduction, where labels are already MODZ signatures and "
            "X_ctl stores CCLE/TCGA cell features."
        ),
    )
    return p.parse_args()


# ── Model registry (from multidcp_ae_utils.py initialize_model_registry) ────

def initialize_model_registry():
    """Exact copy of multidcp_ae_utils.py L7-23."""
    reg = defaultdict(lambda: None)
    reg['drug_input_dim'] = {'atom': 62, 'bond': 6}
    reg['drug_emb_dim'] = 128
    reg['conv_size'] = [16, 16]
    reg['degree'] = [0, 1, 2, 3, 4, 5]
    reg['gene_emb_dim'] = 128
    reg['gene_input_dim'] = 128
    reg['cell_id_input_dim'] = 978
    reg['cell_id_emb_dim'] = 50
    reg['cell_decoder_dim'] = 978
    reg['pert_idose_emb_dim'] = 4
    reg['hid_dim'] = 128
    reg['num_gene'] = 978
    reg['loss_type'] = 'point_wise_mse'
    reg['initializer'] = torch.nn.init.kaiming_uniform_
    return reg


# ── Optimization 2: pre-compute drug features ───────────────────────────────

def precompute_drug_cache(unique_smiles):
    """Parse each unique SMILES once via RDKit, cache features + connectivity."""
    cache = {}
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smi}")
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append({
                'idx': atom.GetIdx(),
                'features': atom_features(atom),
            })
        bonds = []
        for bond in mol.GetBonds():
            bonds.append({
                'idx': bond.GetIdx(),
                'begin': bond.GetBeginAtom().GetIdx(),
                'end': bond.GetEndAtom().GetIdx(),
                'features': bond_features(bond),
            })
        cache[smi] = {'atoms': atoms, 'bonds': bonds}
    return cache


def build_molecules_from_cache(smiles_batch, cache):
    """Build a Molecules batch object from pre-computed cache (no RDKit calls).

    Produces an object identical to Molecules(smiles_batch) but skips all RDKit
    parsing and feature extraction.
    """
    mol_obj = Molecules.__new__(Molecules)
    mol_obj.batch_size = len(smiles_batch)
    mol_obj.atom_dict = {}
    mol_obj.bond_dict = {}
    mol_obj.atom_list = []
    mol_obj.bond_list = []
    mol_obj.degree_nodelist = {}

    for batch_idx, smi in enumerate(smiles_batch):
        data = cache[smi]
        prefix = str(batch_idx)

        atom_nodes_by_orig_idx = {}
        for ad in data['atoms']:
            ext_id = node_id(prefix, node_id(smi, ad['idx']))
            node = Node('atom', ext_id, ad['features'])
            atom_nodes_by_orig_idx[ad['idx']] = node
            mol_obj.atom_dict[ext_id] = node
            mol_obj.atom_list.append(node)

        for bd in data['bonds']:
            src_node = atom_nodes_by_orig_idx[bd['begin']]
            tgt_node = atom_nodes_by_orig_idx[bd['end']]
            ext_id = node_id(prefix, node_id(smi, bd['idx']))
            bond_node = Node('bond', ext_id, bd['features'])
            bond_node.add_neighbors([src_node, tgt_node])
            src_node.add_neighbors([bond_node, tgt_node])
            tgt_node.add_neighbors([bond_node, src_node])
            mol_obj.bond_dict[ext_id] = bond_node
            mol_obj.bond_list.append(bond_node)

    mol_obj.sort_atom_by_degree()
    return mol_obj


def convert_smile_to_feature_cached(smiles_batch, device, cache):
    """Drop-in replacement for convert_smile_to_feature using pre-computed cache."""
    molecules = build_molecules_from_cache(smiles_batch, cache)
    node_repr = torch.FloatTensor(
        [node.data for node in molecules.get_node_list('atom')]
    ).to(device).double()
    edge_repr = torch.FloatTensor(
        [node.data for node in molecules.get_node_list('bond')]
    ).to(device).double()
    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}


# ── Batch iterators ─────────────────────────────────────────────────────────

def get_ae_batch_data(features, labels, batch_size, shuffle):
    """Yield (feature_batch, label_batch, dummy_idx) for AE training.

    Mirrors AEDataLoader.train_dataloader() behavior.
    """
    n = len(features)
    if shuffle:
        index = torch.randperm(n).long().numpy()
    for start_idx in range(0, n, batch_size):
        if shuffle:
            excerpt = index[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        dummy = torch.arange(len(features[excerpt])).long()
        yield features[excerpt], labels[excerpt], dummy


def get_perturbed_batch_data(smiles_arr, label_tensor, cell_gex_tensor,
                             pert_idose_tensor, batch_size, shuffle,
                             device, cache, ablation_mode="none"):
    """Yield (features_dict, label_batch, dummy_idx) for perturbed training.

    Mirrors PerturbedDataLoader.collate_fn() + train_dataloader().
    Uses pre-computed drug cache to avoid per-batch RDKit calls.
    """
    n = len(smiles_arr)
    if shuffle:
        index = torch.randperm(n).long().numpy()
    for start_idx in range(0, n, batch_size):
        if shuffle:
            excerpt = index[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        smiles_batch = smiles_arr[excerpt]
        drug_data = convert_smile_to_feature_cached(smiles_batch, device, cache)
        mask = create_mask_feature(drug_data, device)

        ft = dict()
        ft['drug'] = drug_data
        ft['mask'] = mask
        ft['cell_id'] = cell_gex_tensor[excerpt]
        ft['pert_idose'] = pert_idose_tensor[excerpt]

        dummy = torch.arange(len(smiles_batch)).long()
        yield ft, label_tensor[excerpt], dummy


# ── Metrics (local reimplementation, matches utils/metric.py) ───────────────

def bench_rmse(label, predict):
    return np.sqrt(np.mean((label - predict) ** 2))


def bench_correlation(label, predict, corr_type):
    corr_fn = pearsonr if corr_type == 'pearson' else spearmanr
    scores = [corr_fn(lb, pr)[0] for lb, pr in zip(label, predict)]
    return np.mean(scores), scores


def bench_precision_k(label, predict, k):
    """Matches original utils/metric.py precision_k exactly."""
    num_pos = 100
    num_neg = 100
    label_sorted = np.argsort(label, axis=1)
    predict_sorted = np.argsort(predict, axis=1)
    pk_neg, pk_pos = [], []
    for i in range(len(label)):
        neg_test = set(label_sorted[i, :num_neg])
        pos_test = set(label_sorted[i, -num_pos:])
        neg_pred = set(predict_sorted[i, :k])
        pos_pred = set(predict_sorted[i, -k:])
        pk_neg.append(len(neg_test & neg_pred) / k)
        pk_pos.append(len(pos_test & pos_pred) / k)
    return np.mean(pk_neg), np.mean(pk_pos)


def choose_mean_example_index(examples):
    """Return the replicate chosen by original data_utils.choose_mean_example."""
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value) ** 2
    return int(np.argmin(distance))


def eval_epoch(epoch_loss, lb_np, predict_np, steps, epoch, metrics_summary,
               job, split):
    """Compute and log metrics. Matches validation_epoch_end / test_epoch_end."""
    suffix = f'{job}_{split}'
    print(f'{job} {split.capitalize()} loss:')
    print(epoch_loss / steps)
    rmse_val = bench_rmse(lb_np, predict_np)
    metrics_summary[f'rmse_list_{suffix}'].append(rmse_val)
    print(f'{job} RMSE: {rmse_val}')
    pearson, _ = bench_correlation(lb_np, predict_np, 'pearson')
    metrics_summary[f'pearson_list_{suffix}'].append(pearson)
    print(f"{job} Pearson's correlation: {pearson}")
    spearman, _ = bench_correlation(lb_np, predict_np, 'spearman')
    metrics_summary[f'spearman_list_{suffix}'].append(spearman)
    print(f"{job} Spearman's correlation: {spearman}")
    precision = []
    for k in [10, 20, 50, 100]:
        pn, pp = bench_precision_k(lb_np, predict_np, k)
        print(f"{job} Precision@{k} Positive: {pp}")
        print(f"{job} Precision@{k} Negative: {pn}")
        precision.append([pp, pn])
    metrics_summary[f'precisionk_list_{suffix}'].append(precision)


def report_final_results(metrics_summary):
    """Matches multidcp_ae_utils.py report_final_results()."""
    for job, label in [('ae', 'AE'), ('perturbed', 'Perturbed')]:
        dev_key = f'pearson_list_{job}_dev'
        if not metrics_summary[dev_key]:
            continue
        best_dev = np.argmax(metrics_summary[dev_key])
        print(f"Epoch {best_dev+1} got best {label} Pearson on dev: "
              f"{metrics_summary[dev_key][best_dev]:.4f}")
        print(f"Epoch {best_dev+1} {label} Spearman on dev: "
              f"{metrics_summary[f'spearman_list_{job}_dev'][best_dev]:.4f}")
        print(f"Epoch {best_dev+1} {label} RMSE on dev: "
              f"{metrics_summary[f'rmse_list_{job}_dev'][best_dev]:.4f}")
        test_key = f'pearson_list_{job}_test'
        print(f"Epoch {best_dev+1} {label} Pearson on test w.r.t dev: "
              f"{metrics_summary[test_key][best_dev]:.4f}")
        print(f"Epoch {best_dev+1} {label} Spearman on test w.r.t dev: "
              f"{metrics_summary[f'spearman_list_{job}_test'][best_dev]:.4f}")
        print(f"Epoch {best_dev+1} {label} RMSE on test w.r.t dev: "
              f"{metrics_summary[f'rmse_list_{job}_test'][best_dev]:.4f}")
        best_test = np.argmax(metrics_summary[test_key])
        print(f"Epoch {best_test+1} got best {label} Pearson on test: "
              f"{metrics_summary[test_key][best_test]:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fold_idx = args.fold
    split_col = args.split_col or f"nm_drug_blind_{fold_idx + 1}"

    if args.output_dir:
        out_dir = args.output_dir
    elif args.ablation_mode == "none":
        out_dir = os.path.join(
            RESULTS_BASE, f"main_benchmark/multidcp/fold_{fold_idx}")
    else:
        out_dir = os.path.join(
            RESULTS_BASE, f"main_ablation/multidcp/{args.ablation_mode}/fold_{fold_idx}")

    print(f"{'='*60}")
    print(f"MultiDCP | Fold {fold_idx} | Split: {split_col}")
    print(f"Ablation: {args.ablation_mode}")
    print(f"Output: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    # ── Seed (matches multidcp_ae.py L225-228) ───────────────────────────────
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(args.dev)
    else:
        device = torch.device("cpu")
    print("Use GPU: %s" % torch.cuda.is_available())

    for d in ["checkpoints", "predictions", "logs", "metrics"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    adata = sc.read_h5ad(args.h5ad_path)
    idx2smi = np.load(args.smi_path, allow_pickle=True).item()
    gene_vec = np.load(args.gene_vec_path)  # (N_gene, 128)
    gene = torch.from_numpy(gene_vec.astype(np.float64)).to(device)
    num_gene = gene_vec.shape[0]
    print(f"  h5ad: {adata.shape}")
    print(f"  Gene vector: {gene_vec.shape} → num_gene={num_gene}")
    print(f"  SMILES dict: {len(idx2smi)} drugs")

    x_pert = adata.X
    if hasattr(x_pert, 'toarray'):
        x_pert = x_pert.toarray()
    x_ctrl = adata.obsm['X_ctl']
    if hasattr(x_ctrl, 'toarray'):
        x_ctrl = x_ctrl.toarray()

    pert_idx_all = adata.obs['pert_idx'].values.astype(int)
    cell_idx_all = adata.obs['cell_idx'].values.astype(int)
    if 'cell_mfc_name' in adata.obs.columns:
        cell_name_all = adata.obs['cell_mfc_name'].values.astype(str)
    elif 'cell_id' in adata.obs.columns:
        cell_name_all = adata.obs['cell_id'].values.astype(str)
    else:
        cell_name_all = cell_idx_all.astype(str)
    splits = adata.obs[split_col].values.astype(str)
    pert_idx_original = pert_idx_all.copy()

    if args.original_data_filter:
        keep = np.ones(len(adata), dtype=bool)
        if 'sig_id' in adata.obs.columns:
            sig_ids = adata.obs['sig_id'].values.astype(str)
            keep &= np.array(['24H' in sid for sid in sig_ids], dtype=bool)
        if 'pert_id' in adata.obs.columns:
            pert_ids = adata.obs['pert_id'].values.astype(str)
            keep &= ~np.isin(pert_ids, list(ORIGINAL_EXCLUDED_PERT_IDS))
        n_before = len(adata)
        adata = adata[keep].copy()
        x_pert = x_pert[keep]
        x_ctrl = x_ctrl[keep]
        pert_idx_all = pert_idx_all[keep]
        cell_idx_all = cell_idx_all[keep]
        cell_name_all = cell_name_all[keep]
        splits = splits[keep]
        pert_idx_original = pert_idx_original[keep]
        print(f"  Original DATA_FILTER: {n_before} → {len(adata)} samples")

    if args.dedup and 'pert_id' in adata.obs.columns and 'pert_idose' in adata.obs.columns:
        n_before = len(adata)
        group_keys = adata.obs[['pert_id', 'cell_idx', 'pert_idose']].astype(str).agg('|'.join, axis=1).values
        row_df = pd.DataFrame({'key': group_keys, 'row': np.arange(len(group_keys))})
        keep_idx = []
        if args.dedup_strategy == 'original':
            for _, grp in row_df.groupby('key', sort=True):
                rows = grp['row'].to_numpy()
                keep_idx.append(rows[choose_mean_example_index(x_pert[rows])])
        else:
            row_norms = np.linalg.norm(x_pert, axis=1)
            for _, grp in row_df.groupby('key', sort=True):
                rows = grp['row'].to_numpy()
                ranks = pd.Series(row_norms[rows]).rank().sub((len(rows) + 1) / 2).abs()
                keep_idx.append(rows[int(ranks.to_numpy().argmin())])
        keep_idx = sorted(keep_idx)
        keep = np.array(keep_idx, dtype=int)
        adata = adata[keep].copy()
        x_pert = x_pert[keep]
        x_ctrl = x_ctrl[keep]
        pert_idx_all = pert_idx_all[keep]
        cell_idx_all = cell_idx_all[keep]
        cell_name_all = cell_name_all[keep]
        splits = splits[keep]
        pert_idx_original = pert_idx_original[keep]
        print(f"  Dedup ({args.dedup_strategy}): {n_before} → {len(adata)} samples "
              f"({n_before - len(adata)} replicates removed)")

    # ── Step 2: Shuffle ablation ─────────────────────────────────────────────
    if args.ablation_mode == 'shuffle':
        print("  [ABLATION] Shuffling drug identity (seed=131419)")
        unique_drugs = np.unique(pert_idx_all)
        rng = np.random.default_rng(seed=131419)
        perm = rng.permutation(len(unique_drugs))
        drug_map = dict(zip(unique_drugs, unique_drugs[perm]))
        pert_idx_all = np.array([drug_map[d] for d in pert_idx_all])

    # ── Pre-compute drug feature cache ─────────────────────────────────────
    print("\n[1.5/6] Pre-computing drug features ...")
    t0 = time.time()
    used_drug_indices = np.unique(pert_idx_all)
    unique_smiles = list(set(idx2smi[int(i)] for i in used_drug_indices))
    drug_cache = precompute_drug_cache(unique_smiles)
    print(f"  Cached {len(drug_cache)} unique SMILES in {time.time()-t0:.1f}s")

    # ── Step 3: Build split data ─────────────────────────────────────────────
    print("\n[2/6] Building split data ...")
    train_mask = splits == 'train'
    valid_mask = splits == 'valid'
    test_mask = splits == 'test'
    print(f"  train={train_mask.sum()}, valid={valid_mask.sum()}, test={test_mask.sum()}")

    if args.gene_order_mode == "raw":
        if not args.cell_ge_file:
            raise ValueError("--cell_ge_file is required when --gene_order_mode raw")
        cell_ge_file = args.cell_ge_file if args.cell_ge_file.endswith(".csv") else args.cell_ge_file + ".csv"
        cell_ge_df = pd.read_csv(cell_ge_file, index_col=0)
        if cell_ge_df.shape[1] != num_gene:
            raise ValueError(f"Expected {num_gene} raw cell-gene columns, got {cell_ge_df.shape[1]}")
        cell_gex_dict = {}
        for cidx in np.unique(cell_idx_all):
            names = np.unique(cell_name_all[cell_idx_all == cidx])
            if len(names) != 1:
                raise ValueError(f"cell_idx={cidx} maps to multiple cell names: {names}")
            cell_name = names[0]
            if cell_name not in cell_ge_df.index:
                raise ValueError(f"Cell {cell_name} not found in {cell_ge_file}")
            cell_gex_dict[cidx] = cell_ge_df.loc[cell_name].values.astype(np.float64)
        print(f"  Cell basal expression: raw MultiDCP CSV order from {cell_ge_file}")
        print(f"  Raw cell-gene first columns: {list(cell_ge_df.columns[:5])}")
    else:
        # Cell basal expression: per-cell-line mean X_ctrl from training set
        # (existing dpb adapter behavior).
        cell_gex_dict = {}
        for cidx in np.unique(cell_idx_all[train_mask]):
            mask_c = cell_idx_all[train_mask] == cidx
            cell_gex_dict[cidx] = x_ctrl[train_mask][mask_c].mean(axis=0)
        # For cells only in valid/test (not in train), use their own mean X_ctrl
        for cidx in np.unique(cell_idx_all):
            if cidx not in cell_gex_dict:
                mask_c = cell_idx_all == cidx
                cell_gex_dict[cidx] = x_ctrl[mask_c].mean(axis=0)
        print(f"  Cell basal expression: aligned h5ad X_ctl order ({len(cell_gex_dict)} cell lines)")

    # Dose: read from h5ad, normalize, map to canonical vocabulary (matches original DATA_FILTER)
    dose_vocab = sorted(args.dose_vocab.split(','))
    DOSE_DIM = len(dose_vocab)
    dose_dict = {d: i for i, d in enumerate(dose_vocab)}

    def _normalize_dose(s):
        m = re.match(r'([\d.]+)\s*(um|uM)', s.strip(), re.IGNORECASE)
        return f"{float(m.group(1))} um" if m else s.strip()

    raw_doses = adata.obs['pert_idose'].values.astype(str)
    norm_doses = np.array([_normalize_dose(d) for d in raw_doses])
    dose_keep = np.array([d in dose_dict for d in norm_doses])
    n_dropped = (~dose_keep).sum()
    if n_dropped > 0:
        print(f"  Dose: dropped {n_dropped} samples with doses outside vocab")
        adata = adata[dose_keep].copy()
        norm_doses = norm_doses[dose_keep]
        x_pert = x_pert[dose_keep]
        x_ctrl = x_ctrl[dose_keep]
        pert_idx_all = pert_idx_all[dose_keep]
        pert_idx_original = pert_idx_original[dose_keep]
        cell_idx_all = cell_idx_all[dose_keep]
        cell_name_all = cell_name_all[dose_keep]
        train_mask = train_mask[dose_keep]
        valid_mask = valid_mask[dose_keep]
        test_mask = test_mask[dose_keep]

    dose_onehot_all = np.zeros((len(adata), DOSE_DIM), dtype=np.float64)
    for i, d in enumerate(norm_doses):
        dose_onehot_all[i, dose_dict[d]] = 1.0
    unique_doses = sorted(set(norm_doses))
    print(f"  Dose: {DOSE_DIM} levels in vocab, {len(unique_doses)} present in data: {unique_doses}")
    if 'pert_id' in adata.obs.columns:
        pert_key_all = adata.obs['pert_id'].values.astype(str)
    else:
        pert_key_all = pert_idx_all.astype(str)
    sample_sort_key = np.array([
        f"{pert},{cell},{dose}"
        for pert, cell, dose in zip(pert_key_all, cell_name_all, norm_doses)
    ])

    # AE data: external CSV files or X_ctl from h5ad
    if args.ae_data_prefix:
        print(f"  AE data: loading from external CSV: {args.ae_data_prefix}_{{train,dev,test}}.csv")
        h5ad_genes = list(adata.var_names)
        ae_train_df = pd.read_csv(args.ae_data_prefix + '_train.csv', index_col=0)
        ae_csv_genes = list(ae_train_df.columns)
        if args.gene_order_mode == "raw":
            print(f"  AE gene order: raw MultiDCP CSV order (no reindexing)")
            print(f"  Raw AE first columns: {ae_csv_genes[:5]}")
            ae_train = torch.from_numpy(ae_train_df.values.astype(np.float64)).to(device)
            ae_dev = torch.from_numpy(
                pd.read_csv(args.ae_data_prefix + '_dev.csv', index_col=0).values.astype(np.float64)
            ).to(device)
            ae_test = torch.from_numpy(
                pd.read_csv(args.ae_data_prefix + '_test.csv', index_col=0).values.astype(np.float64)
            ).to(device)
        elif ae_csv_genes != h5ad_genes:
            _GENE_ALIASES = {
                'ADCK3': 'COQ8A', 'FAM63A': 'MINDY1', 'HDGFRP3': 'HDGFL3',
                'HN1L': 'JPT2', 'IKBKAP': 'ELP1', 'KIAA0196': 'WASHC5',
                'KIAA0907': 'KHDC4', 'KIAA1033': 'WASHC4', 'LRRC16A': 'CARMIL1',
                'NARFL': 'CIAO3', 'PAPD7': 'TENT4A', 'PRUNE': 'PRUNE1',
                'SQRDL': 'SQOR', 'TMEM110': 'STIMATE', 'TMEM2': 'CEMIP2',
                'TMEM5': 'RXYLT1', 'TOMM70A': 'TOMM70',
            }
            _GENE_ALIASES.update({v: k for k, v in _GENE_ALIASES.items()})
            ae_csv_set = set(ae_csv_genes)
            def _resolve(g):
                if g in ae_csv_set:
                    return g
                return _GENE_ALIASES.get(g, g)
            reorder_idx = [ae_csv_genes.index(_resolve(g)) for g in h5ad_genes if _resolve(g) in ae_csv_set]
            assert len(reorder_idx) == len(h5ad_genes), \
                f"Gene mismatch: {len(reorder_idx)} overlap out of {len(h5ad_genes)}"
            n_aliased = sum(1 for g in h5ad_genes if g not in ae_csv_set and _GENE_ALIASES.get(g) in ae_csv_set)
            print(f"  AE gene order: reindexing {len(reorder_idx)} genes ({n_aliased} via alias)")
            ae_train = torch.from_numpy(
                ae_train_df.values[:, reorder_idx].astype(np.float64)).to(device)
            ae_dev = torch.from_numpy(
                pd.read_csv(args.ae_data_prefix + '_dev.csv', index_col=0).values[:, reorder_idx].astype(np.float64)
            ).to(device)
            ae_test = torch.from_numpy(
                pd.read_csv(args.ae_data_prefix + '_test.csv', index_col=0).values[:, reorder_idx].astype(np.float64)
            ).to(device)
        else:
            print(f"  AE gene order: already matches h5ad")
            ae_train = torch.from_numpy(ae_train_df.values.astype(np.float64)).to(device)
            ae_dev = torch.from_numpy(
                pd.read_csv(args.ae_data_prefix + '_dev.csv', index_col=0).values.astype(np.float64)
            ).to(device)
            ae_test = torch.from_numpy(
                pd.read_csv(args.ae_data_prefix + '_test.csv', index_col=0).values.astype(np.float64)
            ).to(device)
        del ae_train_df
        use_external_ae = True
    else:
        use_external_ae = False

    def build_split_data(mask):
        """Build arrays for one split."""
        idx = np.where(mask)[0]
        if args.original_sort:
            idx = idx[np.argsort(sample_sort_key[idx], kind="mergesort")]

        pidx = pert_idx_all[idx]
        smiles = np.array([idx2smi[int(i)] for i in pidx])
        label = torch.from_numpy(x_pert[idx].astype(np.float64)).to(device)

        cidx_arr = cell_idx_all[idx]
        cell_gex = np.stack([cell_gex_dict[c] for c in cidx_arr]).astype(np.float64)
        cell_gex_t = torch.from_numpy(cell_gex).to(device)

        dose_t = torch.from_numpy(dose_onehot_all[idx]).to(device)

        if not use_external_ae:
            ae_data = torch.from_numpy(x_ctrl[idx].astype(np.float64)).to(device)
        else:
            ae_data = None

        return smiles, label, cell_gex_t, dose_t, ae_data, idx

    train_smiles, train_label, train_cell_gex, train_dose, ae_train_split, train_idx = \
        build_split_data(train_mask)
    dev_smiles, dev_label, dev_cell_gex, dev_dose, ae_dev_split, dev_idx = \
        build_split_data(valid_mask)
    test_smiles, test_label, test_cell_gex, test_dose, ae_test_split, test_idx = \
        build_split_data(test_mask)

    if not use_external_ae:
        ae_train = ae_train_split
        ae_dev = ae_dev_split
        ae_test = ae_test_split

    print(f"  #Train: {len(train_smiles)}, #Dev: {len(dev_smiles)}, #Test: {len(test_smiles)}")
    print(f"  #AE Train: {len(ae_train)}, #AE Dev: {len(ae_dev)}, #AE Test: {len(ae_test)}")

    # ── Step 4: Create model (matches multidcp_ae.py L250-261) ───────────────
    print("\n[3/6] Creating model ...")
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
        'num_gene': num_gene,
        'pert_idose_input_dim': DOSE_DIM,  # 6, same as original
        'dropout': args.dropout,
        'linear_encoder_flag': False,  # TransformerEncoder (default)
    })

    model = multidcp.MultiDCP_AE(device=device,
                                  model_param_registry=model_param_registry)
    model.init_weights(pretrained=None)
    model.to(device)
    model = model.double()

    # Zero ablation: register forward hook to zero NeuralFingerprint output
    if args.ablation_mode == 'zero':
        print("[ABLATION] Registering forward hook to zero drug encoder output")
        def zero_drug_output(module, input, output):
            if not hasattr(zero_drug_output, '_logged'):
                print(f"[HOOK] drug_fp output zeroed: shape={output.shape}, "
                      f"original_max={output.abs().max():.4f}")
                zero_drug_output._logged = True
            return torch.zeros_like(output)
        model.multidcp.drug_fp.register_forward_hook(zero_drug_output)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Encoder: TransformerEncoder (linear_encoder_flag=False)")

    # ── Step 5: Training (matches multidcp_ae.py model_training L30-198) ─────
    es_info = (f", early stop patience={args.early_stop_patience} after epoch {args.min_epochs}"
               if args.early_stop_patience > 0 else "")
    print(f"\n[4/6] Training ({args.max_epoch} epochs, AE+Perturbed joint{es_info}) ...")
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # L32
    best_dev_pearson = float("-inf")  # L33
    best_dev_epoch = 0
    best_model_state = None

    metrics_summary = defaultdict(list)
    batch_size = args.batch_size
    train_loss_list = []

    for epoch in range(args.max_epoch):
        print("Iteration %d:" % epoch)
        for pg in optimizer.param_groups:
            print("============current learning rate is {0!r}".format(pg['lr']))

        # ── AE Train (L39-56) ──
        model.train()
        epoch_loss = 0
        for i, (feature, label, _) in enumerate(get_ae_batch_data(
                ae_train, ae_train, batch_size, shuffle=True)):
            optimizer.zero_grad()
            predict, cell_hidden_ = model(input_cell_gex=feature,
                                          job_id='ae', epoch=epoch)
            loss_t = model.loss(label, predict)
            loss_t.backward()
            optimizer.step()
            epoch_loss += loss_t.item()
        print('AE Train loss:')
        print(epoch_loss / (i + 1))

        # ── AE Validation (L58-72) ──
        model.eval()
        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, (feature, label, _) in enumerate(get_ae_batch_data(
                    ae_dev, ae_dev, batch_size, shuffle=False)):
                predict, _ = model(input_cell_gex=feature,
                                   job_id='ae', epoch=epoch)
                loss = model.loss(label, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            eval_epoch(epoch_loss, lb_np, predict_np, i + 1, epoch,
                       metrics_summary, 'ae', 'dev')

        # ── Perturbed Train (L74-100) ──
        model.train()
        epoch_loss = 0
        for i, (ft, lb, _) in enumerate(get_perturbed_batch_data(
                train_smiles, train_label, train_cell_gex, train_dose,
                batch_size, shuffle=True, device=device,
                cache=drug_cache, ablation_mode=args.ablation_mode)):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            optimizer.zero_grad()
            predict, cell_hidden_ = model(input_cell_gex=cell_feature,
                                          input_drug=drug, input_gene=gene,
                                          mask=mask, input_pert_idose=pert_idose,
                                          job_id='perturbed', epoch=epoch)
            loss_t = model.loss(lb, predict)
            loss_t.backward()
            optimizer.step()
            epoch_loss += loss_t.item()
        print('Perturbed gene expression profile Train loss:')
        print(epoch_loss / (i + 1))
        train_loss_list.append(epoch_loss / (i + 1))

        # ── Perturbed Validation (L102-128) ──
        model.eval()
        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(get_perturbed_batch_data(
                    dev_smiles, dev_label, dev_cell_gex, dev_dose,
                    batch_size, shuffle=False, device=device,
                    cache=drug_cache, ablation_mode=args.ablation_mode)):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(input_cell_gex=cell_feature,
                                   input_drug=drug, input_gene=gene,
                                   mask=mask, input_pert_idose=pert_idose,
                                   job_id='perturbed', epoch=epoch)
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            eval_epoch(epoch_loss, lb_np, predict_np, i + 1, epoch,
                       metrics_summary, 'perturbed', 'dev')

            # L125-128: save best model (note: `or epoch == 1` matches original)
            if best_dev_pearson < metrics_summary['pearson_list_perturbed_dev'][-1] \
                    or epoch == 1:
                best_dev_pearson = metrics_summary['pearson_list_perturbed_dev'][-1]
                best_dev_epoch = epoch
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

        # ── AE Test (L131-153) ──
        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, (feature, label, _) in enumerate(get_ae_batch_data(
                    ae_test, ae_test, batch_size, shuffle=False)):
                predict, hidden = model(input_cell_gex=feature, job_id='ae')
                loss = model.loss(label, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            eval_epoch(epoch_loss, lb_np, predict_np, i + 1, epoch,
                       metrics_summary, 'ae', 'test')

        # ── Perturbed Test (L155-198) ──
        epoch_loss = 0
        lb_np_ls = []
        predict_np_ls = []
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(get_perturbed_batch_data(
                    test_smiles, test_label, test_cell_gex, test_dose,
                    batch_size, shuffle=False, device=device,
                    cache=drug_cache, ablation_mode=args.ablation_mode)):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(input_cell_gex=cell_feature,
                                   input_drug=drug, input_gene=gene,
                                   mask=mask, input_pert_idose=pert_idose,
                                   job_id='perturbed')
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np_ls.append(lb.cpu().numpy())
                predict_np_ls.append(predict.cpu().numpy())

            lb_np = np.concatenate(lb_np_ls, axis=0)
            predict_np = np.concatenate(predict_np_ls, axis=0)
            eval_epoch(epoch_loss, lb_np, predict_np, i + 1, epoch,
                       metrics_summary, 'perturbed', 'test')

        # ── Early stopping ──
        if args.early_stop_patience > 0 \
                and epoch >= args.min_epochs \
                and epoch - best_dev_epoch >= args.early_stop_patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement since epoch {best_dev_epoch})")
            break

    train_time = time.time() - start_time

    # ── Report (matches multidcp_ae_utils.py report_final_results) ───────────
    report_final_results(metrics_summary)
    print(f"Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    torch.save(best_model_state,
               os.path.join(out_dir, "checkpoints", "best_model.pt"))

    # ── Save training curve ──────────────────────────────────────────────────
    n_epochs = len(train_loss_list)
    curve_df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'train_loss': train_loss_list,
        'dev_pearson': metrics_summary['pearson_list_perturbed_dev'][:n_epochs],
        'dev_spearman': metrics_summary['spearman_list_perturbed_dev'][:n_epochs],
        'dev_rmse': metrics_summary['rmse_list_perturbed_dev'][:n_epochs],
        'test_pearson': metrics_summary['pearson_list_perturbed_test'][:n_epochs],
        'test_spearman': metrics_summary['spearman_list_perturbed_test'][:n_epochs],
        'test_rmse': metrics_summary['rmse_list_perturbed_test'][:n_epochs],
        'ae_dev_pearson': metrics_summary['pearson_list_ae_dev'][:n_epochs],
        'ae_test_pearson': metrics_summary['pearson_list_ae_test'][:n_epochs],
    })
    curve_df.to_csv(os.path.join(out_dir, "logs", "training_curve.csv"), index=False)

    # ── [5/6] Predict on test set with best model ────────────────────────────
    best_dev_epoch = np.argmax(metrics_summary['pearson_list_perturbed_dev'])
    print(f"\n[5/6] Predicting on test set with best model (epoch {best_dev_epoch + 1}) ...")
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(get_perturbed_batch_data(
                test_smiles, test_label, test_cell_gex, test_dose,
                batch_size, shuffle=False, device=device,
                cache=drug_cache, ablation_mode=args.ablation_mode)):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            predict, _ = model(input_cell_gex=cell_feature,
                               input_drug=drug, input_gene=gene,
                               mask=mask, input_pert_idose=pert_idose,
                               job_id='perturbed')
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

    # ── [6/6] Save unified output ────────────────────────────────────────────
    print(f"\n[6/6] Saving unified output ...")
    pred_dir = os.path.join(out_dir, "predictions")

    x_ctrl_test = x_ctrl[test_idx].astype(np.float64)
    if args.output_space == "deg":
        test_predictions = (predict_np - x_ctrl_test).astype(np.float32)
        test_ground_truth = (lb_np - x_ctrl_test).astype(np.float32)
        test_ctrl_out = x_ctrl_test.astype(np.float32)
    else:
        test_predictions = predict_np.astype(np.float32)
        test_ground_truth = lb_np.astype(np.float32)
        test_ctrl_out = np.zeros_like(test_ground_truth, dtype=np.float32)

    np.save(os.path.join(pred_dir, "test_predictions.npy"), test_predictions)
    np.save(os.path.join(pred_dir, "test_ground_truth.npy"), test_ground_truth)
    np.save(os.path.join(pred_dir, "test_ctrl.npy"), test_ctrl_out)

    train_x_pert_mean = x_pert[train_idx].mean(axis=0).astype(np.float32)
    np.save(os.path.join(pred_dir, "train_x_pert_mean.npy"), train_x_pert_mean)

    sample_ids = pd.DataFrame({
        'drug_id': pert_idx_original[test_idx],
        'cell_id': cell_idx_all[test_idx],
    })
    sample_ids.to_csv(os.path.join(pred_dir, "test_sample_ids.csv"), index=False)

    drug_ids = sample_ids['drug_id'].values
    unique_drugs = np.unique(drug_ids)
    drug_pccs = []
    for drug in unique_drugs:
        dmask = drug_ids == drug
        pccs = []
        for j in np.where(dmask)[0]:
            gt, pr = test_ground_truth[j], test_predictions[j]
            if np.std(gt) > 0 and np.std(pr) > 0:
                r, _ = pearsonr(gt, pr)
                pccs.append(r)
        if pccs:
            drug_pccs.append({'drug_id': int(drug), 'pcc': float(np.mean(pccs)),
                              'n_samples': int(dmask.sum())})
    pd.DataFrame(drug_pccs).to_csv(
        os.path.join(pred_dir, "per_drug_pcc.csv"), index=False)

    resource_log = {
        "model": "multidcp",
        "fold": fold_idx,
        "split_col": split_col,
        "ablation_mode": args.ablation_mode,
        "seed": args.seed,
        "best_dev_epoch": int(best_dev_epoch) + 1,
        "best_dev_pearson": float(metrics_summary['pearson_list_perturbed_dev'][best_dev_epoch]),
        "best_test_pearson_wrt_dev": float(metrics_summary['pearson_list_perturbed_test'][best_dev_epoch]),
        "n_epochs": args.max_epoch,
        "n_epochs_run": int(n_epochs),
        "max_epoch": int(args.max_epoch),
        "early_stop_patience": int(args.early_stop_patience),
        "min_epochs": int(args.min_epochs),
        "output_space": args.output_space,
        "dose_vocab": args.dose_vocab,
        "gene_order_mode": args.gene_order_mode,
        "cell_ge_file": args.cell_ge_file,
        "ae_data_prefix": args.ae_data_prefix,
        "original_data_filter": bool(args.original_data_filter),
        "dedup": bool(args.dedup),
        "dedup_strategy": args.dedup_strategy if args.dedup else "none",
        "original_sort": bool(args.original_sort),
        "wall_time_s": int(train_time),
        "param_count": param_count,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "n_test": int(test_mask.sum()),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "n_cell_lines": len(cell_gex_dict),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "logs", "resource_log.json"), 'w') as f:
        json.dump(resource_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} complete!")
    print(f"  Best dev epoch: {best_dev_epoch + 1}")
    print(f"  Dev Pearson: {metrics_summary['pearson_list_perturbed_dev'][best_dev_epoch]:.4f}")
    print(f"  Test Pearson (w.r.t. dev): {metrics_summary['pearson_list_perturbed_test'][best_dev_epoch]:.4f}")
    print(f"  Predictions: {test_predictions.shape}")
    print(f"  Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
