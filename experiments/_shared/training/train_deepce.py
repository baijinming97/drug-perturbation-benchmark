"""
DeepCE training — faithful to original main_deepce.py.

Supports pert_type, cell_id, pert_idose auto-detection from data
(matching original transfrom_to_tensor logic). main_benchmark / main_ablation
data has single dose/pert_type so those are disabled; original-dataset data
has 6 doses so pert_idose is enabled automatically.

Usage:
    python experiments/_shared/training/train_deepce.py --fold 0 --split_col fixed_split --seed 343 \
        --h5ad_path .../deepce_original.h5ad --smi_path .../idx2smi.npy
"""

# Suppress RDKit's verbose C++ logger BEFORE any rdkit-using import
# (otherwise GetExplicitValence deprecation spams stderr ~5M lines/run).
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
RDLogger.DisableLog('rdApp.*')

import os
import sys
import argparse
import json
import time
import random
from datetime import datetime

from pathlib import Path
import numpy as np
import torch
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr

# ── Import original DeepCE code via sys.path (zero modification) ─────────────
DEEPCE_ROOT = str(Path(__file__).resolve().parents[3] / "models" / "DeepCE" / "DeepCE")
sys.path.insert(0, DEEPCE_ROOT)

from models import DeepCE
from utils.data_utils import convert_smile_to_feature, create_mask_feature

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[3]))
H5AD_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/l1000_sdst_78453.h5ad")
SMI_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/all_drugs_idx2smi_8981.npy")
GENE_VEC_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/PPI_gene_vector_128d.npy")
RESULTS_BASE = os.path.join(REPO_ROOT, "results")


def parse_args():
    p = argparse.ArgumentParser(description="DeepCE training")
    p.add_argument("--fold", type=int, required=True, help="Fold index 0-4")
    p.add_argument("--ablation_mode", type=str, default="none",
                   choices=["none", "zero", "shuffle"],
                   help="Drug ablation: none=original, zero=zero atom features, "
                        "shuffle=permute drug identity")
    p.add_argument("--split_col", type=str, default=None,
                   help="Override split column name (use 'fixed_split' for original_reproduction/original_ablation)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--h5ad_path", type=str, default=H5AD_DEFAULT)
    p.add_argument("--smi_path", type=str, default=SMI_DEFAULT)
    p.add_argument("--gene_vec_path", type=str, default=GENE_VEC_DEFAULT)
    p.add_argument("--dev", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=343)
    p.add_argument("--max_epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


# ── Batch iterator (matches datareader.py get_batch_data) ────────────────────

def get_batch_data(smiles_arr, label_tensor, feature_tensors,
                   batch_size, shuffle, device, ablation_mode="none"):
    """Yield (features_dict, label_slice) batches.

    feature_tensors is a dict with optional keys: 'pert_type', 'cell_id', 'pert_idose'.
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
        drug_data = convert_smile_to_feature(smiles_batch, device)
        mask = create_mask_feature(drug_data, device)

        output = dict()
        output['drug'] = drug_data
        output['mask'] = mask
        for key in ('pert_type', 'cell_id', 'pert_idose'):
            if key in feature_tensors:
                output[key] = feature_tensors[key][excerpt]

        yield output, label_tensor[excerpt]


# ── Metrics ──────────────────────────────────────────────────────────────────

def bench_rmse(label, predict):
    return np.sqrt(np.mean((label - predict) ** 2))


def bench_correlation(label, predict, corr_type):
    corr_fn = pearsonr if corr_type == 'pearson' else spearmanr
    scores = [corr_fn(lb, pr)[0] for lb, pr in zip(label, predict)]
    return np.mean(scores), scores


def bench_precision_k(label, predict, k):
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fold_idx = args.fold
    split_col = args.split_col or f"nm_drug_blind_{fold_idx + 1}"

    if args.output_dir:
        out_dir = args.output_dir
    elif args.ablation_mode == "none":
        out_dir = os.path.join(
            RESULTS_BASE, f"main_benchmark/deepce/fold_{fold_idx}")
    else:
        out_dir = os.path.join(
            RESULTS_BASE, f"main_ablation/deepce/{args.ablation_mode}/fold_{fold_idx}")

    print(f"{'='*60}")
    print(f"DeepCE | Fold {fold_idx} | Split: {split_col}")
    print(f"Ablation: {args.ablation_mode}")
    print(f"Output: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    # ── Seed (matches datareader.py L3-5) ────────────────────────────────────
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
    print("\n[1/5] Loading data ...")
    adata = sc.read_h5ad(args.h5ad_path)
    idx2smi = np.load(args.smi_path, allow_pickle=True).item()
    gene_vec = np.load(args.gene_vec_path)
    gene = torch.from_numpy(gene_vec.astype(np.float64)).to(device)
    print(f"  h5ad: {adata.shape}")
    print(f"  Gene vector: {gene_vec.shape}")
    print(f"  SMILES dict: {len(idx2smi)} drugs")

    x_pert = adata.X
    if hasattr(x_pert, 'toarray'):
        x_pert = x_pert.toarray()
    x_ctrl = adata.obsm['X_ctl']
    if hasattr(x_ctrl, 'toarray'):
        x_ctrl = x_ctrl.toarray()

    pert_idx_all = adata.obs['pert_idx'].values.astype(int)
    cell_idx_all = adata.obs['cell_idx'].values.astype(int)
    splits = adata.obs[split_col].values.astype(str)

    num_gene = gene_vec.shape[0]
    precision_degree = [10, 20, 50, 100]

    pert_idx_original = pert_idx_all.copy()

    # ── Shuffle ablation ─────────────────────────────────────────────────────
    if args.ablation_mode == 'shuffle':
        print("  [ABLATION] Shuffling drug identity (seed=131419)")
        unique_drugs = np.unique(pert_idx_all)
        rng = np.random.default_rng(seed=131419)
        perm = rng.permutation(len(unique_drugs))
        drug_map = dict(zip(unique_drugs, unique_drugs[perm]))
        pert_idx_all = np.array([drug_map[d] for d in pert_idx_all])

    # ── Step 2: Build split data (matches transfrom_to_tensor) ───────────────
    print("\n[2/5] Building split data ...")
    train_mask = splits == 'train'
    valid_mask = splits == 'valid'
    test_mask = splits == 'test'
    print(f"  train={train_mask.sum()}, valid={valid_mask.sum()}, test={test_mask.sum()}")

    # Auto-detect pert_type, cell_id, pert_idose (matches original L119-146)
    has_pert_type_col = 'pert_type' in adata.obs.columns
    has_pert_idose_col = 'pert_idose' in adata.obs.columns

    if has_pert_type_col:
        pert_type_all = adata.obs['pert_type'].values.astype(str)
        pert_type_set = sorted(list(set(pert_type_all[train_mask])))
        use_pert_type = len(pert_type_set) > 1
    else:
        use_pert_type = False

    cell_id_set = sorted(list(set(cell_idx_all[train_mask].tolist())))
    cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
    n_cells = len(cell_id_set)
    use_cell_id = n_cells > 1

    if has_pert_idose_col:
        pert_idose_all = adata.obs['pert_idose'].values.astype(str)
        pert_idose_set = sorted(list(set(pert_idose_all[train_mask])))
        use_pert_idose = len(pert_idose_set) > 1
    else:
        use_pert_idose = False

    if use_pert_type:
        pert_type_dict = dict(zip(pert_type_set, list(range(len(pert_type_set)))))
        n_pert_types = len(pert_type_set)
    if use_pert_idose:
        pert_idose_dict = dict(zip(pert_idose_set, list(range(len(pert_idose_set)))))
        n_pert_idoses = len(pert_idose_set)

    print(f"  Cell lines: {n_cells} unique (use_cell_id={use_cell_id})")
    print(f"  use_pert_type={use_pert_type}" +
          (f" ({len(pert_type_set)}: {pert_type_set})" if use_pert_type else ""))
    print(f"  use_pert_idose={use_pert_idose}" +
          (f" ({len(pert_idose_set)}: {pert_idose_set})" if use_pert_idose else ""))

    def build_split_data(mask):
        """Build SMILES array, label tensor, feature tensors dict."""
        pidx = pert_idx_all[mask]
        smiles = np.array([idx2smi[int(i)] for i in pidx])
        label = torch.from_numpy(x_pert[mask].astype(np.float64)).to(device)

        features = {}

        if use_pert_type:
            pt = pert_type_all[mask]
            onehot = np.zeros((len(pt), n_pert_types), dtype=np.float64)
            for j, v in enumerate(pt):
                if v in pert_type_dict:
                    onehot[j, pert_type_dict[v]] = 1.0
            features['pert_type'] = torch.from_numpy(onehot).to(device)

        if use_cell_id:
            cidx = cell_idx_all[mask]
            onehot = np.zeros((len(cidx), n_cells), dtype=np.float64)
            for j, c in enumerate(cidx):
                if c in cell_id_dict:
                    onehot[j, cell_id_dict[c]] = 1.0
            features['cell_id'] = torch.from_numpy(onehot).to(device)

        if use_pert_idose:
            pd_arr = pert_idose_all[mask]
            onehot = np.zeros((len(pd_arr), n_pert_idoses), dtype=np.float64)
            for j, v in enumerate(pd_arr):
                if v in pert_idose_dict:
                    onehot[j, pert_idose_dict[v]] = 1.0
            features['pert_idose'] = torch.from_numpy(onehot).to(device)

        return smiles, label, features

    train_smiles, train_label, train_ft = build_split_data(train_mask)
    dev_smiles, dev_label, dev_ft = build_split_data(valid_mask)
    test_smiles, test_label, test_ft = build_split_data(test_mask)

    print(f"  #Train: {len(train_smiles)}")
    print(f"  #Dev: {len(dev_smiles)}")
    print(f"  #Test: {len(test_smiles)}")

    # ── Step 3: Create model (matches main_deepce.py L71-81) ─────────────────
    print("\n[3/5] Creating model ...")
    model = DeepCE(
        drug_input_dim={'atom': 62, 'bond': 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_input_dim=gene_vec.shape[1],
        gene_emb_dim=128,
        num_gene=num_gene,
        hid_dim=128,
        dropout=args.dropout,
        loss_type='point_wise_mse',
        device=device,
        initializer=torch.nn.init.xavier_uniform_,
        pert_type_input_dim=n_pert_types if use_pert_type else 1,
        cell_id_input_dim=n_cells,
        pert_idose_input_dim=n_pert_idoses if use_pert_idose else 1,
        pert_type_emb_dim=4,
        cell_id_emb_dim=4,
        pert_idose_emb_dim=4,
        use_pert_type=use_pert_type,
        use_cell_id=use_cell_id,
        use_pert_idose=use_pert_idose,
    )
    model.to(device)
    model = model.double()

    if args.ablation_mode == 'zero':
        print("[ABLATION] Registering forward hook to zero drug encoder output")
        def zero_drug_output(module, input, output):
            if not hasattr(zero_drug_output, '_logged'):
                print(f"[HOOK] drug_fp output zeroed: shape={output.shape}, "
                      f"original_max={output.abs().max():.4f}")
                zero_drug_output._logged = True
            return torch.zeros_like(output)
        model.drug_fp.register_forward_hook(zero_drug_output)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  linear_dim: {model.linear_dim}")

    # ── Step 4: Training (matches main_deepce.py L84-231) ────────────────────
    print(f"\n[4/5] Training ({args.max_epoch} epochs, no early stopping) ...")
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dev_loss = float("inf")
    best_dev_pearson = float("-inf")
    best_model_state = None

    pearson_list_dev = []
    pearson_list_test = []
    spearman_list_dev = []
    spearman_list_test = []
    rmse_list_dev = []
    rmse_list_test = []
    precisionk_list_dev = []
    precisionk_list_test = []
    train_loss_list = []

    batch_size = args.batch_size

    def extract_features(ft_dict):
        """Extract pert_type, cell_id, pert_idose from batch dict."""
        pt = ft_dict.get('pert_type') if use_pert_type else None
        ci = ft_dict.get('cell_id') if use_cell_id else None
        pi = ft_dict.get('pert_idose') if use_pert_idose else None
        return pt, ci, pi

    for epoch in range(args.max_epoch):
        print("Iteration %d:" % (epoch + 1))

        model.train()
        epoch_loss = 0
        for i, batch in enumerate(get_batch_data(
                train_smiles, train_label, train_ft,
                batch_size, shuffle=True, device=device,
                ablation_mode=args.ablation_mode)):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            pert_type, cell_id, pert_idose = extract_features(ft)
            optimizer.zero_grad()
            predict = model(drug, gene, mask, pert_type, cell_id, pert_idose)
            loss = model.loss(lb, predict)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Train loss:')
        print(epoch_loss / (i + 1))
        train_loss_list.append(epoch_loss / (i + 1))

        model.eval()

        # ── Eval dev ──
        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, batch in enumerate(get_batch_data(
                    dev_smiles, dev_label, dev_ft,
                    batch_size, shuffle=False, device=device,
                    ablation_mode=args.ablation_mode)):
                ft, lb = batch
                drug = ft['drug']
                mask = ft['mask']
                pert_type, cell_id, pert_idose = extract_features(ft)
                predict = model(drug, gene, mask, pert_type, cell_id, pert_idose)
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            print('Dev loss:')
            print(epoch_loss / (i + 1))
            rmse_score = bench_rmse(lb_np, predict_np)
            rmse_list_dev.append(rmse_score)
            print('RMSE: %.4f' % rmse_score)
            pearson, _ = bench_correlation(lb_np, predict_np, 'pearson')
            pearson_list_dev.append(pearson)
            print("Pearson's correlation: %.4f" % pearson)
            spearman, _ = bench_correlation(lb_np, predict_np, 'spearman')
            spearman_list_dev.append(spearman)
            print("Spearman's correlation: %.4f" % spearman)
            precision = []
            for k in precision_degree:
                precision_neg, precision_pos = bench_precision_k(lb_np, predict_np, k)
                print("Precision@%d Positive: %.4f" % (k, precision_pos))
                print("Precision@%d Negative: %.4f" % (k, precision_neg))
                precision.append([precision_pos, precision_neg])
            precisionk_list_dev.append(precision)

            if best_dev_pearson < pearson:
                best_dev_pearson = pearson
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ── Eval test ──
        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, batch in enumerate(get_batch_data(
                    test_smiles, test_label, test_ft,
                    batch_size, shuffle=False, device=device,
                    ablation_mode=args.ablation_mode)):
                ft, lb = batch
                drug = ft['drug']
                mask = ft['mask']
                pert_type, cell_id, pert_idose = extract_features(ft)
                predict = model(drug, gene, mask, pert_type, cell_id, pert_idose)
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            print('Test loss:')
            print(epoch_loss / (i + 1))
            rmse_score = bench_rmse(lb_np, predict_np)
            rmse_list_test.append(rmse_score)
            print('RMSE: %.4f' % rmse_score)
            pearson, _ = bench_correlation(lb_np, predict_np, 'pearson')
            pearson_list_test.append(pearson)
            print("Pearson's correlation: %.4f" % pearson)
            spearman, _ = bench_correlation(lb_np, predict_np, 'spearman')
            spearman_list_test.append(spearman)
            print("Spearman's correlation: %.4f" % spearman)
            precision = []
            for k in precision_degree:
                precision_neg, precision_pos = bench_precision_k(lb_np, predict_np, k)
                print("Precision@%d Positive: %.4f" % (k, precision_pos))
                print("Precision@%d Negative: %.4f" % (k, precision_neg))
                precision.append([precision_pos, precision_neg])
            precisionk_list_test.append(precision)

    train_time = time.time() - start_time

    # ── Report best epoch ────────────────────────────────────────────────────
    best_dev_epoch = np.argmax(pearson_list_dev)
    print("Epoch %d got best Pearson's correlation on dev set: %.4f" % (
        best_dev_epoch + 1, pearson_list_dev[best_dev_epoch]))
    print("Epoch %d got Spearman's correlation on dev set: %.4f" % (
        best_dev_epoch + 1, spearman_list_dev[best_dev_epoch]))
    print("Epoch %d got RMSE on dev set: %.4f" % (
        best_dev_epoch + 1, rmse_list_dev[best_dev_epoch]))
    print("Epoch %d got P@100 POS and NEG on dev set: %.4f, %.4f" % (
        best_dev_epoch + 1,
        precisionk_list_dev[best_dev_epoch][-1][0],
        precisionk_list_dev[best_dev_epoch][-1][1]))

    print("Epoch %d got Pearson's correlation on test set w.r.t dev set: %.4f" % (
        best_dev_epoch + 1, pearson_list_test[best_dev_epoch]))
    print("Epoch %d got Spearman's correlation on test set w.r.t dev set: %.4f" % (
        best_dev_epoch + 1, spearman_list_test[best_dev_epoch]))
    print("Epoch %d got RMSE on test set w.r.t dev set: %.4f" % (
        best_dev_epoch + 1, rmse_list_test[best_dev_epoch]))
    print("Epoch %d got P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (
        best_dev_epoch + 1,
        precisionk_list_test[best_dev_epoch][-1][0],
        precisionk_list_test[best_dev_epoch][-1][1]))

    best_test_epoch = np.argmax(pearson_list_test)
    print("Epoch %d got best Pearson's correlation on test set: %.4f" % (
        best_test_epoch + 1, pearson_list_test[best_test_epoch]))
    print("Epoch %d got Spearman's correlation on test set: %.4f" % (
        best_test_epoch + 1, spearman_list_test[best_test_epoch]))
    print("Epoch %d got RMSE on test set: %.4f" % (
        best_test_epoch + 1, rmse_list_test[best_test_epoch]))
    print("Epoch %d got P@100 POS and NEG on test set: %.4f, %.4f" % (
        best_test_epoch + 1,
        precisionk_list_test[best_test_epoch][-1][0],
        precisionk_list_test[best_test_epoch][-1][1]))
    print(f"Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    torch.save(best_model_state, os.path.join(out_dir, "checkpoints", "best_model.pt"))

    # ── Save training curve ──────────────────────────────────────────────────
    curve_df = pd.DataFrame({
        'epoch': range(1, len(train_loss_list) + 1),
        'train_loss': train_loss_list,
        'dev_pearson': pearson_list_dev,
        'dev_spearman': spearman_list_dev,
        'dev_rmse': rmse_list_dev,
        'test_pearson': pearson_list_test,
        'test_spearman': spearman_list_test,
        'test_rmse': rmse_list_test,
    })
    curve_df.to_csv(os.path.join(out_dir, "logs", "training_curve.csv"), index=False)

    # ── Predict with best model ──────────────────────────────────────────────
    print(f"\n[5/5] Predicting on test set with best model (epoch {best_dev_epoch + 1}) ...")
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, batch in enumerate(get_batch_data(
                test_smiles, test_label, test_ft,
                batch_size, shuffle=False, device=device,
                ablation_mode=args.ablation_mode)):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            pert_type, cell_id, pert_idose = extract_features(ft)
            predict = model(drug, gene, mask, pert_type, cell_id, pert_idose)
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

    # ── Save unified output ──────────────────────────────────────────────────
    pred_dir = os.path.join(out_dir, "predictions")

    x_ctrl_test = x_ctrl[test_mask].astype(np.float64)
    test_predictions = (predict_np - x_ctrl_test).astype(np.float32)
    test_ground_truth = (lb_np - x_ctrl_test).astype(np.float32)
    test_ctrl = x_ctrl_test.astype(np.float32)

    np.save(os.path.join(pred_dir, "test_predictions.npy"), test_predictions)
    np.save(os.path.join(pred_dir, "test_ground_truth.npy"), test_ground_truth)
    np.save(os.path.join(pred_dir, "test_ctrl.npy"), test_ctrl)

    train_x_pert_mean = x_pert[train_mask].mean(axis=0).astype(np.float32)
    np.save(os.path.join(pred_dir, "train_x_pert_mean.npy"), train_x_pert_mean)

    sample_ids = pd.DataFrame({
        'drug_id': pert_idx_original[test_mask],
        'cell_id': cell_idx_all[test_mask],
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
    pd.DataFrame(drug_pccs).to_csv(os.path.join(pred_dir, "per_drug_pcc.csv"), index=False)

    resource_log = {
        "model": "deepce",
        "fold": fold_idx,
        "split_col": split_col,
        "ablation_mode": args.ablation_mode,
        "seed": args.seed,
        "best_dev_epoch": int(best_dev_epoch) + 1,
        "best_dev_pearson": float(pearson_list_dev[best_dev_epoch]),
        "best_test_pearson_wrt_dev": float(pearson_list_test[best_dev_epoch]),
        "n_epochs": args.max_epoch,
        "wall_time_s": int(train_time),
        "param_count": param_count,
        "linear_dim": model.linear_dim,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "n_test": int(test_mask.sum()),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "n_cell_lines": n_cells,
        "use_pert_type": use_pert_type,
        "use_cell_id": use_cell_id,
        "use_pert_idose": use_pert_idose,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "logs", "resource_log.json"), 'w') as f:
        json.dump(resource_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} complete!")
    print(f"  Best dev epoch: {best_dev_epoch + 1}")
    print(f"  Dev Pearson: {pearson_list_dev[best_dev_epoch]:.4f}")
    print(f"  Test Pearson (w.r.t. dev): {pearson_list_test[best_dev_epoch]:.4f}")
    print(f"  Predictions: {test_predictions.shape}")
    print(f"  Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  use_pert_type={use_pert_type}, use_cell_id={use_cell_id}, use_pert_idose={use_pert_idose}")
    print(f"  linear_dim={model.linear_dim}, params={param_count:,}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
