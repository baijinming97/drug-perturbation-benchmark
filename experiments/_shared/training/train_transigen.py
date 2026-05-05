"""
TranSiGen training on the unified dataset (l1000_sdst) with cold-drug splits.

End-to-end: load h5ad → train → predict → save unified output.

Usage:
    python train_transigen.py --fold 0
    python train_transigen.py --fold 0 --n_epochs 5   # quick test
"""

# Suppress RDKit's verbose C++ logger BEFORE any rdkit-using import
# (otherwise GetExplicitValence deprecation spams stderr ~5M lines/run).
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
RDLogger.DisableLog('rdApp.*')

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import pearsonr

from pathlib import Path
TRANSIGEN_ROOT = str(Path(__file__).resolve().parents[3] / "models" / "TranSiGen" / "src")
sys.path.insert(0, TRANSIGEN_ROOT)

from model import TranSiGen
from utils import setup_seed, seed_worker
from bench_dataset import BenchTranSiGenDataset


# ── Paths (absolute) ──────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[3]))
H5AD_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/l1000_sdst_78453.h5ad")
KPGT_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/all_drugs_idx2KPGT.npy")
VAE_X1_DEFAULT = str(Path(__file__).resolve().parents[3] / "reference" / "TranSiGen" / "results" / "trained_model_shRNA_vae_x1" / "best_model.pt")
VAE_X2_DEFAULT = str(Path(__file__).resolve().parents[3] / "reference" / "TranSiGen" / "results" / "trained_model_shRNA_vae_x2" / "best_model.pt")
RESULTS_BASE = os.path.join(REPO_ROOT, "results")


def parse_args():
    p = argparse.ArgumentParser(description="TranSiGen cold-drug training")
    p.add_argument("--fold", type=int, required=True, help="Fold index 0-4")
    p.add_argument("--split_col", type=str, default=None,
                   help="Override split column name (default: nm_drug_blind_<fold+1>; pass e.g. nm_scaffold_1 to use scaffold-disjoint splits)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory (default: results/main_benchmark/transigen/fold_{fold})")
    # Data paths
    p.add_argument("--h5ad_path", type=str, default=H5AD_DEFAULT)
    p.add_argument("--kpgt_path", type=str, default=KPGT_DEFAULT)
    p.add_argument("--vae_x1_path", type=str, default=VAE_X1_DEFAULT)
    p.add_argument("--vae_x2_path", type=str, default=VAE_X2_DEFAULT)
    # Device
    p.add_argument("--dev", type=str, default="cuda:0")
    # Hyperparameters (TranSiGen paper defaults)
    p.add_argument("--seed", type=int, default=895834)
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--n_latent", type=int, default=100)
    p.add_argument("--ablation_mode", type=str, default="none",
                   choices=["none", "zero", "shuffle"],
                   help="Drug ablation: none=original, zero=zero drug embedding, "
                        "shuffle=permute drug identity within batch")
    return p.parse_args()


def main():
    args = parse_args()
    fold_idx = args.fold
    split_col = args.split_col or f"nm_drug_blind_{fold_idx + 1}"

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    elif args.ablation_mode == 'none':
        out_dir = os.path.join(RESULTS_BASE,
            f"main_benchmark/transigen/fold_{fold_idx}")
    else:
        out_dir = os.path.join(RESULTS_BASE,
            f"main_ablation/transigen/{args.ablation_mode}/fold_{fold_idx}")

    print(f"{'='*60}")
    print(f"TranSiGen | Fold {fold_idx} | Split: {split_col}")
    print(f"Output: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    setup_seed(args.seed)
    torch.set_num_threads(1)  # match original TranSiGen — deterministic BLAS
    dev = torch.device(args.dev if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    adata = sc.read_h5ad(args.h5ad_path)
    kpgt_dict = np.load(args.kpgt_path, allow_pickle=True).item()

    # Extract arrays
    x_pert = adata.X  # (78453, 978)
    if hasattr(x_pert, 'toarray'):
        x_pert = x_pert.toarray()
    x_ctrl = adata.obsm['X_ctl']  # (78453, 978)
    if hasattr(x_ctrl, 'toarray'):
        x_ctrl = x_ctrl.toarray()

    pert_idx_all = adata.obs['pert_idx'].values.astype(int)
    pert_idx_original = pert_idx_all.copy()  # preserve for sample_ids output
    cell_idx_all = adata.obs['cell_idx'].values.astype(int)
    splits = adata.obs[split_col].values.astype(str)

    # ── Drug ablation: shuffle at data layer (global drug permutation) ────────
    if args.ablation_mode == 'shuffle':
        print("  [ABLATION] Shuffling drug identity (permutation seed=131419)")
        unique_drugs = np.unique(pert_idx_all)
        rng = np.random.default_rng(seed=131419)
        perm = rng.permutation(len(unique_drugs))
        drug_map = dict(zip(unique_drugs, unique_drugs[perm]))
        pert_idx_all = np.array([drug_map[d] for d in pert_idx_all])

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    train_mask = splits == 'train'
    valid_mask = splits == 'valid'
    test_mask = splits == 'test'

    print(f"  train={train_mask.sum()}, valid={valid_mask.sum()}, test={test_mask.sum()}")

    # ── Step 3: Build DataLoaders ─────────────────────────────────────────────
    print("\n[2/7] Building DataLoaders ...")
    train_ds = BenchTranSiGenDataset(x_ctrl[train_mask], x_pert[train_mask],
                                  pert_idx_all[train_mask], cell_idx_all[train_mask], kpgt_dict)
    valid_ds = BenchTranSiGenDataset(x_ctrl[valid_mask], x_pert[valid_mask],
                                  pert_idx_all[valid_mask], cell_idx_all[valid_mask], kpgt_dict)
    test_ds = BenchTranSiGenDataset(x_ctrl[test_mask], x_pert[test_mask],
                                 pert_idx_all[test_mask], cell_idx_all[test_mask], kpgt_dict)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=2, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=2, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=2, worker_init_fn=seed_worker)

    # ── Step 4: Create output dirs ────────────────────────────────────────────
    ckpt_dir = os.path.join(out_dir, "checkpoints") + "/"
    for d in [os.path.join(out_dir, "predictions"),
              os.path.join(out_dir, "logs"),
              os.path.join(out_dir, "metrics"),
              ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Step 5: Init model + load pretrained VAE ──────────────────────────────
    print("\n[3/7] Initializing model ...")
    model = TranSiGen(
        n_genes=978, n_latent=args.n_latent,
        n_en_hidden=[1200], n_de_hidden=[800],
        features_dim=2304, features_embed_dim=[400],
        init_w=True, beta=args.beta, device=dev, dropout=args.dropout,
        path_model=ckpt_dir, random_seed=args.seed
    )

    # Load pretrained shRNA VAE encoder/decoder weights
    model_dict = model.state_dict()
    for vae_path in [args.vae_x1_path, args.vae_x2_path]:
        print(f"  Loading VAE weights: {os.path.basename(os.path.dirname(vae_path))}")
        vae_model = torch.load(vae_path, map_location='cpu')
        vae_dict = vae_model.state_dict()
        matched = 0
        for k in model_dict.keys():
            if k in vae_dict:
                model_dict[k] = vae_dict[k]
                matched += 1
        print(f"    Matched {matched} parameters")
        del vae_model
    model.load_state_dict(model_dict)
    model.to(dev)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    print(f"\n[4/7] Training ({args.n_epochs} epochs) ...")
    start_time = time.time()

    epoch_hist, best_epoch = model.train_model(
        train_loader=train_loader,
        test_loader=valid_loader,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_model=True,
        ablation_mode=args.ablation_mode
    )

    train_time = time.time() - start_time
    print(f"  Best epoch: {best_epoch}, Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")

    # Save training curve
    curve_df = pd.DataFrame.from_dict(epoch_hist)
    curve_df['epoch'] = range(len(curve_df))
    curve_df.to_csv(os.path.join(out_dir, "logs", "training_curve.csv"), index=False)

    # ── Step 7: Load best model + predict on test set ─────────────────────────
    print("\n[5/7] Predicting on test set ...")
    best_path = ckpt_dir + "best_model.pt"
    model = torch.load(best_path, map_location='cpu')
    model.dev = dev
    model.to(dev)

    # shuffle=False in test_loader ensures predict_profile output aligns with test_ds ordering
    x1_arr, x2_arr, _, _, x2_pred_arr, _, mol_id_arr, cid_arr, _ = \
        model.predict_profile(test_loader, ablation_mode=args.ablation_mode)

    print(f"  x1: {x1_arr.shape}, x2: {x2_arr.shape}, x2_pred: {x2_pred_arr.shape}")

    # ── Step 8: Convert to unified output format ──────────────────────────────
    print("\n[6/7] Saving unified output ...")
    pred_dir = os.path.join(out_dir, "predictions")

    # x_deg = x_pert - x_ctrl
    test_predictions = (x2_pred_arr - x1_arr).astype(np.float32)
    test_ground_truth = (x2_arr - x1_arr).astype(np.float32)
    test_ctrl = x1_arr.astype(np.float32)

    np.save(os.path.join(pred_dir, "test_predictions.npy"), test_predictions)
    np.save(os.path.join(pred_dir, "test_ground_truth.npy"), test_ground_truth)
    np.save(os.path.join(pred_dir, "test_ctrl.npy"), test_ctrl)

    # train_x_pert_mean
    train_x_pert_mean = x_pert[train_mask].mean(axis=0).astype(np.float32)
    np.save(os.path.join(pred_dir, "train_x_pert_mean.npy"), train_x_pert_mean)

    # test_sample_ids.csv — use ORIGINAL drug IDs (not shuffled)
    sample_ids = pd.DataFrame({
        'drug_id': pert_idx_original[test_mask],
        'cell_id': cid_arr.astype(int) if np.issubdtype(cid_arr.dtype, np.number)
                   else [int(c) for c in cid_arr]
    })
    sample_ids.to_csv(os.path.join(pred_dir, "test_sample_ids.csv"), index=False)

    # per_drug_pcc.csv
    drug_ids = sample_ids['drug_id'].values
    unique_drugs = np.unique(drug_ids)
    drug_pccs = []
    for drug in unique_drugs:
        mask = drug_ids == drug
        pccs = []
        for i in np.where(mask)[0]:
            gt = test_ground_truth[i]
            pr = test_predictions[i]
            if np.std(gt) > 0 and np.std(pr) > 0:
                r, _ = pearsonr(gt, pr)
                pccs.append(r)
        if pccs:
            drug_pccs.append({'drug_id': int(drug), 'pcc': float(np.mean(pccs)),
                              'n_samples': int(mask.sum())})
    pd.DataFrame(drug_pccs).to_csv(os.path.join(pred_dir, "per_drug_pcc.csv"), index=False)

    print(f"  Saved: test_predictions ({test_predictions.shape}), "
          f"test_ground_truth, test_ctrl, test_sample_ids ({len(sample_ids)}), "
          f"per_drug_pcc ({len(drug_pccs)} drugs), train_x_pert_mean")

    # ── Step 9: Resource log ──────────────────────────────────────────────────
    print("\n[7/7] Saving resource log ...")
    resource_log = {
        "model": "transigen",
        "fold": fold_idx,
        "split_col": split_col,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "n_epochs": args.n_epochs,
        "wall_time_s": int(train_time),
        "param_count": param_count,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "n_test": int(test_mask.sum()),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    with open(os.path.join(out_dir, "logs", "resource_log.json"), 'w') as f:
        json.dump(resource_log, f, indent=2)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} complete!")
    print(f"  Predictions: {test_predictions.shape}")
    print(f"  Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
