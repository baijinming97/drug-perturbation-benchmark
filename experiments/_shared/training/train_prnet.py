"""
PRnet training on the unified dataset (l1000_sdst) with cold-drug splits.

End-to-end: load h5ad → compute FCFP4 → train → predict → save unified output.

Usage:
    python train_prnet.py --fold 0
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
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import normal
from scipy.stats import pearsonr

from pathlib import Path
PRNET_ROOT = str(Path(__file__).resolve().parents[3] / "models" / "PRnet" / "models")
sys.path.insert(0, PRNET_ROOT)

from PRnet import PGM
from bench_dataset import BenchPRnetDataset


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[3]))
H5AD_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/l1000_sdst_78453.h5ad")
SMI_DEFAULT = os.path.join(REPO_ROOT, "data/XPert/processed_data/all_drugs_idx2smi_8981.npy")
RESULTS_BASE = os.path.join(REPO_ROOT, "results")


def parse_args():
    p = argparse.ArgumentParser(description="PRnet cold-drug training")
    p.add_argument("--fold", type=int, required=True, help="Fold index 0-4")
    p.add_argument("--split_col", type=str, default=None,
                   help="Override split column name (default: nm_drug_blind_<fold+1>; pass e.g. nm_scaffold_1 to use scaffold-disjoint splits)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--h5ad_path", type=str, default=H5AD_DEFAULT)
    p.add_argument("--smi_path", type=str, default=SMI_DEFAULT)
    p.add_argument("--dev", type=str, default="cuda:0")
    # PRnet paper defaults; seed = PRnet original default
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n_epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-8)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--ablation_mode", type=str, default="none",
                   choices=["none", "zero", "shuffle"],
                   help="Drug ablation: none=original, zero=zero drug features, shuffle=permute drug identity")
    p.add_argument("--dose_col", type=str, default=None,
                   help="Per-sample dose column in h5ad obs (default: constant dose=10)")
    return p.parse_args()


def compute_fcfp4_dict(idx2smi, dose=10.0, nbits=1024):
    """Precompute FCFP4 for all drugs. If dose is given, scale by log10(dose+1)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    scale = np.log10(dose + 1) if dose is not None else 1.0
    fcfp4_dict = {}
    failed = []
    for idx, smi in idx2smi.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(idx)
            fcfp4_dict[idx] = np.zeros(nbits, dtype=np.float32)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=nbits)
        arr = np.array(list(fp.ToBitString()), dtype=np.float32)
        fcfp4_dict[idx] = arr * scale
    if failed:
        print(f"  WARNING: {len(failed)} drugs failed SMILES parsing, zeroed out")
    return fcfp4_dict


def main():
    args = parse_args()
    fold_idx = args.fold
    split_col = args.split_col or f"nm_drug_blind_{fold_idx + 1}"

    out_dir = args.output_dir or os.path.join(
        RESULTS_BASE, f"main_benchmark/prnet/fold_{fold_idx}")

    print(f"{'='*60}")
    print(f"PRnet | Fold {fold_idx} | Split: {split_col}")
    print(f"Output: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else "cpu")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    adata = sc.read_h5ad(args.h5ad_path)
    idx2smi = np.load(args.smi_path, allow_pickle=True).item()

    # PRnet original preprocessing (train_lincs.py:62-63)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Preprocess control expression separately (obsm not affected by sc.pp)
    import anndata
    ctrl_raw = adata.obsm['X_ctl']
    if hasattr(ctrl_raw, 'toarray'):
        ctrl_raw = ctrl_raw.toarray()
    adata_ctrl = anndata.AnnData(X=ctrl_raw.copy())
    sc.pp.normalize_total(adata_ctrl)
    sc.pp.log1p(adata_ctrl)

    x_pert = adata.X
    if hasattr(x_pert, 'toarray'):
        x_pert = x_pert.toarray()
    x_ctrl = adata_ctrl.X
    if hasattr(x_ctrl, 'toarray'):
        x_ctrl = x_ctrl.toarray()
    print(f"  Data after preprocessing: x_pert range [{x_pert.min():.2f}, {x_pert.max():.2f}], "
          f"x_ctrl range [{x_ctrl.min():.2f}, {x_ctrl.max():.2f}]")

    pert_idx_all = adata.obs['pert_idx'].values.astype(int)
    cell_idx_all = adata.obs['cell_idx'].values.astype(int)
    splits = adata.obs[split_col].values.astype(str)

    # ── Step 2: Compute FCFP4 features ────────────────────────────────────────
    print("\n[2/7] Computing FCFP4 drug features ...")
    if args.dose_col:
        fcfp4_dict = compute_fcfp4_dict(idx2smi, dose=None)
        doses = adata.obs[args.dose_col].values.astype(float)
        fcfp4_all = np.stack([fcfp4_dict[idx] * np.log10(d + 1)
                              for idx, d in zip(pert_idx_all, doses)])
        print(f"  Per-sample dose from '{args.dose_col}': min={doses.min():.4f}, max={doses.max():.4f}")
    else:
        fcfp4_dict = compute_fcfp4_dict(idx2smi, dose=10.0)
        fcfp4_all = np.stack([fcfp4_dict[idx] for idx in pert_idx_all])
    print(f"  FCFP4 shape: {fcfp4_all.shape}")

    # ── Drug ablation (data-layer) ──────────────────────────────────────────
    # Both zero and shuffle are applied at data level (before model).
    # Zero: sets FCFP4 input to 0; CombAdaptor bias term is a fixed constant
    # with no drug-specific information, so this effectively removes drug signal.
    # Shuffle: global drug→sample permutation (seed=131419), matching the seed convention.
    if args.ablation_mode == 'zero':
        print("  [ABLATION] Zeroing all drug features")
        fcfp4_all[:] = 0.0
    elif args.ablation_mode == 'shuffle':
        print("  [ABLATION] Shuffling drug identity (permutation seed=131419)")
        unique_drugs = np.unique(pert_idx_all)
        rng = np.random.default_rng(seed=131419)
        perm = rng.permutation(len(unique_drugs))
        drug_map = dict(zip(unique_drugs, unique_drugs[perm]))
        shuffled_idx = np.array([drug_map[d] for d in pert_idx_all])
        fcfp4_all = np.stack([fcfp4_dict[idx] for idx in shuffled_idx])

    # ── Step 3: Split + DataLoaders ───────────────────────────────────────────
    print("\n[3/7] Building DataLoaders ...")
    train_mask = splits == 'train'
    valid_mask = splits == 'valid'
    test_mask = splits == 'test'
    print(f"  train={train_mask.sum()}, valid={valid_mask.sum()}, test={test_mask.sum()}")

    train_ds = BenchPRnetDataset(x_ctrl[train_mask], x_pert[train_mask], fcfp4_all[train_mask])
    valid_ds = BenchPRnetDataset(x_ctrl[valid_mask], x_pert[valid_mask], fcfp4_all[valid_mask])
    test_ds = BenchPRnetDataset(x_ctrl[test_mask], x_pert[test_mask], fcfp4_all[test_mask])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # ── Step 4: Create output dirs ────────────────────────────────────────────
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    for d in [os.path.join(out_dir, "predictions"),
              os.path.join(out_dir, "logs"),
              os.path.join(out_dir, "metrics"),
              ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Step 5: Init model ────────────────────────────────────────────────────
    print("\n[4/7] Initializing PGM model ...")
    # PRnet paper defaults for L1000
    model = PGM(
        x_dim=978, c_dim=64, n_dim=10,
        hidden_layer_sizes=[128], z_dimension=64,
        adaptor_layer_sizes=[128], comb_adapt_dim=1024,  # 1 drug × 1024-bit FCFP4
        dr_rate=0.05
    ).to(dev)

    # PRnet original weight init (PRnetTrainer.py:107, 491-504)
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
    model.apply(weight_init)

    # When drug input is zeroed, the adapter's final Linear bias is the only
    # surviving signal (W @ 0 + b = b). Freeze it at zero so the model truly
    # receives a zero drug embedding, not a learnable constant.
    if args.ablation_mode == 'zero':
        model.CombAdaptor.comb_encoder.bias.data.zero_()
        model.CombAdaptor.comb_encoder.bias.requires_grad = False
        print("  [ABLATION] Froze CombAdaptor.comb_encoder.bias at zero")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Optimizer + scheduler (match PRnetTrainer exactly)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=1e-8)
    criterion = nn.GaussianNLLLoss()

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    print(f"\n[5/7] Training (max {args.n_epochs} epochs, early stop patience={args.early_stop_patience}) ...")
    start_time = time.time()

    best_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    epoch_hist = defaultdict(list)

    for epoch in range(args.n_epochs):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for ctrl, target, drug_enc in train_loader:
            ctrl = ctrl.to(dev)
            target = target.to(dev)
            drug_enc = drug_enc.to(dev)
            noise = torch.randn(ctrl.size(0), 10, device=dev)

            optimizer.zero_grad()
            output = model(ctrl, drug_enc, noise)
            dim = output.size(1) // 2
            gene_means = output[:, :dim]
            gene_vars = F.softplus(output[:, dim:])
            loss = criterion(gene_means, target, gene_vars)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        epoch_hist['train_loss'].append(avg_train_loss)

        # ── Validate ──
        model.eval()
        val_mse_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for ctrl, target, drug_enc in valid_loader:
                ctrl = ctrl.to(dev)
                target = target.to(dev)
                drug_enc = drug_enc.to(dev)
                noise = torch.randn(ctrl.size(0), 10, device=dev)

                output = model(ctrl, drug_enc, noise)
                dim = output.size(1) // 2
                gene_means = output[:, :dim]
                gene_vars = F.softplus(output[:, dim:])

                # MSE for early stopping (match PRnetTrainer)
                dist = normal.Normal(
                    torch.clamp(gene_means, min=1e-3, max=1e3),
                    torch.clamp(gene_vars.sqrt(), min=1e-3, max=1e3))
                nb_sample = dist.sample()
                mse = torch.mean((nb_sample - target) ** 2).item()
                val_mse_sum += mse
                val_batches += 1

        avg_val_mse = val_mse_sum / max(val_batches, 1)
        epoch_hist['val_mse'].append(avg_val_mse)
        scheduler.step(avg_val_mse)

        if epoch % 10 == 0 or epoch < 5:
            print(f"  [Epoch {epoch:3d}] train_loss={avg_train_loss:.4f}, val_mse={avg_val_mse:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # ── Early stopping ──
        if avg_val_mse < best_mse:
            best_mse = avg_val_mse
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(ckpt_dir, "best_model.pt"))
        elif patience_counter <= args.early_stop_patience:
            patience_counter += 1
        else:
            print(f"  Early stopping at epoch {epoch} (patience={args.early_stop_patience})")
            break

    train_time = time.time() - start_time
    print(f"  Best epoch: {best_epoch}, Best val MSE: {best_mse:.4f}")
    print(f"  Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")

    # Save training curve
    curve_df = pd.DataFrame(epoch_hist)
    curve_df['epoch'] = range(len(curve_df))
    curve_df.to_csv(os.path.join(out_dir, "logs", "training_curve.csv"), index=False)

    # ── Step 7: Predict on test set ───────────────────────────────────────────
    print("\n[6/7] Predicting on test set ...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=dev))
    model.eval()

    # Fix seed for reproducible sampling
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    all_ctrl = []
    all_target = []
    all_pred = []

    with torch.no_grad():
        for ctrl, target, drug_enc in test_loader:
            ctrl = ctrl.to(dev)
            target = target.to(dev)
            drug_enc = drug_enc.to(dev)
            noise = torch.randn(ctrl.size(0), 10, device=dev)

            output = model(ctrl, drug_enc, noise)
            dim = output.size(1) // 2
            gene_means = output[:, :dim]
            gene_vars = F.softplus(output[:, dim:])

            # Sample from Normal (match original PRnet test behavior)
            dist = normal.Normal(
                torch.clamp(gene_means, min=1e-3, max=1e3),
                torch.clamp(gene_vars.sqrt(), min=1e-3, max=1e3))
            pred = dist.sample()

            all_ctrl.append(ctrl.cpu().numpy())
            all_target.append(target.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    x_ctrl_test = np.concatenate(all_ctrl, axis=0)
    x_pert_test = np.concatenate(all_target, axis=0)
    x_pred_test = np.concatenate(all_pred, axis=0)

    print(f"  Shapes: ctrl={x_ctrl_test.shape}, pert={x_pert_test.shape}, pred={x_pred_test.shape}")

    # ── Step 8: Save unified output ───────────────────────────────────────────
    print("\n[7/7] Saving unified output ...")
    pred_dir = os.path.join(out_dir, "predictions")

    test_predictions = (x_pred_test - x_ctrl_test).astype(np.float32)
    test_ground_truth = (x_pert_test - x_ctrl_test).astype(np.float32)
    test_ctrl = x_ctrl_test.astype(np.float32)

    np.save(os.path.join(pred_dir, "test_predictions.npy"), test_predictions)
    np.save(os.path.join(pred_dir, "test_ground_truth.npy"), test_ground_truth)
    np.save(os.path.join(pred_dir, "test_ctrl.npy"), test_ctrl)

    # train_x_pert_mean
    train_x_pert_mean = x_pert[train_mask].mean(axis=0).astype(np.float32)
    np.save(os.path.join(pred_dir, "train_x_pert_mean.npy"), train_x_pert_mean)

    # test_sample_ids.csv (aligned with test_loader shuffle=False)
    sample_ids = pd.DataFrame({
        'drug_id': pert_idx_all[test_mask],
        'cell_id': cell_idx_all[test_mask]
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
            gt, pr = test_ground_truth[i], test_predictions[i]
            if np.std(gt) > 0 and np.std(pr) > 0:
                r, _ = pearsonr(gt, pr)
                pccs.append(r)
        if pccs:
            drug_pccs.append({'drug_id': int(drug), 'pcc': float(np.mean(pccs)),
                              'n_samples': int(mask.sum())})
    pd.DataFrame(drug_pccs).to_csv(os.path.join(pred_dir, "per_drug_pcc.csv"), index=False)

    # Resource log
    resource_log = {
        "model": "prnet",
        "fold": fold_idx,
        "split_col": split_col,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "n_epochs_run": len(epoch_hist['train_loss']),
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

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} complete!")
    print(f"  Predictions: {test_predictions.shape}")
    print(f"  Best epoch: {best_epoch}, Val MSE: {best_mse:.4f}")
    print(f"  Wall time: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
