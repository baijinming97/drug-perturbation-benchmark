"""
Drug-blind MLP baseline.
  - output_dir from command line
  - saves unified output format (test_predictions.npy etc.)

Model: PureMLP(L3, H2048) — NO drug features.
Input: control expression X1 (978 genes) only.
Output: delta (978 genes), prediction = X1 + delta.
Loss: 0.5 * MSE + 0.5 * (1 - DEG_PCC)

Usage:
    python train_mlp.py --split_col nm_drug_blind_1 --output_dir <results_dir>/fold_0
"""

import sys
import argparse
import numpy as np
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scanpy as sc
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ── Model ──

class PureMLP(nn.Module):
    """X1(978) -> hidden layers -> delta(978). No drug features."""

    def __init__(self, num_layers=3, mlp_hidden=2048, mlp_dropout=0.1):
        super().__init__()
        self.X1_DIM = 978
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.X1_DIM if i == 0 else mlp_hidden
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
            ))
        self.output_fc = nn.Linear(mlp_hidden, self.X1_DIM)

    def forward(self, x1):
        h = x1
        for layer in self.layers:
            h = layer(h)
        return self.output_fc(h)


# ── Dataset ──

class XPertSplitDataset(Dataset):
    """Load XPert h5ad data using its predefined split columns."""

    def __init__(self, adata, split_col, split_value):
        mask = adata.obs[split_col] == split_value
        self.x1 = adata.obsm['X_ctl'][mask].astype(np.float32)
        self.x2 = adata.X[mask].astype(np.float32)
        # Handle sparse matrix
        if hasattr(self.x2, 'toarray'):
            self.x2 = self.x2.toarray()
        if hasattr(self.x1, 'toarray'):
            self.x1 = self.x1.toarray()
        print(f"  {split_value}: {len(self.x1)} samples")

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x1[idx]),
                torch.from_numpy(self.x2[idx]))


# ── Training ──

def compute_deg_corr(x1, x2_pred, x2_true):
    """Batch DEG Pearson correlation (differentiable)."""
    delta_true = x2_true - x1
    delta_pred = x2_pred - x1
    dt_norm = delta_true - delta_true.mean(dim=1, keepdim=True)
    dp_norm = delta_pred - delta_pred.mean(dim=1, keepdim=True)
    num = (dt_norm * dp_norm).sum(dim=1)
    den = torch.sqrt((dt_norm ** 2).sum(dim=1) * (dp_norm ** 2).sum(dim=1))
    return (num / (den + 1e-8)).mean()


def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss, total_mse, total_deg, n = 0, 0, 0, 0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        delta = model(x1)
        x2_pred = x1 + delta

        mse_loss = F.mse_loss(x2_pred, x2)
        deg_corr = compute_deg_corr(x1, x2_pred, x2)
        loss = 0.5 * mse_loss + 0.5 * (1.0 - deg_corr)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_deg += deg_corr.item()
        n += 1

    return total_loss / n, total_mse / n, total_deg / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_mse, total_deg, n = 0, 0, 0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        delta = model(x1)
        x2_pred = x1 + delta
        total_mse += F.mse_loss(x2_pred, x2).item()
        total_deg += compute_deg_corr(x1, x2_pred, x2).item()
        n += 1
    return total_mse / n, total_deg / n


# ── All 21 evaluation metrics ──

def precision_k(y_true, y_pred, k):
    yt_sorted = np.argsort(y_true, axis=1)
    yp_sorted = np.argsort(y_pred, axis=1)
    pp, pn = [], []
    for i in range(len(y_true)):
        pp.append(len(set(yt_sorted[i, -100:]) & set(yp_sorted[i, -k:])) / k)
        pn.append(len(set(yt_sorted[i, :100]) & set(yp_sorted[i, :k])) / k)
    return round(np.mean(pp), 3), round(np.mean(pn), 3)


def avg_pearson(a, b):
    return round(np.mean([pearsonr(a[i], b[i])[0] for i in range(len(a))]), 3)


def avg_spearman(a, b):
    return round(np.mean([spearmanr(a[i], b[i])[0] for i in range(len(a))]), 3)


def ws_per_gene(y_true, y_pred):
    return round(float(np.mean([
        wasserstein_distance(y_true[:, g], y_pred[:, g])
        for g in range(y_pred.shape[1])
    ])), 3)


def compute_all_21(y_true, y_pred, ctl_true):
    m = {}
    m['MSE'] = round(float(mean_squared_error(y_true, y_pred)), 3)
    m['RMSE'] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3)
    m['MAE'] = round(float(mean_absolute_error(y_true, y_pred)), 3)
    m['R²'] = round(float(r2_score(y_true, y_pred)), 3)
    m['PCC'] = avg_pearson(y_true, y_pred)
    m['Spearman'] = avg_spearman(y_true, y_pred)
    print("    PCC/Spearman done")
    m['Wasserstein'] = ws_per_gene(y_true, y_pred)
    print("    Wasserstein done")

    deg_t = y_true - ctl_true
    deg_p = y_pred - ctl_true
    m['R²_deg'] = round(float(r2_score(deg_t, deg_p)), 3)
    m['PCC_deg'] = avg_pearson(deg_t, deg_p)
    m['Spearman_deg'] = avg_spearman(deg_t, deg_p)
    print("    DEG PCC/Spearman done")
    for k in [10, 20, 50, 100]:
        pp, pn = precision_k(deg_t, deg_p, k)
        m[f'Pos P@{k}_deg'] = pp
        m[f'Neg P@{k}_deg'] = pn
    print("    Precision@K done")
    m['Wasserstein_deg'] = ws_per_gene(deg_t, deg_p)
    print("    Wasserstein_deg done")
    return m


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_col', type=str, required=True,
                        help='e.g. nm_drug_blind_1')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='e.g. results/mlp_baseline/default/fold_0')
    parser.add_argument('--h5ad', type=str,
                        default='data/XPert/processed_data/l1000_sdst_78453.h5ad')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mlp_hidden', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (in eval_intervals, same as XPert)')
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    for d in ['checkpoints', 'predictions', 'logs', 'metrics']:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  Shape: {adata.shape}, split_col: {args.split_col}")

    train_ds = XPertSplitDataset(adata, args.split_col, 'train')
    val_ds   = XPertSplitDataset(adata, args.split_col, 'valid')
    test_ds  = XPertSplitDataset(adata, args.split_col, 'test')

    # XPert convention: if no valid set, use test set for early stopping
    if len(val_ds) == 0:
        print(f"  No valid set found — using test set for validation (same as XPert)")
        val_ds = test_ds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = PureMLP(num_layers=args.num_layers, mlp_hidden=args.mlp_hidden)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PureMLP: L{args.num_layers}_H{args.mlp_hidden}, params={n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_deg = -1
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs (eval every {args.eval_interval})...\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, tr_mse, tr_deg = train_epoch(model, optimizer, train_loader, device)

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_mse, val_deg = eval_epoch(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:>4d} | loss={loss:.4f} tr_deg={tr_deg:.4f} | "
                  f"val_mse={val_mse:.4f} val_deg={val_deg:.4f} | {elapsed:.0f}s")

            if val_deg > best_val_deg:
                best_val_deg = val_deg
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_deg_pcc': val_deg,
                }, os.path.join(out_dir, 'checkpoints', 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best={best_epoch}, val_deg={best_val_deg:.4f})")
                    break

    print(f"\nBest epoch: {best_epoch}, val_deg_pcc: {best_val_deg:.4f}")

    # Load best model and evaluate on test set
    ckpt = torch.load(os.path.join(out_dir, 'checkpoints', 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Collect test predictions
    all_x1, all_x2, all_pred = [], [], []
    with torch.no_grad():
        for x1, x2 in test_loader:
            x1 = x1.to(device)
            delta = model(x1)
            x2_pred = x1 + delta
            all_x1.append(x1.cpu().numpy())
            all_x2.append(x2.numpy())
            all_pred.append(x2_pred.cpu().numpy())

    ctl_true = np.concatenate(all_x1)
    y_true = np.concatenate(all_x2)
    y_pred = np.concatenate(all_pred)

    print(f"\nTest set: {len(y_true)} samples")

    # ── Save unified output format ──
    pred_dir = os.path.join(out_dir, 'predictions')
    deg_pred = y_pred - ctl_true
    deg_true = y_true - ctl_true

    np.save(os.path.join(pred_dir, 'test_predictions.npy'), deg_pred.astype(np.float32))
    np.save(os.path.join(pred_dir, 'test_ground_truth.npy'), deg_true.astype(np.float32))
    np.save(os.path.join(pred_dir, 'test_ctrl.npy'), ctl_true.astype(np.float32))

    # train_x_pert_mean
    train_mask = adata.obs[args.split_col] == 'train'
    train_x_pert = adata.X[train_mask]
    if hasattr(train_x_pert, 'toarray'):
        train_x_pert = train_x_pert.toarray()
    np.save(os.path.join(pred_dir, 'train_x_pert_mean.npy'),
            train_x_pert.mean(axis=0).astype(np.float32))

    # test_sample_ids.csv
    import pandas as pd
    test_mask = adata.obs[args.split_col] == 'test'
    test_obs = adata.obs[test_mask]
    pd.DataFrame({
        'drug_id': test_obs['pert_idx'].values,
        'cell_id': test_obs['cell_idx'].values,
    }).to_csv(os.path.join(pred_dir, 'test_sample_ids.csv'), index=False)

    # per_drug_pcc.csv
    from scipy.stats import pearsonr as _pearsonr
    drug_ids = test_obs['pert_idx'].values.astype(int)
    drug_pccs = []
    for drug in np.unique(drug_ids):
        mask = drug_ids == drug
        pccs = [_pearsonr(deg_true[i], deg_pred[i])[0]
                for i in np.where(mask)[0]
                if np.std(deg_true[i]) > 0 and np.std(deg_pred[i]) > 0]
        if pccs:
            drug_pccs.append({'drug_id': int(drug), 'pcc': float(np.mean(pccs)),
                              'n_samples': int(mask.sum())})
    pd.DataFrame(drug_pccs).to_csv(os.path.join(pred_dir, 'per_drug_pcc.csv'), index=False)

    print(f"Unified output saved to {pred_dir}")

    # ── Resource log ──
    wall_time = time.time() - t0
    resource_log = {
        'model': 'mlp_baseline',
        'split_col': args.split_col,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'n_epochs_run': epoch,
        'wall_time_s': int(wall_time),
        'param_count': n_params,
        'n_test': len(y_true),
        'hostname': os.uname().nodename,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(out_dir, 'logs', 'resource_log.json'), 'w') as f:
        json.dump(resource_log, f, indent=2)

    # Training curve
    # (already printed per-epoch, save a summary)
    print(f"\n{'='*50}")
    print(f"  MLP Baseline — Fold done")
    print(f"  Best epoch: {best_epoch}, val_deg_pcc: {best_val_deg:.4f}")
    print(f"  Test samples: {len(y_true)}")
    print(f"  Wall time: {wall_time:.0f}s")
    print(f"  Output: {out_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
