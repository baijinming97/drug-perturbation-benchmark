"""
Unified Evaluation Harness
======================================
Library: computes 26 standard error / correlation / precision / distribution
metrics on unified-format predictions (test_predictions.npy +
test_ground_truth.npy + optional test_ctrl.npy + train_x_pert_mean.npy).
Outputs all_metrics.json per fold.

All metric implementations aligned with XPert native code
(evaluation_metrics/get_evaluation_metrics.py) as ground truth.

Library entry point: ``evaluate_predictions(...)``. Wrapped by
``evaluate_one_fold.py`` for command-line use.

Most metrics are pure NumPy/SciPy. MMD (#25) requires CUDA.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


# =============================================================================
# Metric implementations
# =============================================================================

# --- Shared helpers ---

def _center(x):
    """Center each row (sample) to zero mean."""
    return x - x.mean(axis=1, keepdims=True)


def _pcc_vectorized(y_true, y_pred):
    """Per-sample Pearson correlation. (N,G) -> (N,)"""
    yt, yp = _center(y_true), _center(y_pred)
    num = (yt * yp).sum(axis=1)
    den = np.sqrt((yt ** 2).sum(axis=1) * (yp ** 2).sum(axis=1))
    return num / (den + 1e-8)


# --- A. Error metrics (#1-3) ---

def per_sample_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean(axis=1)

def per_sample_rmse(y_true, y_pred):
    return np.sqrt(per_sample_mse(y_true, y_pred))

def per_sample_mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean(axis=1)


# --- B. Correlation on x_deg (#4-7) ---

def per_sample_pcc(y_true, y_pred):
    """#4: PCC(x_deg) — PRIMARY metric."""
    return _pcc_vectorized(y_true, y_pred)

def per_sample_spearman(y_true, y_pred):
    """#5: Spearman(x_deg)."""
    n = y_true.shape[0]
    corrs = np.zeros(n)
    for i in range(n):
        corrs[i], _ = stats.spearmanr(y_true[i], y_pred[i])
    return corrs

def pcc_top_k_deg(y_true, y_pred, k=100):
    """#6: PCC on top-K DEGs per sample."""
    n = y_true.shape[0]
    pccs = np.zeros(n)
    for i in range(n):
        top_idx = np.argsort(np.abs(y_true[i]))[-k:]
        yt, yp = y_true[i, top_idx], y_pred[i, top_idx]
        if yt.std() < 1e-8 or yp.std() < 1e-8:
            pccs[i] = 0.0
        else:
            pccs[i] = np.corrcoef(yt, yp)[0, 1]
    return pccs

def r2_per_gene(y_true, y_pred):
    """#7/#11/#12: R2 computed per-gene (across samples), then averaged.

    Matches sklearn.metrics.r2_score(y_true, y_pred) on 2D arrays
    and XPert's native evaluation (evaluation_metrics/get_evaluation_metrics.py).

    For each gene j:
        R2_j = 1 - sum_i (y_true[i,j] - y_pred[i,j])^2
                    / sum_i (y_true[i,j] - mean_i(y_true[:,j]))^2
    Returns: mean over genes, and per-gene array.
    """
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)           # (G,)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)  # (G,)
    r2_genes = 1 - ss_res / (ss_tot + 1e-8)
    return float(np.mean(r2_genes)), float(np.std(r2_genes))

def r2_top_k_deg(y_true, y_pred, k=100):
    """#7: R2 on top-K DEGs — per-gene across samples, restricted to
    genes that are top-K DEGs on average (by mean |x_deg| across samples)."""
    mean_abs_deg = np.abs(y_true).mean(axis=0)  # (G,)
    top_gene_idx = np.argsort(mean_abs_deg)[-k:]
    yt_sub = y_true[:, top_gene_idx]
    yp_sub = y_pred[:, top_gene_idx]
    r2_mean, r2_std = r2_per_gene(yt_sub, yp_sub)
    return r2_mean, r2_std


# --- C. Correlation on x_pert (#8-10) ---
# #8, #9: reuse per_sample_pcc / per_sample_spearman on x_pert arrays
# #10: Pearson(delta_pert) — needs train_x_pert_mean

def pearson_delta_pert(y_pert_true, y_pert_pred, train_mean):
    """#10: cor(pred_pert - dataset_mean, true_pert - dataset_mean)."""
    return _pcc_vectorized(y_pert_true - train_mean, y_pert_pred - train_mean)


# --- D. Goodness-of-Fit (#11-14) ---

def per_sample_wmse(y_true, y_pred, weights):
    """#13: WMSE — weighted MSE per sample."""
    n = y_true.shape[0]
    wmses = np.zeros(n)
    for i in range(n):
        w = weights[i]
        wmses[i] = (w * (y_true[i] - y_pred[i]) ** 2).sum() / (w.sum() + 1e-8)
    return wmses

def per_sample_wr2(y_true, y_pred, weights):
    """#14: WR2(delta_ctrl) — weighted R2 per sample."""
    n = y_true.shape[0]
    wr2s = np.zeros(n)
    for i in range(n):
        w = weights[i]
        mu = (w * y_true[i]).sum() / (w.sum() + 1e-8)
        ss_res = (w * (y_true[i] - y_pred[i]) ** 2).sum()
        ss_tot = (w * (y_true[i] - mu) ** 2).sum()
        wr2s[i] = 1 - ss_res / (ss_tot + 1e-8)
    return wr2s


# --- E. Precision@K (#15-22) ---

def precision_at_k(y_true, y_pred, k=20, direction='pos'):
    """#15-22: Precision@K matching XPert convention.

    Ground truth set is ALWAYS top/bottom 100 genes (fixed).
    Predicted set is top/bottom K genes.
    Precision = |GT_100 ∩ Pred_K| / K.
    """
    GT_SIZE = 100
    n = y_true.shape[0]
    precisions = np.zeros(n)
    for i in range(n):
        if direction == 'pos':
            true_set = set(np.argsort(y_true[i])[-GT_SIZE:])
            pred_set = set(np.argsort(y_pred[i])[-k:])
        else:
            true_set = set(np.argsort(y_true[i])[:GT_SIZE])
            pred_set = set(np.argsort(y_pred[i])[:k])
        precisions[i] = len(true_set & pred_set) / k
    return precisions


# --- F. Distribution metrics (#23-26) ---

def per_gene_wasserstein(y_true, y_pred):
    """#23/#24: Wasserstein-1 distance computed per-gene (across samples), then averaged.

    Matches XPert native: for each gene, compute wasserstein_distance
    between true and predicted value distributions across all test samples.
    """
    n_genes = y_true.shape[1]
    dists = np.zeros(n_genes)
    for j in range(n_genes):
        dists[j] = wasserstein_distance(y_true[:, j], y_pred[:, j])
    return float(np.mean(dists)), float(np.std(dists))

def per_gene_mmd_rbf(y_true, y_pred, gamma=1.0,
                     device='cuda', chunk_genes=16, dtype=None):
    """MMD with RBF kernel, computed per-gene over all N samples, then averaged.

    Biased estimator (full N×N kernel matrix per gene), gamma=1.0 — matches
    the convention of XPert's `compute_mmd`. Runs on GPU; chunks genes to
    control peak memory (peak per chunk = chunk_genes × N² × 8 bytes at fp64).

    Returns (mean, std) over the n_genes per-gene biased MMD values.
    Kernel: K(x, y) = exp(-gamma * ||x − y||²)
    """
    import torch
    if dtype is None:
        dtype = torch.float64
    N, G = y_true.shape
    assert y_pred.shape == (N, G)

    free_mem = torch.cuda.mem_get_info()[0]
    peak_per_gene = N * N * 8  # fp64 K matrix
    chunk_genes = max(1, min(chunk_genes,
                             int(free_mem * 0.5) // max(peak_per_gene, 1)))

    with torch.no_grad():
        pred_t = torch.from_numpy(np.ascontiguousarray(y_pred)).to(device=device, dtype=dtype)
        true_t = torch.from_numpy(np.ascontiguousarray(y_true)).to(device=device, dtype=dtype)
        out = torch.empty(G, device=device, dtype=dtype)
        for g0 in range(0, G, chunk_genes):
            g1 = min(g0 + chunk_genes, G)
            x = pred_t[:, g0:g1].T.unsqueeze(2).contiguous()  # (cg, N, 1)
            y = true_t[:, g0:g1].T.unsqueeze(2).contiguous()
            d = (x - x.transpose(1, 2)).pow_(2)
            Kxx_mean = torch.exp_(d.mul_(-gamma)).mean(dim=(1, 2))
            del d
            d = (y - y.transpose(1, 2)).pow_(2)
            Kyy_mean = torch.exp_(d.mul_(-gamma)).mean(dim=(1, 2))
            del d
            d = (x - y.transpose(1, 2)).pow_(2)
            Kxy_mean = torch.exp_(d.mul_(-gamma)).mean(dim=(1, 2))
            del d, x, y
            out[g0:g1] = Kxx_mean + Kyy_mean - 2 * Kxy_mean
        per_gene = out.cpu().numpy()
    return float(np.mean(per_gene)), float(np.std(per_gene))


# =============================================================================
# Main evaluation
# =============================================================================

def _stat(arr):
    """Return mean and std as floats."""
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def evaluate_predictions(y_deg_true, y_deg_pred, y_ctrl=None,
                         sample_ids=None, train_x_pert_mean=None,
                         skip_mmd=False, mmd_only=False):
    """
    Compute all 26 metrics (see the top-of-file docstring for categories).

    Args:
        y_deg_true:  (N, 978) ground truth x_deg
        y_deg_pred:  (N, 978) predicted x_deg
        y_ctrl:      (N, 978) basal expression [optional, for x_pert metrics]
        sample_ids:  dict with 'drug_id' array [optional, for per-drug]
        train_x_pert_mean: (978,) mean x_pert from training [optional, for #10]

    Returns:
        dict with all metrics
    """
    metrics = {}

    if not mmd_only:
        # =====================================================================
        # A. Error metrics (#1-3) — same for x_pert and x_deg
        # =====================================================================
        mse_ps = per_sample_mse(y_deg_true, y_deg_pred)
        rmse_ps = per_sample_rmse(y_deg_true, y_deg_pred)
        mae_ps = per_sample_mae(y_deg_true, y_deg_pred)

        metrics['mse_mean'], metrics['mse_std'] = _stat(mse_ps)
        metrics['rmse_mean'], metrics['rmse_std'] = _stat(rmse_ps)
        metrics['mae_mean'], metrics['mae_std'] = _stat(mae_ps)
        # Miller L2 = sqrt(978) * RMSE, per sample
        l2_ps = rmse_ps * np.sqrt(978)
        metrics['l2_mean'], metrics['l2_std'] = _stat(l2_ps)

        # =====================================================================
        # B. Correlation on x_deg (#4-7)
        # =====================================================================
        pcc_deg = per_sample_pcc(y_deg_true, y_deg_pred)
        spearman_deg = per_sample_spearman(y_deg_true, y_deg_pred)

        metrics['pcc_deg_mean'], metrics['pcc_deg_std'] = _stat(pcc_deg)
        metrics['spearman_deg_mean'], metrics['spearman_deg_std'] = _stat(spearman_deg)

        for k in [50, 100, 200]:
            pcc_topk = pcc_top_k_deg(y_deg_true, y_deg_pred, k=k)
            metrics[f'pcc_top{k}_deg_mean'], metrics[f'pcc_top{k}_deg_std'] = _stat(pcc_topk)
            r2_topk_mean, r2_topk_std = r2_top_k_deg(y_deg_true, y_deg_pred, k=k)
            metrics[f'r2_top{k}_deg_mean'] = r2_topk_mean
            metrics[f'r2_top{k}_deg_std'] = r2_topk_std

        # =====================================================================
        # D. Goodness-of-Fit on x_deg (#11)
        # =====================================================================
        metrics['r2_deg_mean'], metrics['r2_deg_std'] = r2_per_gene(y_deg_true, y_deg_pred)

        # #13-14: WMSE / WR2 with fallback weights = |x_deg_true|
        deg_weights = np.abs(y_deg_true)
        wmse = per_sample_wmse(y_deg_true, y_deg_pred, deg_weights)
        wr2 = per_sample_wr2(y_deg_true, y_deg_pred, deg_weights)
        metrics['wmse_mean'], metrics['wmse_std'] = _stat(wmse)
        metrics['wr2_mean'], metrics['wr2_std'] = _stat(wr2)

        # =====================================================================
        # E. Precision@K on x_deg (#15-22)
        # =====================================================================
        for k in [10, 20, 50, 100]:
            pos_pk = precision_at_k(y_deg_true, y_deg_pred, k=k, direction='pos')
            neg_pk = precision_at_k(y_deg_true, y_deg_pred, k=k, direction='neg')
            metrics[f'pos_p{k}_mean'], metrics[f'pos_p{k}_std'] = _stat(pos_pk)
            metrics[f'neg_p{k}_mean'], metrics[f'neg_p{k}_std'] = _stat(neg_pk)

        # =====================================================================
        # F. Distribution metrics on x_deg — Wasserstein (#23)
        # =====================================================================
        metrics['wasserstein_deg_mean'], metrics['wasserstein_deg_std'] = \
            per_gene_wasserstein(y_deg_true, y_deg_pred)

    # =====================================================================
    # F. Distribution metrics on x_deg — MMD (#25)
    # =====================================================================
    if not skip_mmd:
        metrics['mmd_deg_mean'], metrics['mmd_deg_std'] = \
            per_gene_mmd_rbf(y_deg_true, y_deg_pred, gamma=1.0)

    # =====================================================================
    # C. x_pert metrics (#8-10, #12) + F x_pert (#24, #26)
    # =====================================================================
    if y_ctrl is not None:
        y_pert_true = y_ctrl + y_deg_true
        y_pert_pred = y_ctrl + y_deg_pred

        if not mmd_only:
            pcc_pert = per_sample_pcc(y_pert_true, y_pert_pred)
            spearman_pert = per_sample_spearman(y_pert_true, y_pert_pred)

            metrics['pcc_pert_mean'], metrics['pcc_pert_std'] = _stat(pcc_pert)
            metrics['spearman_pert_mean'], metrics['spearman_pert_std'] = _stat(spearman_pert)
            metrics['r2_pert_mean'], metrics['r2_pert_std'] = r2_per_gene(y_pert_true, y_pert_pred)

            metrics['wasserstein_pert_mean'], metrics['wasserstein_pert_std'] = \
                per_gene_wasserstein(y_pert_true, y_pert_pred)

            # #10: Pearson(delta_pert)
            if train_x_pert_mean is not None:
                pcc_delta_pert = pearson_delta_pert(y_pert_true, y_pert_pred, train_x_pert_mean)
                metrics['pcc_delta_pert_mean'], metrics['pcc_delta_pert_std'] = _stat(pcc_delta_pert)

        # MMD on x_pert is dropped (only mmd_deg is reported, since it's the
        # form downstream summaries consume).

    # =====================================================================
    # Per-drug aggregation
    # =====================================================================
    if sample_ids is not None and 'drug_id' in sample_ids:
        drug_ids = sample_ids['drug_id']
        unique_drugs = np.unique(drug_ids)
        drug_pcc_means = []
        drug_pcc_dict = {}
        for drug in unique_drugs:
            mask = drug_ids == drug
            drug_mean_pcc = float(pcc_deg[mask].mean())
            drug_pcc_means.append(drug_mean_pcc)
            drug_pcc_dict[str(drug)] = {
                'pcc_mean': drug_mean_pcc,
                'n_samples': int(mask.sum())
            }
        drug_pcc_arr = np.array(drug_pcc_means)
        metrics['per_drug_pcc_mean'] = float(drug_pcc_arr.mean())
        metrics['per_drug_pcc_std'] = float(drug_pcc_arr.std())
        metrics['per_drug_pcc'] = drug_pcc_dict

    # =====================================================================
    # Meta
    # =====================================================================
    metrics['n_samples'] = int(y_deg_true.shape[0])
    metrics['n_genes'] = int(y_deg_true.shape[1])

    return metrics



