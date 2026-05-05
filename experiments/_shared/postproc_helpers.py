"""Shared post-processing helpers.

Single source of truth for how dpb writes the per-prediction sidecar files
(test_sample_ids.csv, per_drug_pcc.csv, train_x_pert_mean.npy). The 7 model
shims under experiments/_shared/training/ inline the same logic; this module
exists so that any data-normalization tool (or future shim refactor) reuses
the canonical code path rather than re-implementing it.

dpb canonical sample_ids schema: integer drug_id (= adata.obs['pert_idx'])
and integer cell_id (= adata.obs['cell_idx']).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def write_sidecar_files(
    pred_dir: Path,
    deg_true: np.ndarray,
    deg_pred: np.ndarray,
    h5ad_path: Path,
    split_col: str,
) -> None:
    """Generate train_x_pert_mean.npy + test_sample_ids.csv + per_drug_pcc.csv
    from the raw test predictions and the original h5ad's split column.

    Assumes test_predictions.npy / test_ground_truth.npy / test_ctrl.npy have
    already been written by the caller.
    """
    import scanpy as sc

    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(h5ad_path)
    train_mask = (adata.obs[split_col].astype(str) == "train").values
    test_mask = (adata.obs[split_col].astype(str) == "test").values
    if test_mask.sum() != deg_pred.shape[0]:
        raise ValueError(
            f"split_col={split_col} test rows={test_mask.sum()} "
            f"≠ predictions rows {deg_pred.shape[0]}"
        )

    train_x_pert = adata.X[train_mask]
    if hasattr(train_x_pert, "toarray"):
        train_x_pert = train_x_pert.toarray()
    np.save(pred_dir / "train_x_pert_mean.npy",
            train_x_pert.mean(axis=0).astype(np.float32))

    pert_idx_all = adata.obs["pert_idx"].values.astype(int)
    cell_idx_all = adata.obs["cell_idx"].values.astype(int)
    sample_ids = pd.DataFrame({
        "drug_id": pert_idx_all[test_mask],
        "cell_id": cell_idx_all[test_mask],
    })
    sample_ids.to_csv(pred_dir / "test_sample_ids.csv", index=False)

    drug_ids = sample_ids["drug_id"].values
    drug_pccs = []
    for drug in np.unique(drug_ids):
        dmask = drug_ids == drug
        pccs = []
        for i in np.where(dmask)[0]:
            gt, pr = deg_true[i], deg_pred[i]
            if np.std(gt) > 0 and np.std(pr) > 0:
                r, _ = pearsonr(gt, pr)
                pccs.append(r)
        if pccs:
            drug_pccs.append({
                "drug_id": int(drug),
                "pcc": float(np.mean(pccs)),
                "n_samples": int(dmask.sum()),
            })
    pd.DataFrame(drug_pccs).to_csv(pred_dir / "per_drug_pcc.csv", index=False)
