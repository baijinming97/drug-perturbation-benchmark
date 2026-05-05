"""Evaluate a single fold directory. Wraps evaluate_all.py logic."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_all import evaluate_predictions
import numpy as np
import pandas as pd
import json
from pathlib import Path

fold_dir = Path(sys.argv[1])
skip_mmd = '--skip_mmd' in sys.argv[2:]
pred_dir = fold_dir / 'predictions'
metric_dir = fold_dir / 'metrics'
metric_dir.mkdir(exist_ok=True)

print(f"Evaluating: {fold_dir}")

y_pred = np.load(pred_dir / 'test_predictions.npy')
y_true = np.load(pred_dir / 'test_ground_truth.npy')
print(f"  pred={y_pred.shape}, gt={y_true.shape}")

y_ctrl = None
ctrl_file = pred_dir / 'test_ctrl.npy'
if ctrl_file.exists():
    y_ctrl = np.load(ctrl_file)

ids_df = None
ids_file = pred_dir / 'test_sample_ids.csv'
if ids_file.exists():
    ids_df = pd.read_csv(ids_file)

train_mean = None
mean_file = pred_dir / 'train_x_pert_mean.npy'
if mean_file.exists():
    train_mean = np.load(mean_file)

metrics = evaluate_predictions(y_true, y_pred, y_ctrl, ids_df, train_mean, skip_mmd=skip_mmd)

out_file = metric_dir / 'all_metrics.json'
with open(out_file, 'w') as f:
    json.dump(metrics, f, indent=2, default=float)
print(f"  Saved: {out_file}")

# Key metrics summary
print(f"  PCC_deg={metrics.get('pcc_deg_mean',0):.4f}  R2_deg={metrics.get('r2_deg_mean',0):.4f}  RMSE_deg={metrics.get('rmse_deg_mean',0):.4f}")

# Per-drug PCC. test_predictions.npy / test_ground_truth.npy are already DEG
# (delta-expression) — every shim writes them in DEG form. Subtracting y_ctrl
# again would compute pearson(DEG - x_ctl, DEG - x_ctl), which has no meaning.
if ids_df is not None and 'drug_id' in ids_df.columns:
    from evaluate_all import _pcc_vectorized
    pcc_arr = _pcc_vectorized(y_true, y_pred)
    ids_df['pcc'] = pcc_arr
    drug_df = ids_df.groupby('drug_id').agg(pcc=('pcc','mean'), n_samples=('pcc','count')).reset_index()
    drug_csv = pred_dir / 'per_drug_pcc.csv'
    drug_df.to_csv(drug_csv, index=False)
    print(f"  per_drug_pcc: {len(drug_df)} drugs")

print("[DONE]")
