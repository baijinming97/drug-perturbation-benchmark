"""
Prepare PertDiT embeddings for original Lincs data.

PertDiT and PRNet share the same h5ad (Lincs_L1000.h5ad → prnet_original.h5ad).
This script only converts embeddings:
  1. Re-key original drug embeddings: SMILES → pert_idx
  2. Re-key dose embeddings: np.float64 → rounded float, with tolerance matching
  3. Create dose_map array for h5ad dose values → embedding dict keys

Input:
  - PRNet converted idx2smi.npy (pert_idx → SMILES)
  - PRNet converted h5ad (dose values in obs)
  - Original PertDiT pert_smiles_emb.pkl (SMILES → Tensor)
  - Original PertDiT dosage_prompt_emb_lincs.pkl (dose → Tensor)

Output:
  - p6_drug_emb.pkl: {int(pert_idx) → Tensor(L, 1024)} + 'negative_ctrl'
  - p6_dose_emb.pkl: {float(dose) → Tensor(L, 1024)} with h5ad-compatible keys

PertDiT training reads the h5ad directly from data/_converted/prnet/prnet_original.h5ad;
this script no longer creates symlinks under data/_converted/pertdit/.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
# PRnet must be converted first — this script reads its prnet_original.h5ad
# and idx2smi.npy. prepare.py orders PRnet (step F) before PertDiT (step G).
PRNET_CONV = str(REPO_ROOT / "data" / "_converted" / "prnet")
# step_extract_pertdit_data unrars data/PertDiT/lincs_l1000.h5ad (a RAR
# despite its .h5ad name) into data/PertDiT/extracted/, which contains
# pert_smiles_emb.pkl + dosage_prompt_emb_lincs.pkl + lincs_adata.h5ad.
PERTDIT_DATA = str(REPO_ROOT / "data" / "PertDiT" / "extracted")
OUT_DIR = str(REPO_ROOT / "data" / "_converted" / "pertdit")


def find_nearest_key(val, keys_array, tol=0.01):
    diffs = np.abs(keys_array - val)
    best_idx = np.argmin(diffs)
    if diffs[best_idx] < tol:
        return keys_array[best_idx]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prnet_dir", type=str, default=PRNET_CONV)
    p.add_argument("--pertdit_data", type=str, default=PERTDIT_DATA)
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    # PertDiT conversion depends on PRnet's converted output. Fail fast with a
    # clear message if PRnet hasn't been converted yet, instead of letting
    # np.load / sc.read_h5ad surface a confusing FileNotFoundError later.
    required = [
        os.path.join(args.prnet_dir, "prnet_original.h5ad"),
        os.path.join(args.prnet_dir, "idx2smi.npy"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        sys.exit(
            "ERROR: convert_pertdit.py depends on PRnet's converted output, "
            "but the following files are missing:\n  - "
            + "\n  - ".join(missing)
            + "\nRun convert_prnet.py first (prepare.py orders it before pertdit)."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PertDiT embedding conversion")
    print("=" * 60)

    # ── 1. Load idx2smi from PRNet converter ────────────────────────────────
    print("\n[1/4] Loading idx2smi ...")
    idx2smi = np.load(os.path.join(args.prnet_dir, "idx2smi.npy"),
                      allow_pickle=True).item()
    print(f"  {len(idx2smi)} drugs")

    # ── 2. Re-key drug embeddings: SMILES → pert_idx ────────────────────────
    print("\n[2/4] Converting drug embeddings ...")
    orig_drug = torch.load(
        os.path.join(args.pertdit_data, "pert_smiles_emb.pkl"),
        map_location="cpu", weights_only=False)

    p6_drug = {"negative_ctrl": orig_drug["negative_ctrl"]}
    missing = []
    for idx, smi in idx2smi.items():
        if smi in orig_drug:
            p6_drug[idx] = orig_drug[smi]
        else:
            missing.append(idx)
    print(f"  Mapped: {len(p6_drug)-1}/{len(idx2smi)} drugs")
    if missing:
        print(f"  WARNING: {len(missing)} drugs missing (will use negative_ctrl)")
        for idx in missing:
            p6_drug[idx] = orig_drug["negative_ctrl"]

    # ── 3. Re-key dose embeddings with tolerance matching ───────────────────
    print("\n[3/4] Converting dose embeddings ...")
    orig_dose = torch.load(
        os.path.join(args.pertdit_data, "dosage_prompt_emb_lincs.pkl"),
        map_location="cpu", weights_only=False)

    emb_keys_float = sorted([float(k) for k in orig_dose.keys()])
    emb_keys_array = np.array(emb_keys_float)
    print(f"  Original dose embedding keys: {len(emb_keys_float)}")

    # Load unique dose values from h5ad
    print("  Loading dose values from PRNet h5ad ...")
    adata = sc.read_h5ad(os.path.join(args.prnet_dir, "prnet_original.h5ad"),
                         backed="r")
    data_doses = np.unique(adata.obs["dose"].values.astype(float))
    print(f"  Unique dose values in data: {len(data_doses)}")

    # Build dose embedding dict with data-compatible keys
    p6_dose = {}
    matched = 0
    unmatched_doses = []

    for dose_val in data_doses:
        nearest = find_nearest_key(dose_val, emb_keys_array, tol=0.01)
        if nearest is not None:
            orig_key = [k for k in orig_dose.keys() if abs(float(k) - nearest) < 1e-8][0]
            p6_dose[dose_val] = orig_dose[orig_key]
            matched += 1
        else:
            unmatched_doses.append(dose_val)

    print(f"  Matched: {matched}/{len(data_doses)} doses")
    if unmatched_doses:
        print(f"  Unmatched: {len(unmatched_doses)} doses → using dose=10.0 embedding")
        dose_10_key = [k for k in orig_dose.keys() if abs(float(k) - 10.0) < 0.01][0]
        fallback_emb = orig_dose[dose_10_key]
        for d in unmatched_doses:
            p6_dose[d] = fallback_emb

    # ── 4. Save ─────────────────────────────────────────────────────────────
    print("\n[4/4] Saving ...")
    drug_path = os.path.join(args.output_dir, "p6_drug_emb.pkl")
    dose_path = os.path.join(args.output_dir, "p6_dose_emb.pkl")

    torch.save(p6_drug, drug_path)
    torch.save(p6_dose, dose_path)

    print(f"  Saved: {drug_path} ({len(p6_drug)-1} drugs + negative_ctrl)")
    print(f"  Saved: {dose_path} ({len(p6_dose)} dose levels)")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
