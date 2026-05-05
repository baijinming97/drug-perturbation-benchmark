#!/usr/bin/env python
"""prepare.py — scaffold_generalization (Bemis-Murcko scaffold-disjoint splits).

Same h5ad as main_benchmark but uses `nm_scaffold_1..5` instead of drug-blind splits.

Usage:
    python experiments/scaffold_generalization/prepare.py
    python experiments/scaffold_generalization/prepare.py --check-only
    python experiments/scaffold_generalization/prepare.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from prepare_steps import (  # noqa: E402
    add_argparse_options,
    run_steps,
    step_extract_processed_data,
    step_extract_xpert_unimol_arr,
    step_extract_pertdit_data,
    step_generate_gene_feature_nm,
    step_generate_pertdit_embeddings,
    step_verify_transigen_vae,
    step_generate_scaffold_splits,
    step_generate_xpert_ablation_data,
    step_verify_xpert_hg_data,
)

STEPS = [
    ("A. extract data/XPert/processed_data.zip → data/XPert/processed_data/",      step_extract_processed_data),
    ("A2. extract Figshare all_drugs_unimol_arr.zip → data/XPert/processed_data/", step_extract_xpert_unimol_arr),
    ("B. generate data/CIGER/gene_feature_nm.csv (5-gene rename)",      step_generate_gene_feature_nm),
    ("C. verify reference/TranSiGen/results/ VAE checkpoints",          step_verify_transigen_vae),
    ("D. add nm_scaffold_1..5 to h5ad (Bemis-Murcko scaffold-disjoint)", step_generate_scaffold_splits),
    ("D2. unrar data/PertDiT/lincs_l1000.h5ad → data/PertDiT/extracted/ (RAR ~8.9 GB → 13 GB; needed by step E for negative_ctrl token)",
                                                                        step_extract_pertdit_data),
    ("E. generate PertDiT MolT5+BioLinkBERT embeddings (~25 min)",      step_generate_pertdit_embeddings),
    ("F. verify reference/XPert/HG_data exists (XPert input)",          step_verify_xpert_hg_data),
    ("G. generate XPert zero drug arrays (for --ablation_mode zero)",   step_generate_xpert_ablation_data),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argparse_options(p)
    args = p.parse_args()
    print("prepare.py — scaffold_generalization (scaffold-disjoint splits)")
    ok = run_steps(STEPS, force=args.force, check_only=args.check_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
