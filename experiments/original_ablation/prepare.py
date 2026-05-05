#!/usr/bin/env python
"""prepare.py — original_ablation (per-model original datasets + drug ablation).

Each model trains on its own paper's original dataset + drug ablation. Data
preparation:
  - extract MultiDCP tar (~11 GB → 30 GB)
  - unrar PertDiT lincs RAR (~8.9 GB → 13 GB)
  - run 6 convert_<M>.py to produce data/_converted/<M>/<M>_original.h5ad
    (+ idx2smi.npy + gene_feature_p6.csv etc.)

Heavy step (~30-60 min total).

Usage:
    python experiments/original_ablation/prepare.py
    python experiments/original_ablation/prepare.py --check-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from prepare_steps import (  # noqa: E402
    add_argparse_options,
    run_steps,
    step_extract_multidcp_data,
    step_extract_pertdit_data,
    step_convert_ciger,
    step_convert_deepce,
    step_convert_multidcp,
    step_convert_pertdit,
    step_convert_prnet,
    step_convert_transigen,
    step_generate_original_drug_splits,
)

STEPS = [
    ("A. extract data/MultiDCP/data.tar.gz",                    step_extract_multidcp_data),
    ("B. unrar  data/PertDiT/lincs_l1000.h5ad",                 step_extract_pertdit_data),
    ("C. convert CIGER     → data/_converted/ciger/",           step_convert_ciger),
    ("D. convert DeepCE    → data/_converted/deepce/",          step_convert_deepce),
    ("E. convert MultiDCP  → data/_converted/multidcp/",        step_convert_multidcp),
    # PRnet must run before PertDiT — convert_pertdit.py reads PRnet's
    # converted prnet_original.h5ad + idx2smi.npy.
    ("F. convert PRnet     → data/_converted/prnet/",           step_convert_prnet),
    ("G. convert PertDiT   → data/_converted/pertdit/",         step_convert_pertdit),
    ("H. convert TranSiGen → data/_converted/transigen/",       step_convert_transigen),
    ("I. add drug_split_0..4 to deepce/multidcp/transigen",     step_generate_original_drug_splits),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argparse_options(p)
    args = p.parse_args()
    print("prepare.py — original_ablation (per-model datasets + ablation)")
    ok = run_steps(STEPS, force=args.force, check_only=args.check_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
