#!/usr/bin/env python
"""prepare.py — mlp_baseline (drug-blind MLP).

The MLP baseline only needs the L1000 h5ad and drug-blind splits — it does not
use CIGER gene-feature mapping or TranSiGen VAE checkpoints.

Usage:
    python experiments/mlp_baseline/prepare.py
    python experiments/mlp_baseline/prepare.py --check-only
    python experiments/mlp_baseline/prepare.py --force
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
    step_generate_drug_blind_splits,
)

STEPS = [
    ("A. extract data/XPert/processed_data.zip → data/XPert/processed_data/", step_extract_processed_data),
    ("B. add nm_drug_blind_1..5 to h5ad (5-fold drug-disjoint)",   step_generate_drug_blind_splits),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argparse_options(p)
    args = p.parse_args()
    print("prepare.py — mlp_baseline (drug-blind MLP)")
    ok = run_steps(STEPS, force=args.force, check_only=args.check_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
