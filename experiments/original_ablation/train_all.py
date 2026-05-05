#!/usr/bin/env python
"""Run original_ablation: each model on its own paper dataset
with drug-input ablation (none + zero).

60 runs total: 6 models × {none, zero} × 5 folds (default).

Quick sanity (1 epoch, fold 0, none+zero × 6 models = 12 runs, ~20-40 min):
    python experiments/original_ablation/train_all.py --epochs 1 --folds 0

Full reproduction (paper defaults, all 5 folds):
    python experiments/original_ablation/train_all.py
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "experiments" / "_shared" / "training"
EVAL = REPO_ROOT / "experiments" / "_shared" / "evaluation" / "evaluate_one_fold.py"
CONV_DIR = REPO_ROOT / "data" / "_converted"
DEFAULT_RESULTS = REPO_ROOT / "results" / "original_ablation"

# Each model's CLI shape. extras() returns the model-specific data flags.
# All share: --fold --split_col --ablation_mode --output_dir + epoch flag.
MODELS = {
    "ciger":     {"epoch_arg": "--max_epoch",
                  "extras": lambda c: ["--h5ad_path", str(c / "ciger_original.h5ad"),
                                       "--smi_path",  str(c / "idx2smi.npy"),
                                       "--gene_file", str(c / "gene_feature_p6.csv")]},
    "deepce":    {"epoch_arg": "--max_epoch",
                  "extras": lambda c: ["--h5ad_path", str(c / "deepce_original.h5ad"),
                                       "--smi_path",  str(c / "idx2smi.npy")]},
    "multidcp":  {"epoch_arg": "--max_epoch",
                  "extras": lambda c: ["--h5ad_path", str(c / "multidcp_original.h5ad"),
                                       "--smi_path",  str(c / "idx2smi.npy"),
                                       # MultiDCP needs ae_data_prefix + dose_vocab — adjust per data
                                       "--ae_data_prefix",
                                       str(REPO_ROOT / "data" / "MultiDCP" / "extracted" / "data" /
                                           "gene_expression_for_ae" /
                                           "gene_expression_combat_norm_978_split4"),
                                       "--dose_vocab", "10.0 um"]},
    "pertdit":   {"epoch_arg": "--n_epochs",
                  "extras": lambda c: ["--h5ad_path",     str(c.parent / "prnet" / "prnet_original.h5ad"),
                                       "--drug_emb_path", str(c / "p6_drug_emb.pkl"),
                                       "--dose_emb_path", str(c / "p6_dose_emb.pkl"),
                                       "--dose_col",      "dose"]},
    "prnet":     {"epoch_arg": "--n_epochs",
                  "extras": lambda c: ["--h5ad_path", str(c / "prnet_original.h5ad"),
                                       "--smi_path",  str(c / "idx2smi.npy"),
                                       "--dose_col",  "dose"]},
    "transigen": {"epoch_arg": "--n_epochs",
                  "extras": lambda c: ["--h5ad_path", str(c / "transigen_original.h5ad"),
                                       "--kpgt_path", str(c / "idx2kpgt.npy")]},
}

ABLATION_CHOICES = ("none", "zero")
DEFAULT_ABLATION_MODES = ["none", "zero"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--ablation_modes", nargs="+",
                   default=DEFAULT_ABLATION_MODES, choices=ABLATION_CHOICES,
                   help="Ablation modes to run (default: none zero).")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--continue_on_fail", action="store_true")
    return p.parse_args()


def run_one(model: str, ab: str, fold: int, args) -> tuple[str, int, float]:
    cfg = MODELS[model]
    conv = CONV_DIR / model
    outdir = args.results_dir / model / ab / f"fold_{fold}"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAIN_DIR / f"train_{model}.py"),
        "--fold", str(fold),
        "--split_col", f"drug_split_{fold}",     # original_ablation uses drug_split_<N>
        "--ablation_mode", ab,
        "--output_dir", str(outdir),
        "--dev", args.dev,
    ] + cfg["extras"](conv)
    if args.epochs is not None:
        cmd += [cfg["epoch_arg"], str(args.epochs)]

    label = f"{model}/{ab}/fold_{fold}"
    print(f"\n{'='*72}\n  RUN  {label}\n  CMD  {shlex.join(cmd)}\n{'='*72}", flush=True)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        # All 6 models in original_ablation skip MMD: pertdit/prnet test sets
        # are ~170k samples (biased full-N kernel would OOM at 200+ GB), and
        # the other 4 models skip too for schema uniformity. Matches the
        # original_reproduction/train_all.py convention.
        print(f"\n  [EVAL] {label}  (skip_mmd=True)")
        subprocess.run([sys.executable, str(EVAL), str(outdir), "--skip_mmd"],
                       check=True, cwd=REPO_ROOT)
        return label, 0, time.time() - t0
    except subprocess.CalledProcessError as e:
        if args.continue_on_fail:
            print(f"\n[WARN] {label} exited rc={e.returncode}; continuing")
            return label, e.returncode, time.time() - t0
        raise


def main():
    args = parse_args()
    runs = list(product(args.models, args.ablation_modes, args.folds))
    print(f"original_ablation driver — per-model datasets + drug ablation")
    print(f"  models     : {args.models}")
    print(f"  ablation_modes : {args.ablation_modes}")
    print(f"  folds      : {args.folds}")
    print(f"  total runs : {len(runs)}")
    print(f"  results dir: {args.results_dir}")
    print(f"  epochs     : {args.epochs if args.epochs is not None else 'paper default'}")

    summary = []
    for model, ab, fold in runs:
        summary.append(run_one(model, ab, fold, args))

    print(f"\n{'='*72}\n  DONE — {len(summary)} runs\n{'='*72}")
    fails = sum(1 for _, rc, _ in summary if rc != 0)
    print(f"  {len(summary) - fails} OK, {fails} FAIL")
    for label, rc, secs in summary:
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  {status:<10} {secs:>7.1f}s  {label}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
