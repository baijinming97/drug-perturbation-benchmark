#!/usr/bin/env python
"""Run main_benchmark: cold-drug 5-fold reproduction.

This is the **SLURM-free** entry point — sequential subprocess calls, runs on a
single GPU. Use this when reproducing without a job scheduler (laptop, single-
GPU server, or interactive HPC node).

Quick verification (1 epoch, fold 0, all 7 models, ~30-50 min on A100):
    python experiments/main_benchmark/train_all.py --epochs 1 --folds 0

Single model, all 5 folds, paper-default epochs:
    python experiments/main_benchmark/train_all.py --models ciger

Full reproduction (~20 hr on A100):
    python experiments/main_benchmark/train_all.py

Output layout:
    results/main_benchmark/<model>/fold_<N>/{checkpoints, predictions, metrics, logs}

XPert is wrapped via experiments/_shared/training/train_xpert.py (translates the
unified CLI to XPert's YAML-driven entry point + post-processes outputs to the
unified output format). Reviewers can also run XPert standalone via
`models/XPert/train_xpert.py --config config_l1000 ...` for paper-figure scripts.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "experiments" / "_shared" / "training"
EVAL = REPO_ROOT / "experiments" / "_shared" / "evaluation" / "evaluate_one_fold.py"
DEFAULT_RESULTS = REPO_ROOT / "results" / "main_benchmark"

# Each entry: (script filename, epoch CLI flag) — paper defaults are taken from
# each train_<M>.py argparse default; override via --epochs for fast verification.
MODELS = {
    "ciger":     {"train": "train_ciger.py",     "epoch_arg": "--max_epoch"},
    "deepce":    {"train": "train_deepce.py",    "epoch_arg": "--max_epoch"},
    "multidcp":  {"train": "train_multidcp.py",  "epoch_arg": "--max_epoch"},
    "pertdit":   {"train": "train_pertdit.py",   "epoch_arg": "--n_epochs"},
    "prnet":     {"train": "train_prnet.py",     "epoch_arg": "--n_epochs"},
    "transigen": {"train": "train_transigen.py", "epoch_arg": "--n_epochs"},
    "xpert":     {"train": "train_xpert.py",     "epoch_arg": "--max_epoch"},
}

ABLATION_CHOICES = ("none", "zero", "shuffle")
SPLIT_COL_TEMPLATE = "nm_drug_blind_{fold_one_indexed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run main_benchmark cold-drug reproduction without SLURM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Quick verification")[1].rsplit("XPert", 1)[0],
    )
    p.add_argument(
        "--models", nargs="+", default=list(MODELS),
        choices=list(MODELS),
        help="Model subset (default: all 6).",
    )
    p.add_argument(
        "--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Fold indices 0..4 (default: all).",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs (default: each model's paper default).",
    )
    p.add_argument(
        "--ablation", choices=ABLATION_CHOICES, default="none",
        help="Drug ablation mode (default: none = no ablation).",
    )
    p.add_argument(
        "--results_dir", type=Path, default=DEFAULT_RESULTS,
        help=f"Output base dir (default: {DEFAULT_RESULTS.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--dev", default="cuda:0",
        help="PyTorch device string (default: cuda:0).",
    )
    p.add_argument(
        "--continue_on_fail", action="store_true",
        help="If a model×fold fails, log and continue instead of stopping.",
    )
    return p.parse_args()


def run_one(model: str, fold: int, args: argparse.Namespace) -> tuple[str, int, float]:
    cfg = MODELS[model]
    split_col = SPLIT_COL_TEMPLATE.format(fold_one_indexed=fold + 1)
    outdir = args.results_dir / model / f"fold_{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRAIN_DIR / cfg["train"]),
        "--fold", str(fold),
        "--split_col", split_col,
        "--ablation_mode", args.ablation,
        "--output_dir", str(outdir),
        "--dev", args.dev,
    ]
    if args.epochs is not None:
        cmd += [cfg["epoch_arg"], str(args.epochs)]

    label = f"{model} fold_{fold}"
    print(f"\n{'='*72}")
    print(f"  RUN  {label}  ({split_col}, ablation={args.ablation})")
    print(f"  CMD  {shlex.join(cmd)}")
    print(f"{'='*72}", flush=True)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        rc = 0
        # Auto-evaluate: produce metrics/all_metrics.json
        print(f"\n  [EVAL] {label}")
        subprocess.run([sys.executable, str(EVAL), str(outdir)],
                       check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        rc = e.returncode
        if not args.continue_on_fail:
            raise
        print(f"\n[WARN] {label} exited with code {rc}; continuing per --continue_on_fail")
    return label, rc, time.time() - t0


def main() -> None:
    args = parse_args()
    print(f"main_benchmark driver — cold-drug 5-fold reproduction")
    print(f"  repo root  : {REPO_ROOT}")
    print(f"  results dir: {args.results_dir}")
    print(f"  models     : {args.models}")
    print(f"  folds      : {args.folds}")
    print(f"  epochs     : {args.epochs if args.epochs is not None else 'paper defaults'}")
    print(f"  ablation   : {args.ablation}")
    print(f"  device     : {args.dev}")

    summary: list[tuple[str, int, float]] = []
    for model in args.models:
        for fold in args.folds:
            summary.append(run_one(model, fold, args))

    print(f"\n{'='*72}")
    print(f"  DONE — {len(summary)} runs total")
    print(f"{'='*72}")
    for label, rc, secs in summary:
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  {status:<10} {secs:>7.1f}s  {label}")

    failed = [s for s in summary if s[1] != 0]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
