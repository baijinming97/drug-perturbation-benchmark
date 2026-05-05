#!/usr/bin/env python
"""Layers × hidden grid sweep over MLP architectures.

Default sweep matrix:
    layers  ∈ {1, 2, 3, 4}
    hidden  ∈ {256, 512, 1024, 2048, 4096}
    folds   ∈ {0, 1, 2, 3, 4}

= 4 × 5 × 5 = 100 runs (sequential on one GPU). On a100 each run takes ≈ 5–10
min for 500 epochs, so the full sweep is ~10–15 hours.

Output: results/mlp_baseline/sweep/L<layers>_H<hidden>/fold_<N>/

Examples:
    # Quick check: only 2 configs × 1 fold × 1 epoch
    python experiments/mlp_baseline/sweep.py \
        --layers_set 1 4 --hidden_set 256 4096 --folds 0 --epochs 1

    # Full sweep
    python experiments/mlp_baseline/sweep.py
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
TRAIN = REPO_ROOT / "experiments" / "_shared" / "training" / "train_mlp.py"
EVAL = REPO_ROOT / "experiments" / "_shared" / "evaluation" / "evaluate_one_fold.py"
DEFAULT_RESULTS = REPO_ROOT / "results" / "mlp_baseline" / "sweep"

LAYERS_DEFAULT = [1, 2, 3, 4]
HIDDEN_DEFAULT = [256, 512, 1024, 2048, 4096]
FOLDS_DEFAULT = [0, 1, 2, 3, 4]
SEED_DEFAULT = 131419


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers_set", type=int, nargs="+", default=LAYERS_DEFAULT)
    p.add_argument("--hidden_set", type=int, nargs="+", default=HIDDEN_DEFAULT)
    p.add_argument("--folds", type=int, nargs="+", default=FOLDS_DEFAULT)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--continue_on_fail", action="store_true")
    p.add_argument("--skip_default", action="store_true",
                   help="Skip L3_H2048 (= main_benchmark default).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing.")
    return p.parse_args()


def run_one(layers: int, hidden: int, fold: int, args) -> tuple[int, float]:
    split_col = f"nm_drug_blind_{fold + 1}"
    outdir = args.results_dir / f"L{layers}_H{hidden}" / f"fold_{fold}"
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRAIN),
        "--split_col", split_col,
        "--output_dir", str(outdir),
        "--num_layers", str(layers),
        "--mlp_hidden", str(hidden),
        "--seed", str(args.seed),
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]

    label = f"L{layers}_H{hidden}/fold_{fold}"
    print(f"\n{'='*72}")
    print(f"  RUN  {label}  (split={split_col})")
    print(f"  CMD  {shlex.join(cmd)}")
    print(f"{'='*72}", flush=True)

    if args.dry_run:
        return 0, 0.0

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        # Auto-evaluate: produce metrics/all_metrics.json from predictions/
        print(f"\n  [EVAL] {label}")
        subprocess.run([sys.executable, str(EVAL), str(outdir)],
                       check=True, cwd=REPO_ROOT)
        return 0, time.time() - t0
    except subprocess.CalledProcessError as e:
        if args.continue_on_fail:
            print(f"\n[WARN] {label} exited rc={e.returncode}; continuing")
            return e.returncode, time.time() - t0
        raise


def main():
    args = parse_args()
    print(f"mlp_baseline sweep — layers × hidden grid")
    print(f"  layers_set : {args.layers_set}")
    print(f"  hidden_set : {args.hidden_set}")
    print(f"  folds      : {args.folds}")
    print(f"  seed       : {args.seed}")
    print(f"  epochs     : {args.epochs if args.epochs is not None else 'paper default'}")
    print(f"  results dir: {args.results_dir}")
    print(f"  skip L3_H2048 (= default): {args.skip_default}")
    print(f"  dry_run    : {args.dry_run}")

    summary: list[tuple[str, int, float]] = []
    for layers, hidden, fold in product(args.layers_set, args.hidden_set, args.folds):
        if args.skip_default and layers == 3 and hidden == 2048:
            continue
        rc, secs = run_one(layers, hidden, fold, args)
        summary.append((f"L{layers}_H{hidden}/fold_{fold}", rc, secs))

    print(f"\n{'='*72}\n  DONE — {len(summary)} runs\n{'='*72}")
    for label, rc, secs in summary:
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  {status:<10} {secs:>7.1f}s  {label}")
    if any(rc != 0 for _, rc, _ in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
