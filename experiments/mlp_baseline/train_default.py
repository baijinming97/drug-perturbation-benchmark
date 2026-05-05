#!/usr/bin/env python
"""Run mlp_baseline: drug-blind MLP, 5-fold cold-drug.

Default configuration: 3 layers × hidden 2048 (paper default = L3_H2048).
For the layers × hidden grid sweep, see `sweep.py`.

Quick sanity (1 epoch, fold 0):
    python experiments/mlp_baseline/train_default.py --epochs 1 --folds 0

Full reproduction (5 fold × 500 epoch ≈ 1 h on a100):
    python experiments/mlp_baseline/train_default.py
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN = REPO_ROOT / "experiments" / "_shared" / "training" / "train_mlp.py"
EVAL = REPO_ROOT / "experiments" / "_shared" / "evaluation" / "evaluate_one_fold.py"
DEFAULT_RESULTS = REPO_ROOT / "results" / "mlp_baseline" / "default"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max epochs (default: paper default 500).")
    p.add_argument("--num_layers", type=int, default=3,
                   help="MLP hidden layers (default 3 = paper default).")
    p.add_argument("--mlp_hidden", type=int, default=2048,
                   help="MLP hidden dimension (default 2048 = paper default).")
    p.add_argument("--seed", type=int, default=131419)
    p.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--continue_on_fail", action="store_true")
    return p.parse_args()


def run_one(fold: int, args) -> tuple[int, float]:
    split_col = f"nm_drug_blind_{fold + 1}"
    outdir = args.results_dir / f"fold_{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRAIN),
        "--split_col", split_col,
        "--output_dir", str(outdir),
        "--num_layers", str(args.num_layers),
        "--mlp_hidden", str(args.mlp_hidden),
        "--seed", str(args.seed),
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]

    print(f"\n{'='*72}")
    print(f"  RUN  fold_{fold}  (L{args.num_layers}_H{args.mlp_hidden}, split={split_col})")
    print(f"  CMD  {shlex.join(cmd)}")
    print(f"{'='*72}", flush=True)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        # Auto-evaluate: produce metrics/all_metrics.json from predictions/
        print(f"\n  [EVAL] {outdir.name}")
        subprocess.run([sys.executable, str(EVAL), str(outdir)],
                       check=True, cwd=REPO_ROOT)
        return 0, time.time() - t0
    except subprocess.CalledProcessError as e:
        if args.continue_on_fail:
            print(f"\n[WARN] fold_{fold} exited rc={e.returncode}; continuing")
            return e.returncode, time.time() - t0
        raise


def main():
    args = parse_args()
    print(f"mlp_baseline driver — drug-blind MLP")
    print(f"  config     : L{args.num_layers}_H{args.mlp_hidden}, seed={args.seed}")
    print(f"  results dir: {args.results_dir}")
    print(f"  folds      : {args.folds}")
    print(f"  epochs     : {args.epochs if args.epochs is not None else 'paper default'}")

    summary: list[tuple[int, int, float]] = []
    for fold in args.folds:
        rc, secs = run_one(fold, args)
        summary.append((fold, rc, secs))

    print(f"\n{'='*72}\n  DONE — {len(summary)} folds\n{'='*72}")
    for fold, rc, secs in summary:
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  {status:<10} {secs:>7.1f}s  fold_{fold}")
    if any(rc != 0 for _, rc, _ in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
