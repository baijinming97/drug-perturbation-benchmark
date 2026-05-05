#!/usr/bin/env python
"""Run original_reproduction: each model on its own paper dataset
in its own native split convention — no ablation.

Per-model split conventions (matching the upstream papers):

    ciger      5-fold drug-blind             results/.../ciger/fold_<F>/
    pertdit    5-fold drug-blind             results/.../pertdit/fold_<F>/
    prnet      5-fold drug-blind             results/.../prnet/fold_<F>/
    deepce     fixed_split × {343,344,345}   results/.../deepce/seed_<S>/
    multidcp   {cell_1, cell_2, cell_3}      results/.../multidcp/cell_<C>/
    transigen  smiles_split × {364039..41}   results/.../transigen/smiles_split/seed_<S>/

Each model's "instances" are its full paper-reproduction set. By default
`train_all.py` runs every model × every instance.

Quick verification (1 epoch, first instance of each of the 6 models):
    python experiments/original_reproduction/train_all.py --epochs 1 --max_instances 1
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
CONV_DIR = REPO_ROOT / "data" / "_converted"
DEFAULT_RESULTS = REPO_ROOT / "results" / "original_reproduction"

# Each model declares the full set of instances it should produce, where each
# instance is a tuple (output_subpath, split_col, extra_cli_args).
#   - output_subpath: relative to results_dir/<model>/  (e.g. "fold_0")
#   - split_col:      adata.obs column carrying train/valid/test labels
#   - extra_cli_args: list[str] appended to the train command for this run
INSTANCES = {
    "ciger":     [(f"fold_{f}",                  f"drug_split_{f}", []) for f in range(5)],
    "deepce":    [(f"seed_{s}",                  "fixed_split",     ["--seed", str(s)]) for s in [343, 344, 345]],
    "multidcp":  [(f"cell_{c}",                  f"cell_{c}",       []) for c in [1, 2, 3]],
    "pertdit":   [(f"fold_{f}",                  f"drug_split_{f}", []) for f in range(5)],
    "prnet":     [(f"fold_{f}",                  f"drug_split_{f}", []) for f in range(5)],
    "transigen": [(f"smiles_split/seed_{s}",     f"smiles_split_{s}", ["--seed", str(s)]) for s in [364039, 364040, 364041]],
}

MODELS = list(INSTANCES)

EPOCH_ARG = {
    "ciger":     "--max_epoch",  "deepce":   "--max_epoch",  "multidcp":  "--max_epoch",
    "pertdit":   "--n_epochs",   "prnet":    "--n_epochs",   "transigen": "--n_epochs",
}

EXTRAS = {
    "ciger":     lambda c: ["--h5ad_path", str(c / "ciger_original.h5ad"),
                            "--smi_path",  str(c / "idx2smi.npy"),
                            "--gene_file", str(c / "gene_feature_p6.csv")],
    "deepce":    lambda c: ["--h5ad_path", str(c / "deepce_original.h5ad"),
                            "--smi_path",  str(c / "idx2smi.npy")],
    "multidcp":  lambda c: ["--h5ad_path", str(c / "multidcp_original.h5ad"),
                            "--smi_path",  str(c / "idx2smi.npy"),
                            "--ae_data_prefix",
                            str(REPO_ROOT / "data" / "MultiDCP" / "extracted" / "data" /
                                "gene_expression_for_ae" /
                                "gene_expression_combat_norm_978_split4"),
                            "--cell_ge_file",
                            str(REPO_ROOT / "data" / "MultiDCP" / "extracted" / "data" /
                                "adjusted_ccle_tcga_ad_tpm_log2.csv"),
                            "--dose_vocab", "0.04 um,0.12 um,0.37 um,1.11 um,3.33 um,10.0 um",
                            "--gene_order_mode", "raw",
                            "--original_data_filter",
                            "--dedup",
                            "--dedup_strategy", "original",
                            "--original_sort",
                            "--output_space", "native",
                            "--early_stop_patience", "0"],
    "pertdit":   lambda c: ["--h5ad_path",     str(c.parent / "prnet" / "prnet_original.h5ad"),
                            "--drug_emb_path", str(c / "p6_drug_emb.pkl"),
                            "--dose_emb_path", str(c / "p6_dose_emb.pkl"),
                            "--dose_col",      "dose"],
    "prnet":     lambda c: ["--h5ad_path", str(c / "prnet_original.h5ad"),
                            "--smi_path",  str(c / "idx2smi.npy"),
                            "--dose_col",  "dose"],
    "transigen": lambda c: ["--h5ad_path", str(c / "transigen_original.h5ad"),
                            "--kpgt_path", str(c / "idx2kpgt.npy")],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--max_instances", type=int, default=None,
                   help="Cap per-model instances (default: run all). "
                        "e.g. --max_instances 1 → quick smoke (one instance per model).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--continue_on_fail", action="store_true")
    return p.parse_args()


def run_one(model: str, inst: tuple[str, str, list[str]], args) -> tuple[str, int, float]:
    sub, split_col, instance_extras = inst
    conv = CONV_DIR / model
    outdir = args.results_dir / model / sub
    outdir.mkdir(parents=True, exist_ok=True)

    # Note: --fold is kept at 0 for the dpb shims that still expect it; the
    # actual split selection is governed by --split_col.
    cmd = [
        sys.executable, str(TRAIN_DIR / f"train_{model}.py"),
        "--fold", "0",
        "--split_col", split_col,
        "--ablation_mode", "none",
        "--output_dir", str(outdir),
        "--dev", args.dev,
    ] + EXTRAS[model](conv) + instance_extras
    if args.epochs is not None:
        cmd += [EPOCH_ARG[model], str(args.epochs)]

    label = f"{model}/{sub}"
    print(f"\n{'='*72}\n  RUN  {label}  (split_col={split_col})\n  CMD  {shlex.join(cmd)}\n{'='*72}",
          flush=True)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        print(f"\n  [EVAL] {label}  (skip_mmd=True; pertdit/prnet test sets are ~175k → MMD OOMs, schema kept uniform across all models)")
        subprocess.run([sys.executable, str(EVAL), str(outdir), "--skip_mmd"], check=True, cwd=REPO_ROOT)
        return label, 0, time.time() - t0
    except subprocess.CalledProcessError as e:
        if args.continue_on_fail:
            print(f"\n[WARN] {label} exited rc={e.returncode}; continuing")
            return label, e.returncode, time.time() - t0
        raise


def main():
    args = parse_args()
    print(f"original_reproduction driver — per-model original datasets")
    print(f"  models      : {args.models}")
    print(f"  results dir : {args.results_dir}")
    print(f"  epochs      : {args.epochs if args.epochs is not None else 'paper default'}")
    if args.max_instances is not None:
        print(f"  max_instances per model: {args.max_instances}")

    summary = []
    for model in args.models:
        instances = INSTANCES[model]
        if args.max_instances is not None:
            instances = instances[: args.max_instances]
        for inst in instances:
            summary.append(run_one(model, inst, args))

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
