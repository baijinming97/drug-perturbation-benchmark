# original_reproduction

Reproduces each model's published numbers on its **original** dataset (not the
unified L1000_sdst). Tests whether the training scripts under `models/<M>/`
faithfully replicate upstream behavior.

6 vendored models (CIGER, DeepCE, MultiDCP, PertDiT, PRnet, TranSiGen) — each
trained on its own paper's released dataset (uniformly converted to h5ad
under `data/_converted/<M>/`). XPert is **not** in this driver because XPert's
"original dataset" is L1000_sdst (the unified benchmark data), which would
duplicate `main_benchmark/`.

Each upstream paper uses a different evaluation protocol; this driver keeps
each model in its native convention rather than imposing a uniform fold scheme:

| Model     | Instance convention                  | Output subpath                      |
|-----------|--------------------------------------|-------------------------------------|
| ciger     | 5-fold drug-blind                    | `ciger/fold_<F>/`                   |
| pertdit   | 5-fold drug-blind                    | `pertdit/fold_<F>/`                 |
| prnet     | 5-fold drug-blind                    | `prnet/fold_<F>/`                   |
| deepce    | fixed_split × 3 seeds (343/344/345)  | `deepce/seed_<S>/`                  |
| multidcp  | 3 cell-line splits (cell_1/2/3)      | `multidcp/cell_<C>/`                |
| transigen | smiles_split × 3 seeds (364039..41)  | `transigen/smiles_split/seed_<S>/`  |

By default `train.sh` runs every model × every instance; `--max_instances 1`
runs one instance per model (smoke test).

## Data preparation

`prepare.py` runs the following idempotent steps (already-finished work is
skipped):

* Extract `data/MultiDCP/data.tar.gz` and `data/PertDiT/lincs_l1000.h5ad` (RAR)
* Convert each upstream format to `data/_converted/<M>/<M>_original.h5ad`
* Add `drug_split_0..4` columns to the deepce / multidcp / transigen h5ad
  files (seed=42, 5-fold KFold over unique drugs; in-place modification).
  This column is consumed by `original_ablation/`; this task itself still
  uses each model's native split as listed in the table above.

Each model's `train_<M>.py` is invoked with model-specific extras (e.g.
`--ae_data_prefix` for MultiDCP, `--drug_emb_path` / `--dose_col` for
PertDiT), all defined in the `MODELS` dict of `train_all.py`.

## Quick start

```bash
conda activate benchmark
python experiments/original_reproduction/prepare.py
bash   experiments/original_reproduction/train.sh --epochs 1 --max_instances 1   # one-instance-per-model sanity
bash   experiments/original_reproduction/train.sh                                 # full run
```

Full CLI: `python experiments/original_reproduction/train_all.py --help`.

## Output layout

```
results/original_reproduction/
├── ciger/fold_<0..4>/{checkpoints, predictions, metrics, logs}
├── pertdit/fold_<0..4>/...
├── prnet/fold_<0..4>/...
├── deepce/seed_<343|344|345>/...
├── multidcp/cell_<1|2|3>/...
└── transigen/smiles_split/seed_<364039|364040|364041>/...
```
