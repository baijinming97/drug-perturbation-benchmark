# main_benchmark

Drug-blind cold-split 5-fold benchmark on the unified L1000_sdst dataset,
covering 7 models (CIGER, DeepCE, MultiDCP, PertDiT, PRnet, TranSiGen, XPert).

## Quick start

```bash
conda activate benchmark
python experiments/main_benchmark/prepare.py                            # one-shot data prep (idempotent)
bash   experiments/main_benchmark/train.sh --epochs 1 --folds 0         # ~30 min sanity
bash   experiments/main_benchmark/train.sh                              # full 5-fold (~20 h)
bash   experiments/main_benchmark/train.sh --models xpert               # single model, full 5-fold
```

Full CLI: `python experiments/main_benchmark/train_all.py --help`.

## Output layout

```
results/main_benchmark/
└── <model>/
    └── fold_<0..4>/{checkpoints, predictions, metrics, logs}
```
