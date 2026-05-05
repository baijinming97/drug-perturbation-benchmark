# mlp_baseline

Drug-blind MLP baseline. Includes the default configuration (L3_H2048: 3 hidden
layers, 2048 hidden units) and a layers × hidden grid sweep.

## Quick start

```bash
conda activate benchmark
python experiments/mlp_baseline/prepare.py                              # one-shot data check (idempotent)
bash   experiments/mlp_baseline/train.sh --epochs 1 --folds 0           # sanity
bash   experiments/mlp_baseline/train.sh                                 # 5 folds × L3_H2048 (~1 h)
```

`train.sh` dispatches to `train_default.py`, forwarding all arguments. Full
CLI: `python experiments/mlp_baseline/train_default.py --help`.

## Layers × hidden sweep

Grid sweep over the matrix
`{1,2,3,4} × {256,512,1024,2048,4096} = 20 configs × 5 folds = 100 runs`:

```bash
python experiments/mlp_baseline/sweep.py                                 # full sweep (~12 h)
python experiments/mlp_baseline/sweep.py \
    --layers_set 1 4 --hidden_set 256 4096 --folds 0 --epochs 1          # sanity (2 configs × 1 fold × 1 epoch)
python experiments/mlp_baseline/sweep.py --skip_default                  # skip the default config (already produced by train.sh)
```

Full CLI: `python experiments/mlp_baseline/sweep.py --help`.

## Output layout

```
results/mlp_baseline/
├── default/
│   └── fold_<0..4>/{checkpoints, predictions, metrics, logs}
└── sweep/
    └── L<layers>_H<hidden>/
        └── fold_<0..4>/{checkpoints, predictions, metrics, logs}
```
