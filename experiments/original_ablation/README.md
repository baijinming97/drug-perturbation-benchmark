# original_ablation

Drug-zero ablation on each model's own original-paper dataset — parallel to
`main_ablation/`, but using each model's native data instead of the unified
L1000_sdst. Two conditions:

| Mode   | What it does                                                  |
|--------|---------------------------------------------------------------|
| `none` | No-ablation control                                           |
| `zero` | Replace each drug's input features with zeros                 |

6 vendored models (CIGER, DeepCE, MultiDCP, PertDiT, PRnet, TranSiGen).
XPert is not in this driver, for the same reason as in
`original_reproduction/`: XPert's "original" dataset is L1000_sdst.

## Quick start

```bash
conda activate benchmark
python experiments/original_ablation/prepare.py
bash   experiments/original_ablation/train.sh --epochs 1 --folds 0     # 12-job sanity
bash   experiments/original_ablation/train.sh                           # 6 models × 2 conditions × 5 folds = 60 runs
bash   experiments/original_ablation/train.sh \
    --models prnet --ablation_modes zero --folds 0 1 2 3 4              # single model, single ablation
```

Full CLI: `python experiments/original_ablation/train_all.py --help`.

## Output layout

```
results/original_ablation/
└── <model>/
    ├── none/fold_<0..4>/{checkpoints, predictions, metrics, logs}
    └── zero/fold_<0..4>/{checkpoints, predictions, metrics, logs}
```
