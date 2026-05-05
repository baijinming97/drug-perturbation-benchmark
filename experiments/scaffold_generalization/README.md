# scaffold_generalization

Same L1000_sdst data as `main_benchmark/`, but with Bemis-Murcko
scaffold-disjoint splits — tests generalization to drug scaffolds unseen in
training, with drug-zero ablation overlaid to test whether models still rely
on drug structure under the harder split.

| Mode   | What it does                                                  |
|--------|---------------------------------------------------------------|
| `none` | Standard scaffold-disjoint benchmark (no ablation)            |
| `zero` | Replace each drug's input features with zeros                 |

## Quick start

```bash
conda activate benchmark
python experiments/scaffold_generalization/prepare.py
bash   experiments/scaffold_generalization/train.sh --epochs 1 --folds 0   # 14-job sanity (~1 h)
bash   experiments/scaffold_generalization/train.sh                         # 7 models × 2 ablations × 5 folds = 70 runs (~40 h)
bash   experiments/scaffold_generalization/train.sh \
    --models xpert --ablation_modes zero --folds 0 1 2 3 4                  # single model, single ablation
```

Full CLI: `python experiments/scaffold_generalization/train_all.py --help`.

## Output layout

```
results/scaffold_generalization/
└── <model>/
    ├── none/fold_<0..4>/{checkpoints, predictions, metrics, logs}
    └── zero/fold_<0..4>/{checkpoints, predictions, metrics, logs}
```
