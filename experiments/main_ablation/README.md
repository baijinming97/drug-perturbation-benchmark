# main_ablation

Drug-feature ablation × cold-drug 5-fold benchmark. Each model is trained on
the same cold-drug splits as `main_benchmark/`, but with its drug input
perturbed, to test whether predictions are genuinely driven by drug structure.
Two ablation modes:

| Mode      | What it does                                                   |
|-----------|----------------------------------------------------------------|
| `zero`    | Replace each drug's input features with zeros                  |
| `shuffle` | Randomly permute drug identity across the dataset              |

## Quick start

```bash
conda activate benchmark
python experiments/main_ablation/prepare.py
bash   experiments/main_ablation/train.sh --epochs 1 --folds 0          # 14-job sanity (~1 h)
bash   experiments/main_ablation/train.sh                                # 7 models × 2 ablations × 5 folds = 70 runs (~40 h)
bash   experiments/main_ablation/train.sh \
    --models xpert --ablation_modes zero --folds 0 1 2 3 4               # single model, single ablation
```

Full CLI: `python experiments/main_ablation/train_all.py --help`.

## Output layout

```
results/main_ablation/
└── <model>/
    ├── zero/fold_<0..4>/{checkpoints, predictions, metrics, logs}
    └── shuffle/fold_<0..4>/{checkpoints, predictions, metrics, logs}
```

Each `fold_<N>/` mirrors `main_benchmark/` exactly, so downstream evaluation
scripts can be reused by simply pointing `--results_dir` here.
