# drug-perturbation-benchmark

Code for a benchmark of seven published deep-learning models for drug-induced
L1000 transcriptomic perturbation prediction. Bai *et al.*, manuscript in
preparation. Archived release with trained checkpoints + per-fold metrics on
Zenodo: <https://zenodo.org/records/20081274>.

The seven evaluated models — **CIGER, DeepCE, MultiDCP, PertDiT, PRnet,
TranSiGen, XPert** — are vendored at pinned upstream commits and adapted into a
unified training pipeline, together with a pure-MLP control baseline as the
eighth training target. All eight share one CLI shape and one conda environment
(`benchmark`).

## Repository layout

```
drug-perturbation-benchmark/
├── environment/             Conda env spec + pip install script
├── reference/
│   ├── <M>/                 Pristine upstream code @ pinned commit (populated by fetch_upstream.sh)
│   ├── fetch_upstream.sh    One-shot upstream code + dataset fetcher
│   └── PROVENANCE.md        Per-model URL, commit, fetch date, license
├── models/
│   ├── <M>/                 Vendored production code; imported by training shims via sys.path
│   └── <M>_CHANGES.md       Per-model audit: file SHA-256 vs upstream + line-level hunks
├── data/<M>/                Original datasets (fetched via reference/fetch_upstream.sh)
├── experiments/
│   ├── <task>/              Self-contained tasks: prepare.py + train_all.py + README.md
│   └── _shared/             Unified training shims + evaluation drivers + prepare-step library
├── results/                 Training outputs from reproduction scripts
└── LICENSE / NOTICE         MIT (top-level) + per-model license caveats (NOTICE)
```

## Tasks

| Directory | Description |
|---|---|
| `main_benchmark/`           | 7 vendored models on L1000 SDST under drug-blind cold-split |
| `main_ablation/`            | 7 vendored models × {zero, shuffle} drug-feature ablation, same split |
| `mlp_baseline/`             | MLP control baseline (L × H hyperparameter sweep; default at saturation point) |
| `scaffold_generalization/`  | 7 vendored models under scaffold-disjoint split |
| `original_reproduction/`    | 6 non-XPert models, each on its own paper's dataset |
| `original_ablation/`        | 6 non-XPert models, drug-zero ablation only, each on its own dataset |

XPert is excluded from `original_reproduction/` and `original_ablation/`
because its "original data" already is L1000. MLP appears only in
`mlp_baseline/`; its results are compared against the published-model
results from `main_benchmark/` separately.

## Setup

```bash
conda env create -f environment/environment.yml      # base layer
conda activate benchmark
bash environment/install_pip.sh                      # ML stack
```

`install_pip.sh` self-activates the `benchmark` env's pip. Install order
matters: torch precedes flash-attn, torch-geometric extensions, and
unimol-tools. Build size ~5.7 GB. See `environment/README.md` for version
pins and known footguns.

## Data

```bash
bash reference/fetch_upstream.sh    # idempotent; pulls all upstream repos + datasets
```

| Model | Upstream repo | Pinned commit | Dataset |
|---|---|---|---|
| CIGER     | [pth1993/CIGER](https://github.com/pth1993/CIGER)                                       | `81c16f1` | bundled in repo |
| DeepCE    | [pth1993/DeepCE](https://github.com/pth1993/DeepCE)                                     | `9b0d04f` | bundled in repo |
| MultiDCP  | [XieResearchGroup/MultiDCP](https://github.com/XieResearchGroup/MultiDCP)               | `36ecdef` | Zenodo [10.5281/zenodo.5172809](https://doi.org/10.5281/zenodo.5172809) (11 GB) |
| PertDiT   | [wangkekekeke/PertDiT](https://github.com/wangkekekeke/PertDiT)                         | `596d681` | Tsinghua Cloud (8.9 GB) |
| PRnet     | [Perturbation-Response-Prediction/PRnet](https://github.com/Perturbation-Response-Prediction/PRnet) | `f19174b` | Zenodo [10.5281/zenodo.14230870](https://doi.org/10.5281/zenodo.14230870) (4 GB) |
| TranSiGen | [myzhengSIMM/TranSiGen](https://github.com/myzhengSIMM/TranSiGen)                       | `8ec2218` | upstream Release v1.0 (~730 MB) + demo subset bundled in repo |
| XPert     | [GSanShui/XPert](https://github.com/GSanShui/XPert)                                     | `d53ff49` | Zenodo [10.5281/zenodo.15357711](https://doi.org/10.5281/zenodo.15357711) (14 GB) + Figshare 28955141 |

Per-model provenance is recorded in `reference/PROVENANCE.md`.

## Reproducing the analyses

Each task is independent. From the repo root:

```bash
python experiments/main_benchmark/prepare.py
python experiments/main_benchmark/train_all.py        # paper defaults: all models, 5 folds
```

For a one-epoch sanity run on a single model:

```bash
python experiments/main_benchmark/train_all.py --models xpert --folds 0 --epochs 1
```

Outputs land under `results/<task>/<model>/<fold>/{checkpoints, predictions, metrics, logs}/`.
Per-task specifics (data preparation, ablation modes, runtime estimates) are
documented in each `experiments/<task>/README.md`.

## Code modifications

Each vendored model has an audit file `models/<M>_CHANGES.md` documenting any
deviations from its pinned upstream commit, with per-file checksums and an
exact diff command reviewers can run against the pristine clones populated by
`reference/fetch_upstream.sh`.

## License

MIT, see [LICENSE](LICENSE). Vendored models retain their upstream licenses,
which may differ from MIT (notably MultiDCP under CC BY-NC; CIGER and DeepCE
without an upstream LICENSE file). See [NOTICE](NOTICE) for the per-model
breakdown and caveats.

## Citation

Bai, J. *et al.* (manuscript in preparation). Until publication, please
reference this repository by its GitHub URL and commit hash, together with
the archived release on Zenodo: <https://zenodo.org/records/20081274>.
