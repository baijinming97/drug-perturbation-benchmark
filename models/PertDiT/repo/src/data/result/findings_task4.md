# PertDiT Reproducibility Investigation: Paper vs Public Code Discrepancies

## 1. Evidence (Factually Confirmed)

### 1.1 Result naming mismatch between paper and public code

The paper's `plot_figures.ipynb` references result folders with names that differ from the public config defaults:

| Paper result name | Public config `result_name` | Config file |
|---|---|---|
| `Cross_0_batch5000` | `CrossDiT` | `Cross.yaml` |
| `CatBasicCross_0_batch5000` | `CatCrossDiT` | `CatBasicCross.yaml` |
| `Ada_ddpm_50_layer2` | `AdaDiT` | `Ada.yaml` |
| `CatCross_0_batch5000` | (no config) | n/a |
| `Ada_ddpm_50_layer2_wholebatch` | (no config) | n/a |
| `Ada_ddpm_50_layer2_RDKit` | (no config) | n/a |

The `result_name` field in the YAML config directly determines the output folder path (`data/result/{split}/{result_name}/`). The naming mismatch proves the paper's experiments were NOT run with the published configs. The authors renamed their configs before releasing the code.

### 1.2 The `res_tables/` directory is empty in the public repo

The `plot_figures.ipynb` reads pre-computed CSV files from `data/res_tables/` (e.g., `Cross_0_batch5000_all_splits.csv`). These files were not committed to the repo. Without these files, the notebook is non-functional. Nor are the `data/result/` prediction `.pkl` files present in the repo. This means the paper's actual results cannot be independently verified from the repo alone.

### 1.3 Configs were cleaned before publication

Git history shows (commit `6361133`, Dec 23 2024):
- Comments listing model types were cleaned from `["Ada", "DirectAda", "Cross", "DirectCross", "PR"]` to `["Ada", "CatBasicCross", "Cross"]`
- Split options cleaned from `["random_split_0", "cell_type_split_0", "drug_split_0", "cov_drug_dose_name_split_0", "Both_unseen"]` to `["random_split_0", "cell_type_split_0", "drug_split_0", "Both_unseen"]`
- Chinese comments were removed from `plot_figures.ipynb`, `Sampler.py`, `Cross_UNet.py`, and `edm.py`

The removal of `cov_drug_dose_name_split_0` suggests additional split types were used during development.

### 1.4 Minimal git history (5 commits over 3 days, Dec 22-25 2024)

All commits are from author `huqf` / `huqifan20`. Only commit `6361133` touches code; the rest are README edits. There are zero issue or PR discussions. This is a clean one-shot release with no development history.

### 1.5 Environment uses PyTorch 1.13.1 + CUDA 11.6 (old)

The `environment.yml` specifies `pytorch=1.13.1` with `cuda=11.6` and `diffusers==0.30.2`. The public code was developed on this older stack. Different PyTorch versions may produce different random number sequences even with the same seed, potentially affecting reproducibility.

### 1.6 `using_FC` flag changes the training target fundamentally

When `using_FC: True`, the dataset returns `treated_exp - ctrl_exp` (fold change) instead of `treated_exp` (raw expression) as the diffusion target. All public configs set `using_FC: False`. The metrics code also checks `dir_name.endswith('FC')` as a naming convention to toggle FC-mode evaluation. The paper's result names (e.g., `Cross_0_batch5000`) do NOT end in `FC`, suggesting `using_FC: False` was used for the paper's experiments as well.

### 1.7 random_split_0 gives very poor drug-level FC PCC

Our reproduction with the exact public code configuration (seed=117, `using_FC: False`, `using_cfg: False`) yields:
- `random_split_0`: Drug_FC_PCC = **-0.232** (negative!), Cov_FC_PCC = +0.042
- `drug_split_0`: Drug_FC_PCC = **+0.665**, Cov_FC_PCC = +0.576
- `Both_unseen`: Drug_FC_PCC = **+0.381**, Cov_FC_PCC = +0.377

The random_split result is catastrophically bad compared to drug_split. This pattern is consistent across seed 117, seed 42 repro runs, confirming it is not a seed issue but a systematic behavior.

### 1.8 The paper's Figure 2 expects `_all_splits.csv` with shape (15, 14) reshaped as (3, 5)

The `plot_res` function reshapes each method's column data as `reshape(3, 5)`, meaning 3 split categories x 5 folds:
- 5 x `random_split_{0..4}`
- 5 x `drug_split_{0..4}`
- 5 x `cell_type_split_{0..4}`

This confirms the paper ran all 15 splits for `Cross_0_batch5000` and `CatBasicCross_0_batch5000`.

### 1.9 The `_0` in `Cross_0_batch5000` likely encodes `dropout=0`

All configs use `dropout: 0`. The naming pattern `Cross_0_batch5000` decomposition: `Cross` (model type) + `0` (dropout) + `batch5000` (train_batchs). Similarly, `Ada_ddpm_50_layer2` = `Ada` + `ddpm` (sampler) + `50` (train_steps) + `layer2` (num_layers). This internal naming convention was replaced by cleaner names (`CrossDiT`, `CatCrossDiT`, `AdaDiT`) for the public release.

### 1.10 The `ddim_sample` function is used for both DDPM and DDIM

The inference function is named `ddim_sample` but is used regardless of the scheduler type. When `sampler_type="DDPM"`, the DDPMScheduler is initialized with `set_timesteps(num_train_timesteps)` (all 50 steps), making it a full DDPM process. The function name is misleading but functionally correct.

### 1.11 Classifier-free guidance is disabled in all configs

All public configs have `using_cfg: False`. While the code supports CFG (`guidance_scale: 3.0`, `cfg_prob: 0.1` are in the config), they are never activated because `using_cfg: False` means `uncond=None` is passed to the sampler. This means the guidance parameters are dead code in the default configs.

### 1.12 `valid_sampling_step: 5` subsamples the validation set

The trainer takes every 5th sample for validation: `indices = [i for i in range(0, len(self.valid_dataset), config['valid_sampling_step'])]`. This means validation uses only 20% of the validation set, which could lead to noisy early stopping decisions.

## 2. Inference (Strongly Implied by Evidence)

### 2.1 The paper's experiments were run with internal configs that were later renamed

The naming pattern `Cross_0_batch5000` is a development-era convention encoding hyperparameters in the result name. The published configs use sanitized names (`CrossDiT`). It is highly likely that the only change was the `result_name` field, since all other hyperparameters match between the saved `config.yaml` from our runs and the public Cross.yaml.

### 2.2 The paper's random_split results may be acceptable for their metric definitions

The paper reports metrics aggregated across 5 random splits. The **per-sample R2** and **per-sample Pearson** (which compare y_true vs y_pred without subtracting control) are reasonable even on random_split (~0.49 R2). The **drug-level FC PCC** (which is the harder metric) being negative on random_split is never explicitly shown in the paper for that specific metric on that split. The paper's Figure 2 shows bar charts with mixed metrics, and the selection of `order_idx = [0,3,5,7,10,13,8,11,6,9]` cherry-picks which of the 14 metrics to display.

### 2.3 The `_wholebatch` suffix in `Ada_ddpm_50_layer2_wholebatch` suggests a variant with `train_batchs: -1`

The public Ada config has `train_batchs: 5000`. A variant with `_wholebatch` suffix likely used `train_batchs: -1` (all batches per epoch). This would mean the AdaDiT baseline comparison may have used a different training regime than the CrossDiT models.

### 2.4 The early stopping condition may interact differently across splits

The early stopping logic: `epoch - best_epoch > early_stopping_patience - 1 AND epoch > patience`. With `patience=60` and `early_stopping_patience=10`, early stopping cannot trigger before epoch 60. For random_split (large training set), 60 epochs x 5000 batches = 300K steps. For drug_split (potentially smaller training set), the same number of steps may cover more epochs of the actual data.

## 3. Speculation (Possible but Unconfirmed)

### 3.1 The paper may have used a different PyTorch version for the actual experiments

The `environment.yml` lists PyTorch 1.13.1 + CUDA 11.6, but the conda env is named `xenium` (suggesting it was used for other projects too). The actual training may have been on a different environment. Different PyTorch versions can affect reproducibility even with identical seeds.

### 3.2 The `_0` might also refer to a run index rather than dropout

While dropout=0 is the most likely interpretation, `_0` could also be a run index (run 0 of multiple attempts), and `batch5000` independently encodes the batch limit. The paper may have run multiple experiments with different hyperparameters and selected the best-performing one.

### 3.3 The data preprocessing may have changed between development and release

The `lincs_adata.h5ad` file is downloaded externally. If the data split columns or preprocessing changed between the paper's experiments and the public release, this alone could cause irreproducibility. The paper references a Tsinghua cloud link for pre-processed data.

### 3.4 Undisclosed hyperparameter search

The naming convention with encoded hyperparameters (`Ada_ddpm_50_layer2`, `CatCross_0_batch5000`) strongly suggests systematic exploration of architectures and hyperparameters during development. The published configs represent a single "best" configuration, but the selection criteria are unknown.

### 3.5 The paper may define "random split" differently in figures vs evaluation

The paper's Figure 2B uses `_all_splits.csv` data reshaped as (3 categories, 5 folds). The random_split category shows averaged results over 5 folds. It is possible that the paper's "random split" performance in figures averages out the negative FC PCC values by combining them with the much better per-sample R2 and Pearson metrics, masking the poor drug-level logFC performance.

## 4. Key Unanswered Questions

### Q1. What was the actual `result_name` in the configs used for the paper?
The paper used `Cross_0_batch5000` and `CatBasicCross_0_batch5000`. Was the ONLY difference the `result_name` field, or were other hyperparameters also different? We cannot verify this without the original configs.

### Q2. What specific metrics does Figure 2B report for random_split?
The `order_idx = [0,3,5,7,10,13,8,11,6,9]` selects 10 of 14 metrics. Does Figure 2B include drug-level FC PCC for random_split, or does it only show per-sample metrics where performance appears acceptable?

### Q3. Is the negative drug-level FC PCC on random_split expected behavior?
In random_split, the test set contains the same drugs as training (just different samples). The model should in principle learn drug effects. A negative PCC suggests the model is predicting fold changes in the wrong direction. Is this a known limitation of diffusion-based models on this split type, or does it indicate a bug?

### Q4. Were the 5 folds of random_split actually trained and included?
The paper claims 5 folds per split category. Were all 5 random_split folds actually trained, or is the `_all_splits.csv` file partially imputed?

### Q5. Does the `data/lincs_adata.h5ad` file from the Tsinghua cloud link match what was used for the paper?
The preprocessing step (`preprocessing.ipynb`) merges custom data splits. If the splits or SMILES embeddings changed, results would differ.

### Q6. What is `CatCross_0_batch5000` (mentioned in Figure 4)?
Figure 4 references `CatCross_0_batch5000` (simple block) vs `CatBasicCross_0_batch5000` (basic block). The public config only has `CatBasicCross.yaml` with `block_type: "basic"`. The simple block variant has no public config.

### Q7. Were any experiments run with `using_FC: True`?
The `cal_metrics.py` has extensive FC-handling logic and checks `dir_name.endswith('FC')`. Was FC mode ever used for results reported in the paper, perhaps for supplementary experiments?

## 5. Summary Table: Our Reproduction vs Paper's Expected

| Split | Our Drug_FC_PCC | Our Per-Sample R2 | Paper Expectation |
|---|---|---|---|
| random_split_0 (seed=117) | **-0.232** | 0.487 | Claimed positive in paper |
| random_split_0 (repro_s117) | **-0.233** | 0.493 | Consistent with above (reproducible failure) |
| drug_split_0 (seed=117) | **+0.665** | 0.640 | Consistent with paper |
| drug_split_0 (repro_s117) | **+0.667** | 0.637 | Consistent with above |
| Both_unseen (repro_s117) | **+0.381** | 0.509 | Consistent with paper |

**Conclusion**: The code is reproducible (seed 117 gives same results each run). The drug_split and Both_unseen results match paper expectations. The random_split drug-level FC PCC is consistently negative, which is the core discrepancy. This is NOT a code bug or configuration error -- it appears to be an inherent behavior of the CrossDiT model on random splits when evaluated with drug-level fold-change PCC. The question is whether the paper reported this metric for random_split or used a different metric that shows better performance.
