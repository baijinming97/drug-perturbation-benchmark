# XPert — Modification Audit

**Upstream**: `https://github.com/GSanShui/XPert` @ commit `d53ff497e465692c80fac7b899e3cc5cc8e7bfed`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature |
|---|---|---|
| `train_xpert.py` | 5 | CLI output-directory parameterization + checkpoint device mapping + 1 bug fix |
| `configs/config_*.yaml` (5 inherited from upstream) | 3-11 lines each | Two path adjustments: (a) `HG_data/` → `reference/XPert/HG_data/` (2 lines/file); (b) `processed_data/` → `data/XPert/processed_data/` (1-9 lines/file) |
| **Unchanged `.py`** | — | 13 files, SHA-256 byte-equivalent (§4) |
| **Added** | — | 9 ablation YAMLs (`configs/config_l1000_ablation_*.yaml`) |

---

## 2. Line-level Hunks

### 2.1 `train_xpert.py` (5 hunks)

#### Hunk 1: CLI adds `--expt_dir` parameter

```diff
@@ -54,7 +54,8 @@
     parser.add_argument('--weighted_loss', type=bool, default=False)
     parser.add_argument('--kl_loss', type=bool, default=False)
-   
+    parser.add_argument('--expt_dir', type=str, default=None,
+                        help='Override experiment output directory (for parallel jobs)')
+
     return parser.parse_args()
```

- **Necessity**: Allows custom output directories for concurrent training jobs, avoiding timestamp collisions.
- **Behavior preservation**: `default=None` — when not explicitly passed, the new branch (Hunk 3) is not entered; strictly equivalent to upstream.
- **Performance impact**: None.

#### Hunk 2: Empty-guard added to the `output_profile` branch

```diff
@@ -223,7 +224,7 @@
         metrics = get_metrics_new(y_true, y_pred, ctl_true)
-        if args.output_profile:
+        if args.output_profile and expt_folder is not None:
             print('Saving output_profile....')
             output_profile_dict = {}
             output_profile_dict['y_true'] = y_true
```

- **Necessity + actual behavior**: This is not a pure equivalence-preserving modification but **a defensive output fix that takes effect on the production path**. The dpb driver passes `--output_profile True` in all phases. On the `mode=train` production main path, after training completes, `validate(..., flag=True)` is called for train, validation, and test:
  - **train** (L570) and **validation** (L575) — **do not pass `expt_folder`** (defaults to `None`)
  - **test** (L580) — **passes `expt_folder=expt_folder, fold=k`**
  
  Additionally, the `mode=test` branch (L632) has its own independent test validate call, also passing `expt_folder` and `fold`.
  
  The upstream version lacks the `expt_folder is not None` guard and would attempt to write profiles to `None/...` paths during train/validation evaluation (causing save failures). With the guard added, **train/validation profile is not saved, test profile is still saved**.
- **Behavior difference**: This hunk changes upstream's "always attempt to save profile regardless of expt_folder" behavior, but **does not affect test metrics computation** (only affects whether train/validation profile files land on disk). Defensive output fix.
- **Performance impact**: None (just one extra None comparison).

#### Hunk 3: `if args.expt_dir:` branch

```diff
@@ -369,7 +370,13 @@
     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
 
-    if args.mode == 'infer':
+    if args.expt_dir:
+        expt_folder = args.expt_dir
+        if os.path.exists(expt_folder) and any(
+            f.endswith(('.pth', '.log', '.csv', '.npy')) for f in os.listdir(expt_folder)
+        ):
+            expt_folder = osp.join(args.expt_dir, f'{timestamp}')
+    elif args.mode == 'infer':
         expt_folder = osp.join('experiment/infer/', f'{timestamp}')
     else:
         if 'l1000' in args.dataset:
```

- **Necessity**: The parameter introduced in Hunk 1 takes effect here. If the target directory already contains training artifacts, a timestamped sub-directory is created underneath to avoid overwriting.
- **Behavior preservation**: With `args.expt_dir` defaulting to `None`, control falls through to the original `elif args.mode == 'infer':` branch; strictly equivalent to upstream.
- **Performance impact**: None.

#### Hunk 4: Test call uses `best_model` + bug fix

```diff
@@ -570,8 +577,7 @@
             logger.info('Tesing on test dataset:')
-            _,_,_,_,_, test_metrics = validate(model, test_dataloader, args, config, flag=True, scaler=gradscaler)
-            test_metrics = val_metrics
+            _,_,_,_,_, test_metrics = validate(best_model, test_dataloader, args, config, flag=True, scaler=gradscaler, expt_folder=expt_folder, fold=k)
             logger.info('Test Metrics:')
             log_nested_dict(test_metrics, logger, indent=1)          
```

- **Necessity**: This hunk does two things:
  - **(a) BUG FIX**: The upstream second line `test_metrics = val_metrics` immediately overwrites the validate result on the first line, **causing test metrics to always equal validation metrics** (a clear bug). dpb removes this line.
  - **(b) Switch to `best_model`** instead of `model`: Upstream evaluates the test set with the last-epoch model; dpb uses the early-stopping best model (more aligned with standard ML practice).
- **Behavior preservation**: **This is a genuine behavior change** — test metrics are computed differently from upstream. But the upstream code is clearly buggy (test result = validation result); the fixed version is the intended behavior.
- **Performance impact**: Test metrics become meaningful. **All P1–P7 evaluations on the production path are based on this fixed version**; the upstream-buggy version's reported "test metrics" were actually validation metrics.

#### Hunk 5: Checkpoint loading supports device mapping

```diff
@@ -610,7 +616,7 @@
                 saved_model = osp.join(args.saved_model_path, '{}_fold_early_stop.pth'.format(k))
             else:
                 logger.info('No trained parameters are provided!')
-            checkpoint = torch.load(saved_model)
+            checkpoint = torch.load(saved_model, map_location=device)
             model.load_state_dict(checkpoint['model_state_dict'])

@@ -672,7 +678,7 @@
             else:
                 logger.info('No trained parameters are provided!')
             
-            checkpoint = torch.load(saved_model)
+            checkpoint = torch.load(saved_model, map_location=device)
```

- **Necessity**: Loading a CPU-saved checkpoint on GPU (or vice versa) requires explicit `map_location`; upstream would otherwise crash.
- **Behavior preservation**: Strictly equivalent on the same device.
- **Performance impact**: None.

### 2.2 `configs/config_*.yaml` (5 inherited files, two path adjustments per file)

Files affected: `config_cdsdb.yaml`, `config_l1000.yaml`, `config_l1000_cdt.yaml`, `config_l1000_example.yaml`, `config_panacea.yaml`. Each file has two independent path adjustments:

#### (a) `HG_data/` → `reference/XPert/HG_data/` (2 lines per file)

```diff
   HG:
-    hg_path: HG_data/
-    drug_hg_pretrained_embed_path: HG_data/saved_embedding/HG_drug_embeddings.npy
+    hg_path: reference/XPert/HG_data/
+    drug_hg_pretrained_embed_path: reference/XPert/HG_data/saved_embedding/HG_drug_embeddings.npy
```

- **Necessity**: Upstream XPert bundles the HG drug embeddings used for training in the repository-root `HG_data/` (relative path), depending on the training script's cwd being the repository root. This directory vendors XPert under `models/XPert/`; if `HG_data/` were kept as the path, it would not resolve at runtime. Changed to `reference/XPert/HG_data/` (populated by `fetch_upstream.sh XPert`).
- **Behavior preservation**: **Data-equivalent** — `reference/XPert/HG_data/` is cloned by the fetch script directly from the upstream commit; contents are byte-equivalent to upstream's bundled `HG_data/`.
- **Performance impact**: None.

#### (b) `processed_data/` → `data/XPert/processed_data/` (1-9 lines per file)

```diff
-    ppi_gene_vector_path: processed_data/PPI_gene_vector_128d.npy
-  l1000_sdst_data_root: processed_data/l1000_sdst_78453.h5ad
-  l1000_mdmt_data_root: processed_data/l1000_mdmt_68830_subset.h5ad
-  ...
+    ppi_gene_vector_path: data/XPert/processed_data/PPI_gene_vector_128d.npy
+  l1000_sdst_data_root: data/XPert/processed_data/l1000_sdst_78453.h5ad
+  l1000_mdmt_data_root: data/XPert/processed_data/l1000_mdmt_68830_subset.h5ad
+  ...
```

(Number of changed lines per file varies with how many `processed_data/...` paths the YAML declares: 1 in `config_l1000_example.yaml`, 6 in `config_cdsdb.yaml`/`config_panacea.yaml`, 9 in `config_l1000.yaml`/`config_l1000_cdt.yaml`.)

- **Necessity**: The upstream `processed_data/` directory is staged by the original XPert author at the repository root and contains data delivered separately (mostly via Figshare). In dpb the same files are placed under `data/XPert/processed_data/`, alongside the source archive `data/XPert/processed_data.zip` from which they are extracted (see `experiments/_shared/prepare_steps.py:step_extract_processed_data` and `step_extract_xpert_unimol_arr`). Updating the YAMLs to the new location keeps configuration internally consistent and lets `experiments/_shared/training/train_xpert.py` (driver shim) absolutize paths via `REPO_ROOT / <path>` without special-casing `processed_data/`.
- **Behavior preservation**: **Data-equivalent** — same files, same contents; only the on-disk location is reorganized.
- **Performance impact**: None.

> **Note**: The 9 added ablation YAMLs (§3) carry the same `data/XPert/processed_data/` prefix throughout, with no `processed_data/` legacy paths.

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `HG_data/` (8.8 MB) | Data | Upstream-bundled HG drug embeddings; **provided instead by `fetch_upstream.sh XPert` at `reference/XPert/HG_data/`**; YAML configs point directly to this path (§2.2) |
| `experiment/` | Demo | Upstream demo placeholder directory |
| `processed_data/` | Demo | Upstream demo data |
| `reproducing/` | Demo | Upstream reproducing demo scripts |
| `saved_model/` | Demo | Upstream demo checkpoints |

### Local-only (not in upstream)

| File | Category | Notes |
|---|---|---|
| `configs/config_l1000_ablation_zero_all.yaml` | Ablation config | All drug structural information zeroed |
| `configs/config_l1000_ablation_zero_atoms.yaml` | Ablation config | Only atom-level embeddings zeroed |
| `configs/config_l1000_ablation_zero_unimol.yaml` | Ablation config | UniMol embedding zeroed |
| `configs/config_l1000_ablation_zero_hg.yaml` | Ablation config | HG embedding zeroed |
| `configs/config_l1000_ablation_zero_hg_shuffle_unimol.yaml` | Ablation config | HG zeroed + UniMol shuffled |
| `configs/config_l1000_ablation_shuffle.yaml` | Ablation config | Drug-identity shuffle (basic) |
| `configs/config_l1000_ablation_shuffle_all.yaml` | Ablation config | All drug features shuffled |
| `configs/config_l1000_ablation_shuffle_hg.yaml` | Ablation config | HG embedding shuffled |
| `configs/config_l1000_ablation_shuffle_hg_zero_unimol.yaml` | Ablation config | HG shuffled + UniMol zeroed |

All 9 ablation YAMLs are derived from upstream's `config_l1000.yaml`, with changes confined to:
- `model.HG.drug_hg_pretrained_embed_path` (pointing to `experiment/ablation_infer/data/HG_drug_embeddings_{zero,shuffle}.npy`, generated before runtime by `prepare_steps.step_generate_xpert_ablation_data`)
- `model.HG.hg_path` (same as §2.2, changed to `reference/XPert/HG_data/`)
- Some files also modify the UniMol embedding path with the same logic

These YAMLs **do not modify any model architecture hyperparameter**. The substantive changes are concentrated on drug-input array paths (HG and UniMol zero/shuffle versions); additional differences include comments, whitespace, and `dataset` config blocks not used by the current `l1000_sdst` path — these do not affect the configuration used by this experiment.

### LICENSE status

✅ `models/XPert/LICENSE` (MIT, Copyright 2025 Guo Yue) is byte-equivalent to upstream (copied verbatim from `reference/XPert/LICENSE`).

---

## 4. SHA-256 Verification

### 4.1 Modified files

| File | Upstream SHA (first 16) | This dir SHA (first 16) |
|---|---|---|
| `train_xpert.py` | `a4c215e5a7eab832` | `eaf910eecd8ab5ff` |

### 4.2 Unchanged `.py` (13 files, byte-equivalent)

| File | SHA-256 (first 16) |
|---|---|
| `datasets/MyDataset.py` | `d4cb1ed1d293d66d` |
| `evaluation_metrics/generate_metrics_cdsdb.py` | `803ee92c3f674362` |
| `evaluation_metrics/generate_metrics_cdt.py` | `6bc4f517ab470961` |
| `evaluation_metrics/generate_metrics_mdmt.py` | `c48b958902e72a03` |
| `evaluation_metrics/generate_metrics_panacea.py` | `410b73e38c768183` |
| `evaluation_metrics/generate_metrics_sdst.py` | `9dd1babd56fa68fc` |
| `evaluation_metrics/get_evaluation_metrics.py` | `8b3e208d819a189b` |
| `metrics.py` | `fef622ced87dddda` |
| `models/model_XPert.py` | `87b08745515f7281` |
| `models/model_utils.py` | `34ae71f4cd1a231f` |
| `pretrain_hg.py` | `fd492b2a3e806e1f` |
| `utils.py` | `da15f152943a602b` |
| `utils_hg.py` | `5f957fea23e9a4fa` |

### 4.3 Upstream YAMLs modified (5 files, both `HG_data/` and `processed_data/` adjustments — see §2.2)

| File | Lines changed | Upstream SHA (first 16) | This dir SHA (first 16) |
|---|---|---|---|
| `configs/config_cdsdb.yaml` | 8 | `8cdc096c5418731c` | `2372580a66154aaa` |
| `configs/config_l1000.yaml` | 11 | `94a2206ced6e679e` | `5734d0f08f44047b` |
| `configs/config_l1000_cdt.yaml` | 11 | `0d1840c4190f194d` | `386140535b304d60` |
| `configs/config_l1000_example.yaml` | 3 | `d1aca6f6491b5f2f` | `375627cdf2ed517a` |
| `configs/config_panacea.yaml` | 8 | `3d591d53ad19c092` | `ba69b43cfdceb269` |

### 4.4 Added ablation YAMLs (9 files; no upstream counterpart)

| File | This dir SHA (first 16) |
|---|---|
| `configs/config_l1000_ablation_zero_all.yaml` | `2d0d06f6b0018a4e` |
| `configs/config_l1000_ablation_zero_atoms.yaml` | `d1049a9f1a42925f` |
| `configs/config_l1000_ablation_zero_unimol.yaml` | `21e562a01af88942` |
| `configs/config_l1000_ablation_zero_hg.yaml` | `e11b84a146a6ccf9` |
| `configs/config_l1000_ablation_zero_hg_shuffle_unimol.yaml` | `abdc3e7b6da82ad3` |
| `configs/config_l1000_ablation_shuffle.yaml` | `aebe498ad2263056` |
| `configs/config_l1000_ablation_shuffle_all.yaml` | `7be37395f2a71d19` |
| `configs/config_l1000_ablation_shuffle_hg.yaml` | `9aed9924467fa566` |
| `configs/config_l1000_ablation_shuffle_hg_zero_unimol.yaml` | `0485f6997e3def21` |

### 4.5 `scripts/` directory (8 `.sh` files, byte-equivalent)

| File |
|---|
| `infer.sh`, `pretrain.sh`, `pretrain_hg.sh`, `test.sh`, `train.sh`, `train_cdsdb.sh`, `train_example.sh`, `train_panacea.sh` |

All 8 byte-equivalent to upstream.

---

## 5. Driver Behavior Summary

XPert differs from the other 6 models in driver architecture: **the driver is a shim**:

```python
# experiments/_shared/training/train_xpert.py (driver shim)
# 1. Select YAML based on --ablation_mode (none / zero / shuffle)
# 2. Absolutize relative paths inside the YAML (absolutize_paths)
# 3. subprocess.run("python train_xpert.py --config <temp> ...")
#    cwd=models/XPert/, directly running the train_xpert.py in this directory
```

| Item | Status |
|---|---|
| Does the driver import XPert source? | ❌ No |
| How does the driver run? | `subprocess.run([sys.executable, "train_xpert.py", "--config", tmp_yaml, ...])` |
| Are the 5 hunks in `train_xpert.py` actually executed? | Partially (see §5.1): Hunks 1–3 take effect on the train main path; Hunk 4 takes effect at the test evaluation at the end of train mode; Hunk 5 only executes under `mode=test` / `mode=infer` (not under default `mode=train`) |

### 5.1 Effect of the 5 hunks on the production path

| Hunk | Effective status | Default-behavior equivalence to upstream |
|---|---|---|
| 1 (`--expt_dir` parameter definition) | Effective | ✓ default `None`; equivalent when not passed |
| 2 (empty-guard) | Effective | **Changes train/val profile saving behavior**: driver does not pass `expt_folder` to tr/val evaluation → guard prevents writing to `None/...`; test call passes `expt_folder` → still saves. **Does not affect test metrics computation** |
| 3 (`if args.expt_dir:` branch) | Effective | ✓ default `None` → falls through to original `elif/else` |
| 4 (test uses best_model + buggy line removed) | Effective | **Behavior change** (upstream is buggy: `test_metrics = val_metrics`) — this is a bug fix |
| 5 (`map_location=device`) | **Only executed in `mode=test` / `mode=infer` branches** | ✓ Equivalent on same device; default production `mode=train` does not enter this branch |

### 5.2 On Hunk 4

Hunk 4 is **the hunk that directly changes test-metrics computation results**; it does two things:

1. **Removes `test_metrics = val_metrics`**: In upstream, regardless of what `validate(...)` returns, the immediately following assignment overwrites it, causing test metrics to always equal validation metrics — an upstream bug.
2. **Replaces `model` with `best_model`**: Switches the test model from the last epoch to the early-stopping best model (more aligned with standard ML practice).

**The XPert test-set metrics reported in the paper are based on this fixed version** (test using best_model + the actual test-set validate output). Any difference from the numbers in the upstream paper may partly stem from this bug fix.

### 5.3 Operational mode of the 9 added ablation YAMLs

`prepare_steps.step_generate_xpert_ablation_data` generates the zero/shuffle arrays under `data/_xpert_ablation/` during the prepare stage; these YAMLs reference those arrays via `drug_hg_pretrained_embed_path`. The XPert source itself requires no modification to support the ablation path — ablation is achieved via YAML config + pre-generated input data arrays.
