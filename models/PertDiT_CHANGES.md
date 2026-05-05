# PertDiT — Modification Audit

**Upstream**: `https://github.com/wangkekekeke/PertDiT` @ commit `596d681f816d3184d7a0a63a133ce68d5838fae0`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature | Active on which path? |
|---|---|---|---|
| `src/model/AdaDit.py` | 1 | Adds `ablation_mode` hook | Only the Ada config (driver uses CrossDit, does not enter) |
| `src/train_split.py` | Multiple | CLI ablation parameters + removal of hardcoded split test loop | ❌ DEAD (driver does not use train_split.py) |
| `src/trainer/Trainer.py` | Multiple | `ablation_mode` kwarg + `using_FC` support + FC PCC metric correction | ❌ DEAD (driver does not import; `run_repro.py` cannot run independently because Cross.yaml is not vendored) |
| `src/utils/utils.py` | 1 | torchmetrics `R2Score` compatibility fallback | ❌ DEAD (driver does not import; `eval_*.py` research scripts are not on automation paths) |
| **Unchanged `.py`** | — | 25 files, SHA byte-equivalent (§4) |
| **Added** | — | `src/bench_dataset.py` (data adapter, used by driver), 3 research scripts (`eval_checkpoint_sweep.py`, `eval_conditioning_ablation.py`, `run_repro.py`) |

**Production path (paper P1–P7 main results)** = dpb driver shim `train_pertdit.py` → directly imports `model.CrossDit.Crossformer` + dpb-added `bench_dataset.BenchPertDiTDataset`. **All 4 modified upstream files are off this path**.

---

## 2. Line-level Hunks

### 2.1 `src/model/AdaDit.py` (1 hunk) — DEAD on the default path

```diff
@@ -93,6 +93,12 @@
         time_embedding = self.time_encoder(time)
         x = self.pre_proj(x)
         cond = self.cond_encoder(cond)
+        # Drug feature ablation
+        if hasattr(self, 'ablation_mode') and self.ablation_mode != 'none':
+            if self.ablation_mode == 'zero':
+                cond = torch.zeros_like(cond)
+            elif self.ablation_mode == 'shuffle':
+                cond = cond[torch.randperm(cond.size(0))]
         x = x + time_embedding + self.positional_encoding.unsqueeze(0)
         cond = cond + time_embedding.squeeze(1)
```

- **Necessity**: Implements drug-feature ablation (zero / shuffle) inside the AdaDit model's forward pass.
- **Behavior preservation**: The `hasattr(self, 'ablation_mode')` guard means the ablation branch is entered only when an external caller actively sets `model.ablation_mode = 'zero'/'shuffle'`. By default `hasattr` is False; behavior is strictly equivalent to upstream.
- **Active status**: The driver shim **does not use AdaDit** (driver line 36 hardcodes `from model.CrossDit import Crossformer`). Therefore this hunk is DEAD CODE on the production path (paper P1–P7). It would only be entered by manually completing the missing config files and explicitly selecting AdaDit through some other entrypoint or modified config. The current `run_repro.py` has no `--cfg` parameter (it hardcodes loading `config/Cross.yaml`) and depends on a non-vendored YAML by default, so existing dpb automation paths cannot trigger it.
- **Performance impact**: None.

### 2.2 `src/train_split.py` (multiple hunks) — DEAD CODE

Main changes:
- Add `--ablation` and `--split` CLI parameters
- Append ablation suffix to `result_name` (avoiding overwrite of baseline results)
- Add `ablation_mode=...` to all `PertDit_Trainer(...)` calls
- **Remove the upstream-hardcoded "Drug_unseen" and "Cell_line_unseen" test loop** (upstream auto-runs two extra hardcoded test splits after training; dpb runs only once)

```diff
-    config['train']['split'] = "Drug_unseen"
-    now = datetime.now()
-    time_str = now.strftime('%H_%M_%S')
-    log_name = f'test_at_{time_str}'
-    my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt)
-    my_trainer.eval(dataset='test')
-
-    config['train']['split'] = "Cell_line_unseen"
     now = datetime.now()
     time_str = now.strftime('%H_%M_%S')
     log_name = f'test_at_{time_str}'
-    my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt)
+    my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt, ablation_mode=ablation_mode)
     my_trainer.eval(dataset='test')
```

- **Necessity**: dpb's experimental design requires running on user-specified splits (`drug_split_<F>` for cold-drug CV); upstream's hardcoded Drug/Cell unseen demonstration tests are not needed.
- **Behavior preservation**: N/A (dead code).
- **Dead-code argument**: The driver shim re-implements the full training loop (within `experiments/_shared/training/train_pertdit.py`) and does not call `train_split.py`. This file is retained as the upstream entrypoint script but does not run on any dpb automation path. `run_repro.py` likewise does not call `train_split.py`; it imports `PertDit_Trainer` directly.
- **Performance impact**: None (dead code).

### 2.3 `src/trainer/Trainer.py` (multiple hunks) — DEAD CODE

#### Hunk A: `__init__` adds `ablation_mode` + sets model attribute

```diff
-    def __init__(self, config, ckpt=None, log_name = 'my_log'):
+    def __init__(self, config, ckpt=None, log_name = 'my_log', ablation_mode='none'):
         self.config = config
+        self.ablation_mode = ablation_mode
         ...
+        # Set drug feature ablation mode
+        self.model.ablation_mode = self.ablation_mode
+        if self.ablation_mode != 'none':
+            print(f"[ABLATION] Drug feature ablation mode: {self.ablation_mode}")
+            logging.info(f"Drug feature ablation mode: {self.ablation_mode}")
```

- **Necessity**: Pass `ablation_mode` from external parameter to the model instance (paired with the §2.1 AdaDit hunk).
- **Behavior preservation**: `ablation_mode='none'` is the default; strictly equivalent to upstream.

#### Hunk B: `using_FC` config support (changes ReLU application + FC PCC metric)

```diff
+        self.using_FC = config.get('using_FC', False)
         ...
-            y_pred = self.relu(y_pred)
+            if not self.using_FC:
+                y_pred = self.relu(y_pred)
         ...
-        else: 
-            coeff, _ = calculate_correlation_coefficients(self.test_dataset.drug_adata.obs, 'condition', total_x, total_y_true, total_y_pred)
+        else:
+            # For FC mode: convert back to absolute expression before computing FC metrics
+            if self.using_FC:
+                eval_y_true = total_y_true + total_x
+                eval_y_pred = total_y_pred + total_x
+            else:
+                eval_y_true = total_y_true
+                eval_y_pred = total_y_pred
+            coeff, _ = calculate_correlation_coefficients(self.test_dataset.drug_adata.obs, 'condition', total_x, eval_y_true, eval_y_pred)
```

- **Necessity**: Supports "Fold Change (FC)" training mode — the model predicts `y_perturbed - y_control` rather than `y_perturbed` directly. ReLU should not be applied in FC mode (differences can be negative), and the FC PCC metric requires adding control back to make the comparison.
- **Behavior preservation**: `config['using_FC']` is absent by default → `using_FC = False` → ReLU applied + FC PCC computed directly; strictly equivalent to upstream.
- **Active status**: The driver shim does not import this Trainer class. `run_repro.py` does import it, but `run_repro.py` cannot currently run independently (it depends on the non-vendored `config/Cross.yaml`). Therefore all hunks in this file are **fully DEAD on dpb's current automation paths**. With `using_FC` defaulting to False when config is missing, even if `run_repro.py`'s config is later completed and run, default behavior would still equal upstream.

#### Hunk C: `model.ablation_mode` setting paired with §2.1

Pairs with Hunk A; not listed separately.

- **Driver-path dead-code argument**: The driver shim does not import `trainer.Trainer.PertDit_Trainer` (the driver implements its own training loop).
- **Updated active argument**: `run_repro.py:42` formally does `from trainer.Trainer import PertDit_Trainer`, but `run_repro.py` loads `config/Cross.yaml`, which is not vendored in dpb (see §3); therefore `run_repro.py` cannot run independently, and these changes are not executed on any of dpb's existing automation paths. Even if the YAML files were manually completed and the script run, default parameters (`ablation_mode='none'`, `using_FC=False`) preserve upstream behavior.

### 2.4 `src/utils/utils.py` (1 hunk) — DEAD on automation paths, ACTIVE only when manually running `eval_*.py`

```diff
@@ -8,7 +8,10 @@
 def cal_r2(y_true,y_pred):
     dim=y_true.shape[0]
-    metric = R2Score(num_outputs=dim, multioutput='raw_values')
+    try:
+        metric = R2Score(num_outputs=dim, multioutput='raw_values')
+    except (TypeError, ValueError):
+        metric = R2Score(multioutput='raw_values')
     if dim==1:
         return metric(y_pred.permute(1,0), y_true.permute(1,0)).unsqueeze(0)
```

- **Necessity**: Older torchmetrics versions of `R2Score` reject the `num_outputs` parameter (raising `TypeError`); newer versions accept it. The try/except keeps the code working under both versions.
- **Behavior preservation**: Strictly equivalent on newer torchmetrics (try branch); falls back to the compatible signature on older versions when `TypeError` is raised.
- **Active status**: The driver shim does not import `utils.utils` — DEAD. dpb research scripts `eval_checkpoint_sweep.py` and `eval_conditioning_ablation.py` (lines 15 / 17) do `from utils.utils import ... cal_r2 ...` — ACTIVE in those scripts only.

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `src/config/Ada.yaml` | Upstream config | AdaDit model default training config; dpb uses CrossDit, not needed |
| `src/config/CatBasicCross.yaml` | Upstream config | Same as above; another model variant |
| `src/config/Cross.yaml` | Upstream config | CrossDit model upstream training config; dpb driver generates its own config |
| `src/data/Lincs_mysplit.pkl` | Demo data | Upstream demo data split; not relevant to the dpb training pipeline |
| `src/data/PertDiT.png` | Documentation | Architecture figure |
| `src/plot_figures.ipynb` (531 KB) | Application notebook | Upstream plotting notebook; not relevant to the training pipeline |
| `src/preprocessing.ipynb` | Application notebook | Upstream data-preprocessing notebook; dpb uses `experiments/main_benchmark/data_prep/prepare_pertdit_embeddings.py` to implement the equivalent pipeline (comments mark "upstream preprocessing.ipynb cell-X" for algorithmic provenance) |
| `src/test_all_simple.ipynb` | Application notebook | Upstream demo notebook; not relevant to the training pipeline |
| `.gitignore` | Metadata | Not vendored |

### Local-only (not in upstream)

| File | Category | Notes |
|---|---|---|
| `src/bench_dataset.py` (~95 lines) | Data adapter | dpb-implemented `BenchPertDiTDataset`, `BenchCollater`, `build_dataloaders`. **The driver imports this file directly** (see §5.1) |
| `src/eval_checkpoint_sweep.py` | Research script | dpb-written checkpoint-sweep evaluation tool (best vs latest comparison on `random_split_0`); experimental debugging tool, not on the production pipeline |
| `src/eval_conditioning_ablation.py` | Research script | dpb-written conditioning-ablation evaluation tool (zero/shuffle conditioning comparison on existing checkpoints) |
| `src/run_repro.py` | Reproduction/debug script | dpb-written research/debug entrypoint using upstream `Trainer.PertDit_Trainer` to train PertDiT. **Currently not independently runnable**: `run_repro.py` loads `config/Cross.yaml`, but that YAML and the other 2 upstream YAMLs are not vendored (see "Upstream-only" above). The dpb automation paths (P1–P7) are completed by the unified driver `experiments/_shared/training/train_pertdit.py` and do not depend on this script |
| `src/data/result/findings_task4.md` | Reference material | dpb-compiled PertDiT reproducibility investigation document (e.g., naming differences between paper and public code); reference document, not part of execution |

### LICENSE status

✅ `models/PertDiT/LICENSE.NOTICE` is in place — upstream README declares Apache 2.0 (no separate LICENSE file); the dpb NOTICE file quotes the README for legal record.

### Added structure: `models/PertDiT/repo/`

PertDiT is the **only one of the seven vendored models that retains a nested structure** (`models/PertDiT/repo/src/...`). Other models (e.g., the earlier flattening of CIGER) tend to flatten upstream `<M>/<M>/...` directly into `models/<M>/...`, but PertDiT keeps the `repo/` nesting so that:
- dpb metadata such as `models/PertDiT/LICENSE.NOTICE` stays isolated from the vendored upstream code
- Future dpb-original tooling will not collide with upstream paths

---

## 4. SHA-256 Verification

### 4.1 Modified files

| File | Upstream SHA (first 16) | This dir SHA (first 16) |
|---|---|---|
| `src/model/AdaDit.py` | `56107c7e126af137` | `ede4c33a689ee76b` |
| `src/train_split.py` | `6d3bd6986f5fbe10` | `7ab78f48e45942b3` |
| `src/trainer/Trainer.py` | `f0e2790b963d4ddd` | `c1ff31bbe7971460` |
| `src/utils/utils.py` | `bfe942d6aba1669b` | `c03233f71009e6aa` |

### 4.2 Unchanged `.py` (25 files, byte-equivalent)

| File | SHA-256 (first 16) |
|---|---|
| `src/dataset/__init__.py` | `e3b0c44298fc1c14` |
| `src/dataset/drug_dose_encoder.py` | `b478bfa74b2f3752` |
| `src/dataset/my_Dataset.py` | `e2474f749dde057a` |
| `src/main.py` | `4e4a79185e561d6d` |
| `src/model/CrossDit.py` | `83d8121871f2f382` |
| `src/model/Cross_UNet.py` | `22c803635e6760f9` |
| `src/model/DirectCrossDit.py` | `f998b47f26438a12` |
| `src/model/EDM_AdaDit.py` | `37884e683b6d96bd` |
| `src/model/EDM_BasicCross.py` | `7364f80b40736253` |
| `src/model/EDM_Cross.py` | `ad14c37e96f2a0a1` |
| `src/model/PRNet.py` | `7a4e6db18cb6db65` |
| `src/model/__init__.py` | `e3b0c44298fc1c14` |
| `src/model/common.py` | `930aabeb2a78ed43` |
| `src/model/model_factory.py` | `454d59eac23a1a6c` |
| `src/sampler/Sampler.py` | `89f48c7ed86d1b0d` |
| `src/sampler/__init__.py` | `e3b0c44298fc1c14` |
| `src/sampler/edm.py` | `0910c3db394de304` |
| `src/test_split.py` | `84be97ec49e53f85` |
| `src/trainer/__init__.py` | `e3b0c44298fc1c14` |
| `src/trainer/lossfunc_and_generator.py` | `d1ab7f7cca373f4a` |
| `src/trainer/optimizer.py` | `ed64100b17624074` |
| `src/utils/__init__.py` | `e3b0c44298fc1c14` |
| `src/utils/cal_metrics.py` | `28c4e56809df6f1d` |
| `src/utils/plot.py` | `33cecf96c1a83cca` |
| `src/utils/seed_everything.py` | `df1957116e15af5b` |

### 4.3 dpb-added

| File | SHA-256 (first 16) |
|---|---|
| `src/bench_dataset.py` | `75211ba69ecaabdc` |
| `src/eval_checkpoint_sweep.py` | `1a82d3ca21a6c60e` |
| `src/eval_conditioning_ablation.py` | `106055c89aefc590` |
| `src/run_repro.py` | `423f05a15f309058` |

---

## 5. Driver Behavior Summary

### 5.1 Symbols actually imported by the driver shim (production path)

```python
# experiments/_shared/training/train_pertdit.py:36-40
from model.CrossDit import Crossformer            ← byte-equivalent
from sampler.Sampler import Diffusion_Sampler     ← byte-equivalent
from trainer.optimizer import get_optimizer_scheduler  ← byte-equivalent
from utils.seed_everything import seed_everything  ← byte-equivalent
from bench_dataset import BenchPertDiTDataset, build_dataloaders  ← dpb-added
```

The driver **does not import** any modified file (`AdaDit.py`, `train_split.py`, `trainer.Trainer`, `utils.utils`).

### 5.2 Active/dead status of modified files on the two paths

| File | Production path (driver shim) | Reproduction path (`run_repro.py`, `eval_*.py`) |
|---|---|---|
| `model/AdaDit.py` | DEAD (driver uses CrossDit) | DEAD (`run_repro.py` goes through `Choose_model`, defaulting to CrossDit; only explicit selection of an `Ada` config would reach AdaDit) |
| `train_split.py` | DEAD (driver does not call) | DEAD (`run_repro.py` implements its own main and does not call this script) |
| `trainer/Trainer.py` | DEAD (driver re-implements training loop) | DEAD (`run_repro.py` cannot currently run independently because `config/Cross.yaml` is not vendored) |
| `utils/utils.py` | DEAD (driver does not import) | DEAD on dpb automation paths; ACTIVE only when manually running the research scripts `eval_checkpoint_sweep.py` / `eval_conditioning_ablation.py` (they import `cal_r2`, but are not auto-triggered by any P1–P7 phase) |

### 5.3 Relationship between paper main results (P1–P7) and modified files

**All paper results P1–P7 are based on the production path** (the `main_benchmark` and `main_ablation` phases), which goes through the driver shim. On this path:
- All 4 modified files are **not imported**
- Training actually uses `model.CrossDit.Crossformer` (byte-equiv) + dpb-written training loop
- Therefore the PertDiT model architecture source is byte-equivalent to upstream commit `596d681`; the dpb driver uses this upstream-equivalent model + CrossDit config + dpb-custom training organization (data adaptation, evaluation output, ablation hooks). The model itself is not rewritten; among the 4 modified files, the model-layer (`AdaDit.py`) ablation hook is not triggered on the production path.

### 5.4 Actual relationship between `run_repro.py` and the `original_reproduction` phase

`experiments/original_reproduction/train_all.py` invokes `experiments/_shared/training/train_pertdit.py` (the unified driver) via subprocess; it **does not invoke** `models/PertDiT/repo/src/run_repro.py`.

`run_repro.py` is a dpb-maintained research/debug script (listed in §3 under "Local-only") for running a complete training locally on a PertDiT split as a small-scale experiment. It references `config/Cross.yaml`, but that YAML and the other 2 upstream YAMLs in `models/PertDiT/repo/src/config/` are not vendored (see §3); therefore `run_repro.py` cannot currently run independently.

**This means**: modifications to `models/PertDiT/repo/src/trainer/Trainer.py` (§2.3) and `utils/utils.py` (§2.4 `cal_r2`) **are not triggered on any of dpb's automation paths (P1–P7)** — because the unified driver `train_pertdit.py` uses `bench_dataset.BenchPertDiTDataset` + a self-implemented training loop, importing neither `Trainer.PertDit_Trainer` nor `cal_r2`.

The same applies to the Drug_unseen/Cell_line_unseen removal in `train_split.py` (§2.2) — `train_pertdit.py` does not import this file, and the default phase does not need it.

### 5.5 Where ablation logic is actually implemented

The driver shim **implements both ablation modes at the data layer**:
- `zero`: replaces `drug_emb_dict[idx]` with the zero-vector equivalent of `negative_ctrl`
- `shuffle`: permutes `pert_idx` with a fixed seed

**The ablation hook in `model.AdaDit.py` (§2.1) is never called by the driver** — even if the `model.ablation_mode` attribute existed, the driver uses CrossDit rather than AdaDit, and the driver handles ablation itself, not relying on internal model hooks.

### 5.6 Conclusion

Source modifications on the PertDiT production path (paper P1–P7) are **limited to the dpb-added `bench_dataset.py`** (data adapter). All 4 upstream-file modifications are off the production path.

The reproduction path (`run_repro.py` / `eval_*.py`) passes through 2 modified files (Trainer.py and utils.utils), but default parameters preserve upstream behavior. The 3 dpb research scripts (`eval_*.py` + `run_repro.py`) are dpb-original experimental debugging tools, coexisting with upstream code without modifying upstream behavior.
