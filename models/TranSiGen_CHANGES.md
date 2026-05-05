# TranSiGen — Modification Audit

**Upstream**: `https://github.com/myzheng-SIMM/TranSiGen` @ commit `8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature | Dead code? |
|---|---|---|---|
| `src/model.py` | 5 | Adds `ablation_mode` hook (`zero` implemented here) | ❌ Driver imports |
| `src/utils.py` | 1 | NumPy `np.str` compatibility fix | ❌ Driver imports |
| `src/dataset.py` | 1 | Restores full LINCS path (uncommented from sample path) | ✅ Driver uses `bench_dataset.py` |
| `src/train_TranSiGen.py` | 1 | Upstream VAE pretraining bug fix (incorrect `x1` path) | ✅ Driver does not reproduce pretraining |
| `src/train_TranSiGen_full_data.py` | 5 | CLI ablation parameter + output dir timestamping | ✅ Driver re-implements training loop |
| **Unchanged `.py`** | — | 3 files, SHA byte-equivalent (§4) |
| **Added** | — | `src/bench_dataset.py` (data adapter, used by driver), `src/evaluate_TranSiGen_full_data.py` (evaluation script) |

---

## 2. Line-level Hunks

### 2.1 `src/model.py` (5 hunks) — ACTIVE

#### Hunk 1: `forward` adds `ablation_mode` parameter + zero implementation

```diff
@@ -158,8 +158,10 @@
-    def forward(self, x1, features):
-        """ Forward pass through full network"""
+    def forward(self, x1, features, ablation_mode='none'):
+        """ Forward pass through full network.
+        ablation_mode: 'none' (default), 'zero', or 'shuffle' for drug feature ablation.
+        """
         z1, mu1, logvar1 = self.encode_x1(x1)
         x1_rec = self.decode_x1(z1)
@@ -167,6 +169,11 @@
         if hasattr(self, 'feat_embeddings'):
             feat_embed = self.feat_embeddings(features)
         else:
             feat_embed = features
+
+        # Drug feature ablation (zero only; shuffle handled at data layer)
+        if ablation_mode == 'zero':
+            feat_embed = torch.zeros_like(feat_embed)
+
         z1_feat = torch.cat([z1, feat_embed], 1)
```

- **Necessity**: Introduces an ablation-mode hook into TranSiGen `forward`. `zero` zeros out the `feat_embed` embedding tensor, completely erasing drug features from the latent encoding.
- **Behavior preservation**: `ablation_mode='none'` is the default, keeping `feat_embed = self.feat_embeddings(features)` unchanged; strictly equivalent to upstream.
- **Performance impact**: None (default path does not enter the zero branch).

#### Hunks 2–5: `train_model`, `test_model`, `predict_profile` thread `ablation_mode` through

```diff
-    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
+    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None, ablation_mode='none'):
         ...
-                x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, z2_pred = self.forward(x1_train, features)
+                x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, z2_pred = self.forward(x1_train, features, ablation_mode=ablation_mode)
         ...
-    def test_model(self, loader, loss_item=None, metrics_func=None):
+    def test_model(self, loader, loss_item=None, metrics_func=None, ablation_mode='none'):
         ...
-                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
+                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features, ablation_mode=ablation_mode)
         ...
-    def predict_profile(self, loader):
+    def predict_profile(self, loader, ablation_mode='none'):
         ...
-                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
+                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features, ablation_mode=ablation_mode)
```

- **Necessity**: Threads `ablation_mode` from external API down to `forward`.
- **Behavior preservation**: All default to `'none'`; equivalent to upstream.
- **Performance impact**: None.

### 2.2 `src/utils.py` (1 hunk) — ACTIVE

```diff
@@ -34,14 +34,14 @@
         for key in f:
             data[key] = np.asarray(f[key])
             if isinstance(data[key][0], np.bytes_):
-                data[key] = data[key].astype(np.str)
+                data[key] = data[key].astype(str)
     return data
 
 def save_to_HDF(fname, data):
     """Save data (a dictionary) to a HDF5 file."""
     with h5py.File(fname, 'w') as f:
         for key, item in data.items():
-            if isinstance(item[0], np.str):
+            if isinstance(item[0], str):
                 item = item.astype(np.bytes_)
             f[key] = item
```

- **Necessity**: `np.str` was deprecated in NumPy 1.20+ and raises errors in 1.24+ (fully removed in 2.x). Switching to Python's native `str`. This change targets primarily NumPy 1.24+ / 2.x compatibility.
- **Behavior preservation**: Byte-equivalent behavior (`np.str` was historically an alias for `str`).
- **Production-path relationship**: The `utils.py` module **is imported on the production path** (the driver uses `setup_seed`/`seed_worker`), but the `np.str → str` modification lives inside `load_from_HDF`/`save_to_HDF`; the driver's critical path does not call these two functions. The modification is a NumPy compatibility fix and does not affect default production training behavior.
- **Performance impact**: None.

### 2.3 `src/dataset.py` (1 hunk) — DEAD CODE

```diff
@@ -11,17 +11,17 @@
         self.cid = cid
-        # self.LINCS_data = load_from_HDF('../data/LINCS2020/processed_data.h5')
-        self.LINCS_data = load_from_HDF('../data/LINCS2020/data_example/processed_data.h5')
+        self.LINCS_data = load_from_HDF('../data/LINCS2020/processed_data.h5')
+        # self.LINCS_data = load_from_HDF('../data/LINCS2020/data_example/processed_data.h5')
         with open('../data/LINCS2020/idx2smi.pickle', 'rb') as f:
             self.idx2smi = pickle.load(f)
         if self.mol_feature_type == 'ECFP4':
-            # with open('../data/LINCS2020/ECFP4_emb2048.pickle', 'rb') as f:
-            with open('../data/LINCS2020/data_example/ECFP4_emb2048.pickle', 'rb') as f:
+            with open('../data/LINCS2020/ECFP4_emb2048.pickle', 'rb') as f:
+            # with open('../data/LINCS2020/data_example/ECFP4_emb2048.pickle', 'rb') as f:
                 self.smi2emb = pickle.load(f)
         elif self.mol_feature_type == 'KPGT':
-            # with open('../data/LINCS2020/KPGT_emb2304.pickle', 'rb') as f:
-            with open('../data/LINCS2020/data_example/KPGT_emb2304.pickle', 'rb') as f:
+            with open('../data/LINCS2020/KPGT_emb2304.pickle', 'rb') as f:
+            # with open('../data/LINCS2020/data_example/KPGT_emb2304.pickle', 'rb') as f:
                 self.smi2emb = pickle.load(f)
```

- **Necessity**: Upstream commented out the full LINCS path and exposed the sample-data (`data_example/`) path. dpb swaps the comments, pointing this file at the full data.
- **Behavior preservation**: N/A (dead code).
- **Dead-code argument**: The driver imports `bench_dataset.BenchTranSiGenDataset` rather than `dataset.TranSiGenDataset`; see §5.1.
- **Performance impact**: None (dead code).

### 2.4 `src/train_TranSiGen.py` (1 hunk) — DEAD CODE, but a genuine upstream bug fix

```diff
@@ -144,7 +144,7 @@
             for k in model_dict.keys():
                 if k in model_base_x1_dict.keys():
                     model_dict[k] = model_base_x1_dict[k]
-            filename = '../results/trained_model_shRNA_vae_x1/best_model.pt'
+            filename = '../results/trained_model_shRNA_vae_x2/best_model.pt'
             model_base_x2 = torch.load(filename, map_location='cpu')
             model_base_x2_dict = model_base_x2.state_dict()
             for k in model_dict.keys():
```

- **Necessity**: **Upstream bug**. This is the VAE x2 initialization flow, but `filename` mistakenly pointed at the `vae_x1` checkpoint, causing x1 and x2 to load the same VAE weights — model functionality broken. dpb corrects the path to `vae_x2`.
- **Behavior preservation**: N/A (dead code).
- **Dead-code argument**: `train_TranSiGen.py` is upstream's VAE pretraining script. dpb directly uses upstream-released pretrained VAE checkpoints (`reference/TranSiGen/results/trained_model_shRNA_vae_{x1,x2}/best_model.pt`) and does not re-pretrain in the production pipeline.
- **Performance impact**: None (dead code).
- **Note**: If a future user wishes to use upstream's `train_TranSiGen.py` to re-pretrain the VAE, this bug fix would benefit them. But because upstream users in practice consume the pretrained checkpoint directly, this bug appears to have gone undetected for years.

### 2.5 `src/train_TranSiGen_full_data.py` (5 hunks) — DEAD CODE

Includes: CLI `--ablation` parameter, runtime timestamp, ablation-suffixed output directory, `train_model(..., ablation_mode=...)` threading, `predict_profile(..., ablation_mode=...)` threading.

- **Dead-code argument**: The dpb driver `train_transigen.py` **re-implements the full training loop** and does not import `train_TranSiGen_full_data.py`. The modifications are scaffolding for ablation hooks added during vendoring, but the production path goes through the driver's own training loop and does not pass through this file.

Per-hunk code is omitted because the entire file is not executed.

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `requirements.txt` | Dependency declaration | dpb maintains unified dependencies at the top level; this file does not need to be vendored |
| `data/LINCS2020/` | Training data | Upstream-bundled LINCS training data (`processed_data_id.h5`, `idx2smi.pickle`, `modz_x1.pickle`, `geneinfo_processed.csv`); in dpb **moved to top-level `data/TranSiGen/LINCS2020/`** |
| `data/PRISM/` | Application data | Upstream PRISM screening data (`KPGT_emb2304.pickle`, `screening_compound.csv`); for case-study use, not read by production training |
| `data/TranSiGen.jpg` | Documentation | Architecture figure |
| `src/drug_response_prediction.ipynb` | Application notebook | Upstream case-study; not relevant to the training pipeline |
| `src/ligand_based_virtual_screening.ipynb` | Application notebook | Same as above |
| `src/phenotype_based_drug_repurposing.ipynb` | Application notebook | Same as above |
| `src/train_TranSiGen_full_data.sh` | Run script | dpb invokes the driver directly; this shell wrapper is not needed |
| `results/trained_model_shRNA_vae_x1/best_model.pt` (8.8 MB) | Pretrained weights | The dpb training driver reads from `reference/TranSiGen/results/...` (populated by `fetch_upstream.sh TranSiGen`); this directory does not duplicate the file to keep the repo small |
| `results/trained_model_shRNA_vae_x2/best_model.pt` (8.8 MB) | Pretrained weights | Same as above |
| `results/trained_models_164_cell_smiles_split/.../best_model.pt` (22 MB) | Pretrained weights | Used by upstream case-study script `prediction.py`; not read by the dpb training pipeline |

### Local-only (not in upstream)

| File | Category | Notes |
|---|---|---|
| `src/bench_dataset.py` (38 lines) | Data adapter | dpb-implemented PyTorch `Dataset` class `BenchTranSiGenDataset`, feeding the TranSiGen training loop with `(x_ctrl, x_pert, pert_idx, cell_idx, kpgt_dict)` tuples in the dpb common format. **The driver imports this file directly** (see §5.1) |
| `src/evaluate_TranSiGen_full_data.py` | Evaluation tool | dpb-written standalone evaluation script that runs predictions + metrics after training. Decoupled from the training pipeline; exists as an independent post-processing tool |

### LICENSE status

✅ `models/TranSiGen/LICENSE` (MIT, Copyright 2023 myzheng-SIMM) is byte-equivalent to upstream (copied verbatim from `reference/TranSiGen/LICENSE`).

---

## 4. SHA-256 Verification

### 4.1 Modified files

| File | Upstream SHA (first 16) | This dir SHA (first 16) | Byte-equivalent? |
|---|---|---|---|
| `src/dataset.py` | `2b2f32b3b41f3a84` | `f7553919c67cba4a` | ✗ (dead code, §2.3) |
| `src/model.py` | `c328864a040a3c1a` | `777152db3b924f23` | ✗ (active, §2.1) |
| `src/train_TranSiGen.py` | `4446df722b2dcd96` | `4137dc37e2b6bef4` | ✗ (dead code + bug fix, §2.4) |
| `src/train_TranSiGen_full_data.py` | `5925f10737588795` | `dee6318f59dce63b` | ✗ (dead code, §2.5) |
| `src/utils.py` | `00500a5866c49551` | `04d81378da8b896b` | ✗ (active, §2.2) |

### 4.2 Unchanged files (byte-equivalent)

| File | SHA-256 (first 16) |
|---|---|
| `src/prediction.py` | `00194e1a06022ce5` |
| `src/vae_x1.py` | `522dac9e9c489cc0` |
| `src/vae_x2.py` | `59e3569bcbd6543f` |

### 4.3 dpb-added

| File | SHA-256 (first 16) |
|---|---|
| `src/bench_dataset.py` | `e91c935e6d5b34c9` (dpb-original; no upstream counterpart) |
| `src/evaluate_TranSiGen_full_data.py` | `b5b850119b14402b` (dpb-original; no upstream counterpart) |

---

## 5. Driver Behavior Summary

### 5.1 Symbols actually imported by the driver from TranSiGen src

```python
# experiments/_shared/training/train_transigen.py:28-30
from model import TranSiGen          ← src/model.py (active modification, §2.1)
from utils import setup_seed, seed_worker  ← src/utils.py (active modification, §2.2)
from bench_dataset import BenchTranSiGenDataset  ← dpb-added src/bench_dataset.py
```

The driver **does not import**: `dataset.py`, `train_TranSiGen.py`, `train_TranSiGen_full_data.py`.

### 5.2 Status of modified files on the production path

| File | On production path? | Default-behavior equivalence to upstream |
|---|---|---|
| `src/model.py` | ✅ Yes (driver instantiates the `TranSiGen` class) | ✓ `ablation_mode='none'` default; equivalent |
| `src/utils.py` | ✅ Yes (driver calls `setup_seed`) | `np.str → str` is semantically equivalent under older NumPy; required for NumPy 1.24+ / 2.x; the modified functions (`load_from_HDF`/`save_to_HDF`) are not on the driver's main path |
| `src/dataset.py` | ❌ No (driver uses `bench_dataset.py`) | N/A |
| `src/train_TranSiGen.py` | ❌ No (driver does not reproduce VAE pretraining) | N/A |
| `src/train_TranSiGen_full_data.py` | ❌ No (driver implements its own training loop) | N/A |

### 5.3 On upstream VAE pretrained checkpoints

The driver loads upstream-released VAE checkpoints during training via `--vae_x1_path` / `--vae_x2_path`:

```
reference/TranSiGen/results/trained_model_shRNA_vae_x1/best_model.pt
reference/TranSiGen/results/trained_model_shRNA_vae_x2/best_model.pt
```

Automatically downloaded from Zenodo by `fetch_upstream.sh TranSiGen`. dpb does not re-pretrain the VAE, so the bug fix in §2.4 has no effect on the production path.

### 5.4 Where ablation logic is actually implemented

- **`zero` mode**: implemented in `model.py.forward` (§2.1 Hunk 1) — zeros out the entire `feat_embed` embedding tensor. This is the **only** source modification the driver actually exercises.
- **`shuffle` mode**: implemented in the driver's data layer (fixed-seed permutation of `pert_idx_all`), independent of the model source.

### 5.5 Conclusion

Source modifications on the TranSiGen production path reduce to two:

1. `model.py`: adds the `ablation_mode` kwarg + zero implementation. With `ablation_mode='none'` as default, non-ablation experiments in paper P1–P7 are strictly equivalent to upstream.
2. `utils.py`: `np.str → str` NumPy compatibility fix (required to run under NumPy 1.24+ / 2.x; semantically equivalent under older NumPy where `np.str` was an alias for `str`). The modification lives in `load_from_HDF`/`save_to_HDF`, which the driver does not call, so default production training behavior is unaffected.

All modifications in the remaining 3 files (`dataset.py` / `train_TranSiGen.py` / `train_TranSiGen_full_data.py`) lie off the production path; reverting them all to upstream byte-equivalent would not change production metrics. Of these, the modification in `train_TranSiGen.py` includes a genuine upstream bug fix (the VAE x2 path mistake), but because dpb does not reproduce VAE pretraining, this fix is not on the production path.
