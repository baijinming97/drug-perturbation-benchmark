# PRnet — Modification Audit

**Upstream**: `https://github.com/Perturbation-Response-Prediction/PRnet` @ commit `f19174bde3ed2633f54c7831799cc38c4ffc7a0d`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature |
|---|---|---|
| `data/Dataset.py` | 1 | Data loader performance optimization (`list.index()` O(N) → dict O(1)) |
| **Unchanged `.py`** | — | 14 files, SHA-256 byte-equivalent (§4) |
| **Added** | — | `models/bench_dataset.py` (data adapter, used by driver) |

`models/PRnet/` is overall close to upstream byte-equivalent, introducing only one semantics-preserving performance optimization. The paper's claim (driver does not modify the PRnet model) holds strictly at the code level.

---

## 2. Line-level Hunks

### 2.1 `data/Dataset.py` (1 hunk) — performance optimization, DEAD CODE

```diff
@@ -45,6 +46,7 @@
         self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
         self.dense_adata_index = self.dense_adata.obs.index.to_list()
+        self.dense_adata_index_map = {idx: pos for pos, idx in enumerate(self.dense_adata_index)}

@@ -64,7 +65,7 @@
-        control_index = self.dense_adata_index.index(self.paired_control_index[index])
+        control_index = self.dense_adata_index_map[self.paired_control_index[index]]
```

- **Necessity**: Upstream `__getitem__` uses `list.index()` for O(N) linear-scan lookup of the control sample index. On large datasets (N ~ 100k+), this is a data-loading bottleneck. Replacing with a pre-built dict reduces lookup complexity from O(N) to O(1).
- **Behavior preservation**: **Strictly equivalent under the precondition that `dense_adata.obs.index` is unique** — `dict` lookup yields the same result as `list.index()`. This precondition aligns with standard AnnData usage (verifiable at runtime via `adata.obs_names.is_unique`); upstream assumes the same precondition. If duplicates exist, `list.index()` returns the first position while a dict comprehension retains the last position — the two would diverge in that case.
- **Dead-code marker**: ✓ (this file is referenced by upstream `PRnetTrainer.py`, which is not on the dpb production path — see §5)
- **Performance impact**: Significantly speeds up training data loading for scenarios using `PRnetTrainer.py` (outside dpb); the dpb production path is unaffected (it uses `bench_dataset.py`).

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `Dockerfile` | Deployment | Upstream Docker image config; no behavioral relevance |
| `checkpoint/lincs_best_epoch_all.pt` | Demo weights | Upstream-bundled LINCS demo checkpoint; production trains from scratch and does not read |
| `checkpoint/sciplex_best_epoch_all.pt` | Demo weights | Upstream-bundled sci-Plex demo checkpoint; same as above |
| `dataset/demo.h5ad` | Demo data | Sample data for upstream demo notebook; not read by the production path |
| `dataset/l1000_data_w.csv` | Application data | Upstream-bundled LINCS weight table; not read by production training |
| `img/PRnet.svg` | Documentation | Architecture figure |
| `figure/drug_candidates_recomandation.ipynb` | Application notebook | Upstream case-study notebook; not needed by production training |
| `figure/latent_tsne_lung_cancer.ipynb` (1.3 MB) | Application notebook | Same as above |
| `figure/SCLC_plot_dsea.ipynb` | Application notebook | Same as above |
| `preprocessing/custom_data_preprocessing.ipynb` | Application notebook | Same as above |

### Local-only (not in upstream)

| File | Category | Notes |
|---|---|---|
| `models/bench_dataset.py` (31 lines) | Data adapter | dpb-implemented PyTorch `Dataset` class (`BenchPRnetDataset`) feeding the PRnet training loop with `(x_ctrl, x_pert, fcfp4_feat)` triples in the dpb common format. **The driver imports this file directly** (see §5) |

### LICENSE status

✅ `models/PRnet/LICENSE` (Apache 2.0) is byte-equivalent to upstream (copied verbatim from `reference/PRnet/LICENSE`).

---

## 4. SHA-256 Verification

### 4.1 Modified files

| File | Upstream SHA (first 16) | This dir SHA (first 16) | Byte-equivalent? |
|---|---|---|---|
| `data/Dataset.py` | `4141f10b...` | `076af6d1...` | ✗ (performance optimization, §2.1) |

### 4.2 Unchanged files (byte-equivalent)

| File | SHA-256 (first 16) |
|---|---|
| `analysis_lincs.py` | `56fd89f54f340b84` |
| `analysis_sciplex.py` | `6fb77340251fd776` |
| `data/__init__.py` | `e3b0c44298fc1c14` |
| `data/_utils.py` | `b72ad61f7e4e1389` |
| `models/PRnet.py` | `024f39a07e2887fe` |
| `models/__init__.py` | `e3b0c44298fc1c14` |
| `test_demo.py` | `7adc897816106a78` |
| `test_lincs.py` | `e50b510bc0b6b5b7` |
| `test_sciplex.py` | `97fddd689ba9b0ba` |
| `train_lincs.py` | `740e564da62c40ef` |
| `train_sciplex.py` | `e1a99dd5d9c344e8` |
| `trainer/PRnetTrainer.py` | `0d0eb2d8e19ce284` |
| `trainer/__init__.py` | `e3b0c44298fc1c14` |
| `trainer/_utils.py` | `d746eeb9cb6c4118` |

### 4.3 dpb-added

| File | SHA-256 (first 16) |
|---|---|
| `models/bench_dataset.py` | (dpb-original; no upstream counterpart) |

---

## 5. Driver Behavior Summary

### 5.1 Symbols actually imported by the driver from PRnet root

```python
# experiments/_shared/training/train_prnet.py:32-33
from PRnet import PGM                    ← upstream models/PRnet.py (byte-equivalent)
from bench_dataset import BenchPRnetDataset  ← dpb-added models/bench_dataset.py
```

The driver **does not import**: `trainer.PRnetTrainer`, `data.Dataset` (including the optimized version of `Dataset.py`).

### 5.2 Status of modified files on the production path

| File | On production path? |
|---|---|
| `models/PRnet.py` | ✅ Yes (driver instantiates the `PGM` class); byte-equivalent to upstream |
| `data/Dataset.py` (performance-optimized version) | ❌ No (driver uses `bench_dataset.py` and does not import this file) |
| `trainer/PRnetTrainer.py` | ❌ No (driver re-implements the training loop and does not import this Trainer class) |

### 5.3 Where ablation logic is actually implemented

The driver implements ablation at the **data layer**:
- `zero` mode: sets `fcfp4_dict[idx]` to zero vectors (before dataloader construction)
- `shuffle` mode: permutes `pert_idx_all` with a fixed seed

The model source needs no modification to support ablation.

### 5.4 Conclusion

**The only source modification on the PRnet production path is `models/bench_dataset.py`** (the dpb-added data adapter). All other upstream `.py` files are byte-equivalent to commit `f19174b`, except `data/Dataset.py` which contains one semantics-preserving performance optimization (O(N) → O(1) data loading); this optimized file itself is not on the production path.
