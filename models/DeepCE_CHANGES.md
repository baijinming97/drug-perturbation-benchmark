# DeepCE — Modification Audit

**Upstream**: `https://github.com/pth1993/DeepCE` @ commit `9b0d04fa920ef73df578b070cfcc3982effdde42`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature |
|---|---|---|
| `DeepCE/utils/molecules.py` | 1 | Python 3.10+ compatibility fix |
| `DeepCE/models/deepce.py` | 2 | Drug-feature ablation hooks |
| `DeepCE/main_deepce.py` | 7 | Argparse typing + output dir + early stopping + ablation wiring |

**Unchanged files**: 14, SHA-256 byte-equivalent (§4)

---

## 2. Line-level Hunks

### 2.1 `DeepCE/utils/molecules.py`

```diff
@@ -1,3 +1,3 @@
 import rdkit
 from .molecule_utils import atom_features, bond_features
-from collections import Iterable
+from collections.abc import Iterable
```

- **Necessity**: `collections.Iterable` was removed in Python 3.10; direct import raises ImportError.
- **Behavior preservation**: Strictly equivalent — `collections.abc.Iterable` and the legacy `collections.Iterable` reference the same object; both are usable on Python 3.3–3.9.
- **Performance impact**: None.

### 2.2 `DeepCE/models/deepce.py`

```diff
@@ -54,7 +54,8 @@
                 else:
                     self.initializer(parameter)
 
-    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose):
+    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose,
+                ablation_mode='none'):
         # input_drug = {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}
         # gene_embed = [num_gene * gene_emb_dim]
         num_batch = input_drug['molecules'].batch_size
@@ -62,6 +63,16 @@
         # drug_atom_embed = [batch * num_node * drug_emb_dim]
         drug_embed = torch.sum(drug_atom_embed, dim=1)
         # drug_embed = [batch * drug_emb_dim]
+
+        # --- Drug feature ablation ---
+        if ablation_mode == 'zero':
+            drug_embed = torch.zeros_like(drug_embed)
+            drug_atom_embed = torch.zeros_like(drug_atom_embed)
+        elif ablation_mode == 'shuffle':
+            perm = torch.randperm(drug_embed.size(0), device=drug_embed.device)
+            drug_embed = drug_embed[perm]
+            drug_atom_embed = drug_atom_embed[perm]
+
         drug_embed = drug_embed.unsqueeze(1)
         # drug_embed = [batch * 1 *drug_emb_dim]
         drug_embed = drug_embed.repeat(1, self.num_gene, 1)
```

- **Necessity**: Provides drug-feature ablation paths (`zero`, `shuffle`) inside `forward`.
- **Behavior preservation**: `ablation_mode='none'` is the default; under that path neither the `if` nor the `elif` branch is entered, control flow is strictly identical to upstream. The new parameter only changes behavior when `'zero'` or `'shuffle'` is explicitly passed.
- **Performance impact**: Zero on the default path. In `zero/shuffle` modes the branches are the ablation intervention itself — this is experimental design rather than engineering modification.
- **⚠ Untriggered branch marker**: `models/deepce.py` **is imported and executed** on the production path, but the production training driver `experiments/_shared/training/train_deepce.py` does not pass `ablation_mode` in any of its 4 `model(...)` calls (L367, L391, L432, L528) — defaulting to `'none'`. Ablation is implemented at the driver layer via `register_forward_hook` injecting zero vectors at `model.drug_fp` output (L311–319) and data-layer batch shuffle (L190+). Therefore the new zero/shuffle branches are not triggered on the production path; behavior is equivalent to upstream — see §5.

### 2.3 `DeepCE/main_deepce.py` (7 hunks, 275 diff lines)

Grouped into two categories.

#### Category A: Argparse typing and new parameters (hunks 1–4)

```diff
@@ -1,12 +1,11 @@
 import os
-os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 import sys
+import time
+import json
 from datetime import datetime
 import torch
 import numpy as np
 import argparse
-# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/models')
-# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils')
 from models import DeepCE

@@ -14,25 +13,44 @@
 parser = argparse.ArgumentParser(description='DeepCE Training')
-parser.add_argument('--drug_file')
-parser.add_argument('--gene_file')
-parser.add_argument('--dropout')
-parser.add_argument('--train_file')
-parser.add_argument('--dev_file')
-parser.add_argument('--test_file')
-parser.add_argument('--batch_size')
-parser.add_argument('--max_epoch')
+parser.add_argument('--drug_file', required=True)
+parser.add_argument('--gene_file', required=True)
+parser.add_argument('--train_file', required=True)
+parser.add_argument('--dev_file', required=True)
+parser.add_argument('--test_file', required=True)
+parser.add_argument('--dropout', type=float, default=0.1)
+parser.add_argument('--batch_size', type=int, default=16)
+parser.add_argument('--max_epoch', type=int, default=500)
+parser.add_argument('--patience', type=int, default=50)
+parser.add_argument('--ablation', type=str, default='none', choices=['none', 'zero', 'shuffle'])
+parser.add_argument('--output_dir', type=str, default=None)
+parser.add_argument('--gpu', type=str, default='0')

 args = parser.parse_args()
+os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
```

- **Necessity**: All upstream parameters lacked `type/required/default`, allowing typos to silently flow through as `None`. `CUDA_VISIBLE_DEVICES` was hardcoded to `"0"`, which must be configurable in multi-GPU environments.
- **Behavior preservation**: Same-name parameters (`--drug_file` etc.) behave unchanged. Default values for new parameters `--ablation='none'`, `--output_dir=None`, `--gpu='0'` preserve upstream's default execution mode as much as possible. **`--patience=50` is a standalone-entrypoint addition for early stopping**; it does not enter the dpb production training path and cannot be claimed strictly equivalent to upstream's training entrypoint. **Note**: the dpb version moved `os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu` (L33) **after `import torch` (L6)** (upstream sets the environment variable before `import torch`). Therefore GPU visibility semantics cannot be claimed strictly equivalent to upstream. However, this standalone entrypoint does not enter the dpb production training path (see §5); the driver imports the `DeepCE` model class directly, and GPU selection is controlled by the dpb driver via `--dev`.
- **Performance impact**: None.

#### Category B: Output directory, early stopping, training history persistence (hunks 5–7)

<details><summary>Diff (~150 lines)</summary>

```diff
@@ -80,10 +97,32 @@
 model.to(device)
 model = model.double()

+# print config
+config_info = {
+    'ablation': ablation_mode, 'max_epoch': max_epoch, 'patience': patience,
+    'batch_size': batch_size, 'dropout': dropout, 'lr': 0.0002, ...
+}
+print('Config:', json.dumps(config_info, indent=2))
+with open(os.path.join(output_dir, 'config.json'), 'w') as f:
+    json.dump(config_info, f, indent=2)
+
 # training
 optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
-best_dev_loss = float("inf")
 best_dev_pearson = float("-inf")
+best_dev_epoch = -1
+epochs_no_improve = 0
+save_best_predictions = False

@@ -114,16 +153,17 @@
-        predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose)
+        predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
+                        ablation_mode=ablation_mode)

@@ -168,9 +208,20 @@
-        if best_dev_pearson < pearson:
+        # Early stopping check
+        if pearson > best_dev_pearson:
             best_dev_pearson = pearson
+            best_dev_epoch = epoch
+            epochs_no_improve = 0
+            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
+            save_best_predictions = True
+        else:
+            epochs_no_improve += 1

@@ +268, ...
+        if save_best_predictions:
+            np.save(os.path.join(output_dir, 'test_labels.npy'), lb_np)
+            np.save(os.path.join(output_dir, 'test_predictions.npy'), predict_np)
+            save_best_predictions = False
+
+    if epochs_no_improve >= patience:
+        print('Early stopping at epoch %d (patience=%d)' % (epoch + 1, patience))
+        break

@@ +303, ... (final results.json dump)
+results = {'ablation': ablation_mode, 'best_dev_epoch': ..., 'history': {...}}
+with open(os.path.join(output_dir, 'results.json'), 'w') as f:
+    json.dump(results, f, indent=2)
```

</details>

- **Necessity**: Allow `main_deepce.py` to be a re-runnable standalone reproduction entrypoint without overwriting itself; produce downstream-analyzable `config.json` / `best_model.pt` / `test_predictions.npy` / `results.json`; early-stop on dev Pearson.
- **Behavior preservation**: `main_deepce.py` is **not invoked** on the production path (see §5); the driver constructs the model directly via `from models import DeepCE` with its own training loop. These changes only affect reproducers running `python main_deepce.py` standalone and do not affect production metrics. The newly added early stopping, output directory, and best-checkpoint-saving constitute standalone-entrypoint behavior changes; this entrypoint cannot be claimed strictly equivalent to upstream's training entrypoint.
- **Performance impact**: None (off-path).

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `docs/` | Documentation | Architecture figures referenced by README |
| `script/` | Scripts | Upstream shell wrappers, superseded by top-level `experiments/` |
| `DeepCE/data/*.csv` (5 files) | Data | Upstream-bundled training data; in dpb located at top-level `data/DeepCE/` (provided by fetch; data file equivalence is not part of this SHA table) |

### Local-only (not in upstream)

None — `models/DeepCE/` adds no extra files to the upstream tree.

### LICENSE status

⚠ The upstream DeepCE repository contains no LICENSE file at the pinned commit, and the README contains no license declaration. This directory `models/DeepCE/` likewise carries no LICENSE. The top-level `NOTICE` documents this — vendoring proceeds under full attribution in `PROVENANCE.md`, but redistribution rights are not explicitly granted by upstream; commercial use requires contacting the upstream author (Thai-Hoang Pham, Ohio State University).

---

## 4. SHA-256 Verification (unchanged files)

| File | SHA-256 (first 16) |
|---|---|
| `DeepCE/main_drug_repurposing.py` | `ea85e2694cdfbb55` |
| `DeepCE/models/__init__.py` | `f34705ba73a8cde3` |
| `DeepCE/models/drug_gene_attention.py` | `e0a883438085d009` |
| `DeepCE/models/graph_degree_conv.py` | `0c96c226501a8f73` |
| `DeepCE/models/loss_utils.py` | `75dd05593116d729` |
| `DeepCE/models/ltr_loss.py` | `e7febf517a037fd4` |
| `DeepCE/models/multi_head_attention.py` | `610b9ccf2b87f790` |
| `DeepCE/models/neural_fingerprint.py` | `20eafe58f42715b9` |
| `DeepCE/models/positionwide_feedforward.py` | `48a18c5de451b70e` |
| `DeepCE/utils/__init__.py` | `267f30b36894523a` |
| `DeepCE/utils/data_utils.py` | `2f3d8fa122fbe0b4` |
| `DeepCE/utils/datareader.py` | `fab3721d4874f5ef` |
| `DeepCE/utils/metric.py` | `c9bbf1564584f51e` |
| `DeepCE/utils/molecule_utils.py` | `6980139aed13cc2e` |

All 14 files byte-equivalent to upstream commit `9b0d04f`.

---

## 5. Driver Behavior Summary

> Explains the operational status of each §2 hunk on the production path (`experiments/_shared/training/train_deepce.py`).

| Model-layer change | Production path | Driver substitute |
|---|---|---|
| `forward(..., ablation_mode='none')` new signature | ✅ Loaded and executed, not triggered | Driver does not pass this parameter in any of its 4 `model(...)` calls; default `'none'` keeps the new branches unentered |
| `forward` inner `if ablation_mode == 'zero'` branch | ✅ Loaded and executed, not triggered | Driver L311–319 uses `model.drug_fp.register_forward_hook` to inject zero vectors externally |
| `forward` inner `elif ablation_mode == 'shuffle'` branch | ✅ Loaded and executed, not triggered | Driver L190+ permutes `pert_idx_all` at the data layer (seed=131419) |
| All `main_deepce.py` changes | ❌ Not invoked | Driver uses `from models import DeepCE` to construct the model directly, with its own training loop |
| `from collections.abc import Iterable` | ✅ Exercised | Triggered via chained import |

**Conclusion**: The DeepCE production path **substantively depends on only 1 hunk** (the compatibility import in §2.1). The remaining 9 hunks are modernizations of upstream's standalone entrypoint (allowing it to still run independently), but the production experiments themselves do not pass through that path. Even if all ablation branches in `models/DeepCE/` and all `main_deepce.py` modifications were reverted to the upstream commit, production metrics would not change.
