# MultiDCP — Modification Audit

**Upstream**: `https://github.com/XieResearchGroup/MultiDCP` @ commit `36ecdef9598de703b61fe0d1b91bcfeeba844274`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature |
|---|---|---|
| (none) | — | No source modifications |

All source files under `models/MultiDCP/` are **byte-equivalent** to the upstream pinned commit. This directory is a faithful copy of upstream, introducing no code-level modifications.

---

## 2. Line-level Hunks

None.

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `docs/` | Documentation | Upstream README figures only; no behavioral impact |
| `trained_models/` (72 MB, 6 `.pt` files) | Application weights | Pre-trained ensemble for upstream case-study scripts (`drug_response_infer.py`, `predict_food_molecules_mcf7.py`). The dpb pipeline trains MultiDCP from scratch and does not depend on these weights; the two case-study scripts are not on the production path. |

### Local-only (not in upstream)

None — `models/MultiDCP/` adds no extra files to the upstream tree.

### LICENSE status

✅ `models/MultiDCP/LICENSE` (Creative Commons BY-NC 4.0) is byte-equivalent to upstream (copied verbatim from `reference/MultiDCP/LICENSE`).

⚠ **Commercial use restriction**: CC BY-NC 4.0 prohibits commercial use. This model's terms differ from the dpb top-level MIT — the vendored subdirectory is constrained by the stricter BY-NC license. The top-level `NOTICE` documents this; commercial use requires obtaining additional authorization from the MultiDCP upstream authors.

---

## 4. SHA-256 Verification (core source + scripts byte-equivalent)

### 4.1 `MultiDCP/` directory (35 `.py` files)

| File | SHA-256 (first 16) |
|---|---|
| `MultiDCP/add_gene_names_to_predictions.py` | `d0334f03f10b8394` |
| `MultiDCP/adr_prediction/adr_data_builder.py` | `e2bc291fc823d6b3` |
| `MultiDCP/adr_prediction/adr_prediction.py` | `9ca434b53b7f5efd` |
| `MultiDCP/adr_prediction/model_builder.py` | `450836cdddedb34f` |
| `MultiDCP/binary_eval_on_DeepCOP_data.py` | `c2383ecb6b16272d` |
| `MultiDCP/binary_multidcp_ae.py` | `dcba160e7fb6e557` |
| `MultiDCP/drug_response_infer.py` | `729653359bb93404` |
| `MultiDCP/drug_response_train.py` | `7f05809125b82db5` |
| `MultiDCP/ehill_multidcp_finetune.py` | `35881cd37603fef1` |
| `MultiDCP/ehill_multidcp_mt.py` | `4f2e393150413b6e` |
| `MultiDCP/ehill_multidcp_pretrain.py` | `d909a2fab3a8f9f7` |
| `MultiDCP/main_deepce.py` | `9f56f89fc439df67` |
| `MultiDCP/models/drug_gene_attention.py` | `27ef64754c7df6d7` |
| `MultiDCP/models/graph_degree_conv.py` | `0c96c226501a8f73` |
| `MultiDCP/models/loss_utils.py` | `11143cd69dd8fc01` |
| `MultiDCP/models/ltr_loss.py` | `c5fb1088194fdf2f` |
| `MultiDCP/models/multi_head_attention.py` | `610b9ccf2b87f790` |
| `MultiDCP/models/multidcp.py` | `f697049cf1b46cc1` |
| `MultiDCP/models/neural_fingerprint.py` | `00b4fd45a6fe8fec` |
| `MultiDCP/models/positionwide_feedforward.py` | `48a18c5de451b70e` |
| `MultiDCP/models/scheduler_lr.py` | `22db8ab486a3003b` |
| `MultiDCP/multidcp_ae.py` | `bdcebfa391e5ac0f` |
| `MultiDCP/multidcp_ae_inferrence.py` | `30146d8ea1c2e4d9` |
| `MultiDCP/multidcp_ae_pretrain.py` | `ace74d2f524e3370` |
| `MultiDCP/multidcp_ae_second.py` | `04d654f543dbaef4` |
| `MultiDCP/predict_food_molecules_mcf7.py` | `7f2ba89749cce0b4` |
| `MultiDCP/pretrain_multidcp.py` | `1ed7f0c240bc0b27` |
| `MultiDCP/utils/data_for_deepCOP_ranking.py` | `2b855598d8501810` |
| `MultiDCP/utils/data_utils.py` | `66af7df4fe6ed277` |
| `MultiDCP/utils/datareader.py` | `20d322f5ccd71e67` |
| `MultiDCP/utils/drug_response_curve_cal.py` | `86e25f6f8e36e778` |
| `MultiDCP/utils/metric.py` | `c37cdd737c2ba0a0` |
| `MultiDCP/utils/molecule_utils.py` | `3eb710c6b0f762d5` |
| `MultiDCP/utils/molecules.py` | `7f9bcf79db0efe18` |
| `MultiDCP/utils/multidcp_ae_utils.py` | `73f4ecda496d339b` |

### 4.2 `script/` directory (16 files)

| File | SHA-256 (first 16) |
|---|---|
| `script/docker_folder/Dockerfile` | `51bc75617a29c212` |
| `script/docker_folder/conda_requirements.txt` | `7c415cf7882c1e35` |
| `script/docker_folder/dockerfile_writing_instruction` | `6ab54be3f7192d2c` |
| `script/docker_folder/requirements.txt` | `799d43157e854d1a` |
| `script/make_change_in_all_file.sh` | `61446471b00ae018` |
| `script/multidcp_ae_pretrain.sh` | `143644ebdcb82b47` |
| `script/multidcp_ae_second.sh` | `6fafe442c1b99e03` |
| `script/train_deepce.sh` | `a9bcf8f1b51dd0ee` |
| `script/train_deepce_cell.sh` | `4c8817004f7c511a` |
| `script/train_multidcp_ae.sh` | `75844c6b5ef3d82c` |
| `script/train_multidcp_ae_infer.sh` | `e619aa054f752882` |
| `script/train_multidcp_ehill_finetune.sh` | `8c04c7cac3230858` |
| `script/train_multidcp_ehill_mt.sh` | `d3ae3b3a999955c6` |
| `script/train_multidcp_ehill_pretraining.sh` | `451fea0467bab948` |
| `script/train_multidcp_pretraining.sh` | `71992e8e0ebdb4d9` |
| `script/zenodo_upload.py` | `c1d4ded9c0853702` |

### 4.3 Top-level files

| File | SHA-256 (first 16) |
|---|---|
| `LICENSE` | `0ef7860efcb03e40` |

A total of 52 core files (`.py` model source + `script/` shell wrappers) are byte-equivalent to upstream commit `36ecdef9`. The complete public file set (including `utils/inchi_merck.csv`, `adr_prediction/adr_prediction.sh`, `make_change_in_all_file.sh`, etc.) is also byte-equivalent after excluding metadata. Reproduce via either of:

```bash
# Method A: directly compare the inner source tree (cleanest)
diff -rq reference/MultiDCP/MultiDCP/ models/MultiDCP/MultiDCP/

# Method B: compare from repository root, explicitly excluding metadata
diff -rq reference/MultiDCP/ models/MultiDCP/ \
  --exclude=README.md --exclude=.gitignore \
  --exclude=ORIGIN.md --exclude=CHANGES.md --exclude=CHANGES.diff
```

Method A produces empty output. Method B shows only `Only in reference/MultiDCP/: docs` and `Only in reference/MultiDCP/: trained_models` (both documented in §3).

---

## 5. Driver Behavior Summary

| Item | Status |
|---|---|
| Training driver | `experiments/_shared/training/train_multidcp.py` |
| Symbols imported from MultiDCP root | `multidcp.MultiDCP_AE` (class), `data_utils.create_mask_feature`, `molecules.{Molecules, Node, node_id, degrees}`, `molecule_utils.{atom_features, bond_features}` |
| Ablation implementation | Data layer (L190+) + forward hook (L588: `model.multidcp.drug_fp.register_forward_hook`); does not enter model source |
| `init_weights(pretrained=...)` | Driver calls `model.init_weights(pretrained=None)` (L575); no upstream `.pt` weights are loaded |

**Conclusion**: `models/MultiDCP/` is a vendored copy, byte-equivalent to upstream pristine `reference/MultiDCP/MultiDCP/`. The dpb production training driver `experiments/_shared/training/train_multidcp.py` performs `sys.path.insert` pointing to `models/MultiDCP/MultiDCP/` (the vendored copy that ships with the repo); `reference/MultiDCP/MultiDCP/` exists as the audit baseline and is populated by `bash reference/fetch_upstream.sh MultiDCP` when reviewers need it. The dpb driver handles training organization, data adaptation, unified evaluation output, and ablation logic (data layer + forward hook); the model source needs no modifications.
