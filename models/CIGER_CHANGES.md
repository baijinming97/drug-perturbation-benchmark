# CIGER — Modification Audit

**Upstream**: `https://github.com/pth1993/CIGER` @ commit `81c16f107cf957ca73840e6fab41de174c85ee8f`

**Audit scope**: This audit uses the pinned upstream snapshot in `reference/` as the comparison baseline, examining vendored model source against the import paths actually exercised by production drivers. The comparison **excludes** all `README.md` files (both upstream and local), `.gitignore`, and dpb metadata files (`CHANGES.md`, `CHANGES.diff`, `ORIGIN.md`); source equivalence applies to upstream model code, scripts, configs, and license files only.

---

## 1. Change Overview

| File | Hunks | Nature |
|---|---|---|
| (none) | — | No source modifications |

All source files under `models/CIGER/` are **byte-equivalent** to the upstream pinned commit. This directory is a faithful copy of upstream, introducing no code-level modifications. The only difference is **directory flattening** (see §3).

---

## 2. Line-level Hunks

None.

---

## 3. File Additions/Removals

### Upstream-only (not in this directory)

| File | Category | Notes |
|---|---|---|
| `docs/` | Documentation | README figures only |
| `data/chemical_signature.csv` | Training data | Upstream-bundled training data; in dpb located at top-level `data/CIGER/` (provided by `reference/fetch_upstream.sh CIGER`; data file equivalence is not part of this SHA table — separate data checksums apply if needed) |
| `data/drug_ecfp.csv` | Training data | Same as above |
| `data/drug_id.csv` | Training data | Same as above |
| `data/drug_smiles.csv` | Training data | Same as above |
| `data/gene_feature.csv` | Training data | Upstream-bundled training data; in dpb located at top-level `data/CIGER/gene_feature.csv` (provided by fetch). In addition, `experiments/_shared/prepare_steps.py` derives `data/CIGER/gene_feature_nm.csv`, mapping 5 deprecated gene names to current HGNC symbols (`KIAA0907→KHDC4`, `PAPD7→TENT4A`, `IKBKAP→ELP1`, `TMEM5→RXYLT1`, `HDGFRP3→HDGFL3`). Production drivers use this nm version to align with the dpb common gene coordinate system, so **CIGER's `gene_feature_nm.csv` on the production path is not byte-equivalent to upstream `gene_feature.csv`** |
| `data/pancreatic_cancer_signature.csv` | Training data | Same as above |
| `drug_repurposing/drugbank_drug_id.csv` (92 KB) | Application data | Upstream drug-repurposing case-study data; not read by the production training path |
| `drug_repurposing/enrichment_score_down.csv` (2.1 MB) | Application data | Same as above |
| `drug_repurposing/enrichment_score_up.csv` (2.1 MB) | Application data | Same as above |
| `drug_repurposing/precision_score.npy` | Application data | Same as above |

### Local-only (not in upstream)

None — `models/CIGER/` adds no extra files to the upstream tree.

### Structural difference (directory flattening)

Upstream wraps all source code in a redundant inner `CIGER/` directory:

```
upstream:  CIGER/             (git repo root)
           └── CIGER/         (inner wrapper)
               ├── models/
               ├── utils/
               ├── train.py
               └── ...

this dir:  models/CIGER/
           ├── models/         (flattened, no inner wrapper)
           ├── utils/
           ├── train.py
           └── ...
```

File contents are byte-equivalent; only one level of directory nesting is removed.

### LICENSE status

⚠ The upstream CIGER repository contains no LICENSE file at the pinned commit, and the README contains no license declaration. This directory `models/CIGER/` likewise carries no LICENSE. The top-level `NOTICE` documents this — vendoring proceeds under full attribution in `PROVENANCE.md`, but redistribution rights are not explicitly granted by upstream; commercial use requires contacting the upstream author (Thai-Hoang Pham, Ohio State University).

---

## 4. SHA-256 Verification (all 19 files, byte-equivalent)

| File | SHA-256 (first 16) |
|---|---|
| `drug_repurposing/drug_screening_gsea.py` | `563cae4cb5a9a821` |
| `drug_repurposing/drug_screening_precision.py` | `ea82fba573ca6f56` |
| `models/__init__.py` | `4aa7241a095bad6c` |
| `models/attention.py` | `25a4cef102760ba5` |
| `models/ciger.py` | `bdd441cbedd81a2b` |
| `models/graph_degree_conv.py` | `7e64f0e3fa77217a` |
| `models/loss_utils.py` | `75dd05593116d729` |
| `models/ltr_loss.py` | `27c6b9259a1ff50c` |
| `models/multi_head_attention.py` | `bc5663b395533ae2` |
| `models/neural_fingerprint.py` | `6e2ca1de05218f06` |
| `models/positionwide_feedforward.py` | `48a18c5de451b70e` |
| `train.py` | `067b346f4456ac31` |
| `train.sh` | `9b5099bef1bfab05` |
| `utils/__init__.py` | `27b4e25f8d05034f` |
| `utils/data_utils.py` | `a2ca01578fc7f9e5` |
| `utils/datareader.py` | `427d1baaea8b2a4c` |
| `utils/metric.py` | `b631eb2d84a0311c` |
| `utils/molecule_utils.py` | `6980139aed13cc2e` |
| `utils/molecules.py` | `1810585717b558d7` |

All 19 files byte-equivalent to upstream commit `81c16f1` (ignoring directory nesting).

---

## 5. Driver Behavior Summary

| Item | Status |
|---|---|
| Training driver | `experiments/_shared/training/train_ciger.py` |
| Symbols imported from CIGER root | `models.CIGER` (class), `utils.data_utils.{read_gene, convert_smile_to_feature, create_mask_feature}`, `utils.metric.ndcg` |
| Ablation implementation | Data-layer batch shuffle (shuffle) + forward hook (zero); does not enter model source |
| Current `sys.path.insert` target | `models/CIGER/` (vendored production copy, ships with the repo) |

**Note on sys.path target**: The driver imports CIGER source from `models/CIGER/`, which is **byte-equivalent** to upstream `reference/CIGER/CIGER/` (proven by §4 SHA table; the two differ only by a directory-flattening step described in §3). `reference/CIGER/CIGER/` is populated separately by `bash reference/fetch_upstream.sh CIGER` when reviewers want a pristine baseline for audit, but the production training path does **not** require running fetch_upstream.

**Conclusion**: The CIGER model architecture source is byte-equivalent to upstream commit `81c16f1` (after directory flattening). Production training uses the dpb-custom driver `experiments/_shared/training/train_ciger.py`, whose `sys.path.insert` points to `models/CIGER/` (the vendored copy that ships with the repo). The driver handles training organization, data format adaptation, unified evaluation output, and ablation logic. **The production path uses `gene_feature_nm.csv` after 5 HGNC name remappings; therefore the model architecture code can be claimed byte-equivalent to upstream, but the full training experiment is not equivalent to running upstream's training entrypoint directly**.
