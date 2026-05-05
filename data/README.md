# data/

This directory hosts each model's original dataset, organised one folder per
model: `data/<M>/`. Per-model contents (download URLs, file lists, SHA256
checksums) are described in the local `data/<M>/README.md`.

The download/setup is performed by `reference/fetch_upstream.sh`, which fetches
from:

| Source | Models |
|--------|--------|
| Upstream repo bundled data | CIGER, DeepCE, TranSiGen demo subset |
| Zenodo `10.5281/zenodo.5172809` | MultiDCP |
| Zenodo `10.5281/zenodo.14230870` | PRnet |
| Zenodo `10.5281/zenodo.15357711` (concept DOI) + Figshare `28955141` | XPert |
| Tsinghua Cloud (PertDiT release) | PertDiT |
| TranSiGen GitHub release `v1.0` | TranSiGen full training assets |

These archives are NOT redistributed in this repository. To populate this
directory yourself, run `bash reference/fetch_upstream.sh`.
