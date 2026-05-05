# Source Provenance

Upstream sources for every model and dataset shipped in this benchmark.
Per-model code modifications are documented in `models/<M>_CHANGES.md`.

License and redistribution caveats for vendored upstream code are centralized
in `NOTICE`.

---

## Models

### CIGER

- **Paper**: Pham et al., "Chemical-induced gene expression ranking and its application to pancreatic cancer drug repurposing." *Patterns* **3**(4): 100441 (2022). DOI: [10.1016/j.patter.2022.100441](https://doi.org/10.1016/j.patter.2022.100441)
- **Repository**: <https://github.com/pth1993/CIGER>
- **Commit**: `81c16f107cf957ca73840e6fab41de174c85ee8f`
- **Original data**: bundled in `reference/CIGER/CIGER/data/`, copied unchanged to `data/CIGER/`

### DeepCE

- **Paper**: Pham et al., "A deep learning framework for high-throughput mechanism-driven phenotype compound screening and its application to COVID-19 drug repurposing." *Nat Mach Intell* **3**, 247–257 (2021). DOI: [10.1038/s42256-020-00285-9](https://doi.org/10.1038/s42256-020-00285-9)
- **Repository**: <https://github.com/pth1993/DeepCE>
- **Commit**: `9b0d04fa920ef73df578b070cfcc3982effdde42`
- **Original data**: bundled in `reference/DeepCE/DeepCE/data/`, copied unchanged to `data/DeepCE/`

### MultiDCP

- **Paper**: Wu et al., "Deep learning prediction of chemical-induced dose-dependent and context-specific multiplex phenotype responses and its application to personalized Alzheimer's disease drug repurposing." *PLoS Comput Biol* **18**(8): e1010367 (2022). DOI: [10.1371/journal.pcbi.1010367](https://doi.org/10.1371/journal.pcbi.1010367)
- **Repository**: <https://github.com/XieResearchGroup/MultiDCP>
- **Commit**: `36ecdef9598de703b61fe0d1b91bcfeeba844274`
- **Original data**: Zenodo [10.5281/zenodo.5172809](https://doi.org/10.5281/zenodo.5172809), downloaded to `data/MultiDCP/`

### PertDiT

- **Paper**: Hu, Chen, Gu, "Predicting drug-perturbed transcriptional responses using multi-conditional diffusion transformer." *Quantitative Biology* 2026;e70016. DOI: [10.1002/qub2.70016](https://doi.org/10.1002/qub2.70016)
- **Repository**: <https://github.com/wangkekekeke/PertDiT>
- **Commit**: `596d681f816d3184d7a0a63a133ce68d5838fae0`
- **Original data**: Tsinghua Cloud direct download (~8.9 GB RAR archive), downloaded to `data/PertDiT/`. Extraction is handled inside each task's `prepare.py`.

### PRnet

- **Paper**: Qi et al., "Predicting transcriptional responses to novel chemical perturbations using deep generative model for drug discovery." *Nat Commun* **15**, 9256 (2024). DOI: [10.1038/s41467-024-53457-1](https://doi.org/10.1038/s41467-024-53457-1)
- **Repository**: <https://github.com/Perturbation-Response-Prediction/PRnet>
- **Commit**: `f19174bde3ed2633f54c7831799cc38c4ffc7a0d`
- **Original data**: Zenodo [10.5281/zenodo.14230870](https://doi.org/10.5281/zenodo.14230870) (~4 GB), downloaded to `data/PRnet/`

### TranSiGen

- **Paper**: Tong et al., "Deep representation learning of chemical-induced transcriptional profile for phenotype-based drug discovery." *Nat Commun* **15**(1): 5378 (2024). DOI: [10.1038/s41467-024-49620-3](https://doi.org/10.1038/s41467-024-49620-3)
- **Repository**: <https://github.com/myzhengSIMM/TranSiGen>
- **Commit**: `8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a`
- **Original data**: upstream [Release v1.0](https://github.com/myzhengSIMM/TranSiGen/releases/tag/v1.0), downloaded to `data/TranSiGen/LINCS2020/`:
    - `processed_data.h5` — 652,048,227 B, sha256 `470109993fb8abdf2a200cb08af97aa9a0ab83542bb5c3f4aa0ea361b04b354e`
    - `KPGT_emb2304.pickle` — 77,663,320 B, sha256 `61b2a1337c8313ecbcb95e9f797ea12caeb5d94fefc2697af810c5955ae962f4`

  Plus the demo subset (`data_example/`, `idx2smi.pickle`, etc.) bundled in `reference/TranSiGen/data/`, copied unchanged to `data/TranSiGen/`.

### XPert

- **Paper**: Guo et al., "Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer." *Nat Mach Intell* (2026). DOI: [10.1038/s42256-025-01165-w](https://doi.org/10.1038/s42256-025-01165-w)
- **Repository**: <https://github.com/GSanShui/XPert>
- **Commit**: `d53ff497e465692c80fac7b899e3cc5cc8e7bfed`
- **Original data**: Zenodo [10.5281/zenodo.15357711](https://doi.org/10.5281/zenodo.15357711) (~14 GB) + Figshare [10.6084/m9.figshare.28955141](https://doi.org/10.6084/m9.figshare.28955141), downloaded to `data/XPert/`

---

## Reproducing the fetch step

The exact commands used to fetch each repository and dataset are encoded in
`reference/fetch_upstream.sh` (idempotent — re-running skips anything already
present). See the **Data** section of `README.md` for usage.
