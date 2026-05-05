# Environments

A single conda env (**`benchmark`**) hosts all 7 training scripts:
CIGER, DeepCE, MultiDCP, PertDiT, PRnet, TranSiGen, XPert (+ MLP baseline).

## Files in this directory

| File | Purpose |
| --- | --- |
| `environment.yml` | Conda spec: Python 3.9 base layer (no ML deps) |
| `install_pip.sh` | Pip install steps in the right order (torch → ext → flash-attn → unimol) |
| `pip_freeze.txt` | Snapshot of `pip freeze` from the reference env (117 packages) |

## Quick start

```bash
conda env create -f environment/environment.yml      # creates "benchmark"
conda activate benchmark
bash environment/install_pip.sh                      # installs ML stack
```

`install_pip.sh` self-activates the `benchmark` env's pip, so you do not have
to `conda activate` first. Order matters: torch is installed before
flash-attn / torch-geometric extensions / unimol-tools.

Verify the install:

```bash
for M in mlp ciger deepce multidcp pertdit prnet transigen; do
    python experiments/_shared/training/train_$M.py --help > /dev/null && echo "$M ✓" || echo "$M ✗"
done
```

All 7 should print `✓`. Final env size ≈ 5.7 GB.

## Pinned versions

```
Python   3.9.23
torch    2.1.0+cu121      torchvision 0.16.0+cu121      torchaudio 2.1.0+cu121
torch-geometric 2.6.1     torch-scatter 2.1.2+pt21cu121 torch-sparse 0.6.18+pt21cu121
pyg-lib  0.4.0+pt21cu121
flash-attn 2.6.0.post1    unimol-tools 0.1.4.post1
diffusers 0.30.2          einops 0.8.2                  torchmetrics 1.6.0
pytorch-lightning 1.9.5
scanpy   1.9.8            anndata 0.10.9                rdkit 2024.3.2
numpy    1.26.4           pandas 2.3.0                  scipy 1.13.1
```

Full list in `pip_freeze.txt`.

## Known footguns

1. **`flash-attn`** must be installed `--no-build-isolation` (already handled
   in `install_pip.sh`); otherwise pip rebuilds torch in a fresh isolated
   environment and the build hangs.
2. **`unimol-tools`** downgrades pandas to 1.5.3 during install; the script
   re-pins to 2.3.0 afterwards. Verified harmless because no downstream
   code needs the older API.
3. **`diffusers` 0.36+** requires `torch.xpu`, which is absent in torch 2.1.
   Pinned to `diffusers==0.30.2`.
4. The conda env intentionally ships only Python 3.9; all CUDA bits come
   from the torch wheel (cu121 runtime), not from `cudatoolkit=12.1` on
   conda channels.

## Optional packages (not installed by default)

The following are imported by sub-scripts that the **training path does
not trigger**. Install on demand if you need them:

| Package | Imported by |
| --- | --- |
| `wandb` | MultiDCP `ehill_*`, DeepCE `main_deepce.py` |
| `apscheduler` | MultiDCP `utils/drug_response_curve_cal.py` |
| `cmapPy` | TranSiGen `prediction.py` |
| `allrank` | MultiDCP `pretrain_multidcp.py` (`pip install git+https://github.com/allegro/allRank.git`) |
