"""Micro-benchmark — TranSiGen (upstream code, original cold-drug split)."""
import os
import sys
import time
import warnings
import pickle
import numpy as np
import torch
from torch import optim

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
TRANSIGEN_SRC = os.path.join(REPO_ROOT, "reference", "TranSiGen", "src")
sys.path.insert(0, TRANSIGEN_SRC)

from dataset import TranSiGenDataset
from model import TranSiGen
from utils import load_from_HDF, split_data, setup_seed, seed_worker
from bench_common import run_bench_loop, save_results, reset_gpu, WARMUP, TOTAL, DEVICE

DATA_DIR = os.path.join(REPO_ROOT, "data", "TranSiGen")
SEED = 364039
BATCH_SIZE = 64


def main():
    setup_seed(SEED)

    # Upstream's bench used `processed_data_id.h5`; dpb's fetch_upstream
    # downloads `processed_data.h5`. Accept either.
    candidates = [
        os.path.join(DATA_DIR, "LINCS2020", "processed_data.h5"),
        os.path.join(DATA_DIR, "LINCS2020", "processed_data_id.h5"),
    ]
    data_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    data = load_from_HDF(data_path)
    print(f"Loaded {len(data['canonical_smiles'])} samples from {data_path}")

    pair, pairv, pairt = split_data(data, n_folds=5, split_type="smiles_split", rnds=SEED)
    print(f"Train: {len(pair['canonical_smiles'])}, Valid: {len(pairv['canonical_smiles'])}")

    orig_cwd = os.getcwd()
    os.chdir(TRANSIGEN_SRC)
    train_ds = TranSiGenDataset(
        LINCS_index=pair["LINCS_index"],
        mol_feature_type="KPGT",
        mol_id=pair["canonical_smiles"],
        cid=pair["cid"],
    )
    os.chdir(orig_cwd)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=False, num_workers=4, worker_init_fn=seed_worker,
    )

    model = TranSiGen(
        n_genes=978, n_latent=100, n_en_hidden=[1200], n_de_hidden=[800],
        features_dim=2304, features_embed_dim=[400],
        init_w=True, beta=0.1, device=DEVICE, dropout=0.1,
        path_model="/tmp/transigen_bench/", random_seed=SEED,
    )
    model.to(DEVICE)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("Pre-fetching batches...")
    batches = []
    for batch in train_loader:
        batches.append(batch)
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        x1, x2, features, mol_id, cid, sig = batches[idx]
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        features = features.to(DEVICE)

        optimizer.zero_grad()
        x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, z2_pred = model.forward(x1, features)
        z2, mu2, logvar2 = model.encode_x2(x2)
        x2_rec = model.decode_x2(z2)
        loss, _, _, _, _, _, _ = model.loss(
            x1, x1_rec, mu1, logvar1,
            x2, x2_rec, mu2, logvar2,
            x2_pert, mu_pred, logvar_pred,
        )
        loss.backward()
        optimizer.step()

    run_bench_loop("TranSiGen", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
