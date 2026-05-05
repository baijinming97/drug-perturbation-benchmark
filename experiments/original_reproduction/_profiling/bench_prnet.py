"""Micro-benchmark — PRnet (upstream code, original cold-drug split)."""
import os
import sys
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
PRNET_DIR = os.path.join(REPO_ROOT, "reference", "PRnet")
PRNET_DATA = os.path.join(REPO_ROOT, "data", "PRnet")
sys.path.insert(0, PRNET_DIR)

import scanpy as sc
from trainer.PRnetTrainer import PRnetTrainer
from bench_common import run_bench_loop, save_results, reset_gpu, WARMUP, TOTAL, DEVICE
BATCH_SIZE = 512
SPLIT_KEY = "drug_split_0"


def main():
    h5ad_path = os.path.join(PRNET_DATA, "Lincs_L1000.h5ad")
    print(f"Loading {h5ad_path}...")
    adata = sc.read(h5ad_path)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    print(f"Loaded {adata.shape[0]} samples")

    config_kwargs = {
        "batch_size": BATCH_SIZE,
        "comb_num": 1,
        "save_dir": "/tmp/prnet_bench/",
        "n_epochs": 500,
        "split_key": SPLIT_KEY,
        "x_dimension": 978,
        "hidden_layer_sizes": [128],
        "z_dimension": 64,
        "adaptor_layer_sizes": [128],
        "comb_dimension": 64,
        "drug_dimension": 1024,
        "dr_rate": 0.05,
        "lr": 1e-3,
        "weight_decay": 1e-8,
        "scheduler_factor": 0.5,
        "scheduler_patience": 10,
        "n_genes": 20,
        "loss": ["GUSS"],
        "obs_key": "cov_drug_name",
    }

    trainer = PRnetTrainer(
        adata,
        batch_size=config_kwargs["batch_size"],
        comb_num=config_kwargs["comb_num"],
        split_key=config_kwargs["split_key"],
        model_save_dir=config_kwargs["save_dir"],
        x_dimension=config_kwargs["x_dimension"],
        hidden_layer_sizes=config_kwargs["hidden_layer_sizes"],
        z_dimension=config_kwargs["z_dimension"],
        adaptor_layer_sizes=config_kwargs["adaptor_layer_sizes"],
        comb_dimension=config_kwargs["comb_dimension"],
        drug_dimension=config_kwargs["drug_dimension"],
        dr_rate=config_kwargs["dr_rate"],
        n_genes=config_kwargs["n_genes"],
        loss=config_kwargs["loss"],
        obs_key=config_kwargs["obs_key"],
    )

    model = trainer.modelPGM
    if hasattr(model, "module"):
        raw_model = model.module
    else:
        raw_model = model
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config_kwargs["lr"],
        weight_decay=config_kwargs["weight_decay"],
    )

    criterion = trainer.criterion

    print("Pre-fetching batches...")
    batches = []
    for i, data in enumerate(trainer.train_dataloader):
        batches.append(data)
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        data = batches[idx]
        (control, target) = data["features"]
        encode_label = data["label"]

        control = control.to(DEVICE, dtype=torch.float32)
        target = target.to(DEVICE, dtype=torch.float32)
        encode_label = encode_label.to(DEVICE, dtype=torch.float32)

        noise = torch.randn(control.shape[0], 10).to(DEVICE)

        model.zero_grad()
        gene_reconstructions = model(control, encode_label, noise)
        dim = gene_reconstructions.size(1) // 2
        gene_means = gene_reconstructions[:, :dim]
        gene_vars = torch.nn.functional.softplus(gene_reconstructions[:, dim:])
        reconstruction_loss = criterion(input=gene_means, target=target, var=gene_vars)
        reconstruction_loss.backward()
        optimizer.step()

    run_bench_loop("PRnet", raw_model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
