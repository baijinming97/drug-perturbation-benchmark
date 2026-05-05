"""Micro-benchmark — PertDiT (upstream code, original cold-drug split)."""
import os
import sys
import time
import warnings
import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
PERTDIT_SRC = os.path.join(REPO_ROOT, "reference", "PertDiT", "src")
PERTDIT_DATA = os.path.join(REPO_ROOT, "data", "PertDiT", "extracted")
sys.path.insert(0, PERTDIT_SRC)

import scanpy as sc
from model.model_factory import Choose_model
from sampler.Sampler import Diffusion_Sampler
from trainer.lossfunc_and_generator import choose_loss_generator
from trainer.optimizer import get_optimizer_scheduler
from dataset.my_Dataset import Choose_dataset_loader
from utils.utils import train_valid_test
from bench_common import run_bench_loop, WARMUP, TOTAL, DEVICE
BATCH_SIZE = 64


def main():
    config_path = os.path.join(PERTDIT_SRC, "config", "Ada.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["train"]["split"] = "drug_split_0"
    config["device"] = str(DEVICE)

    print("Loading PertDiT data...")
    pert_smiles_emb = torch.load(
        os.path.join(PERTDIT_DATA, "pert_smiles_emb.pkl"),
        map_location=DEVICE,
    )
    dosage_prompt_emb = torch.load(
        os.path.join(PERTDIT_DATA, "dosage_prompt_emb_lincs.pkl"),
        map_location=DEVICE,
    )
    lincs_adata = sc.read(os.path.join(PERTDIT_DATA, "lincs_adata.h5ad"))
    lincs_adata.X = np.clip(lincs_adata.X, 0, 1e3)
    sc.pp.normalize_total(lincs_adata)
    sc.pp.log1p(lincs_adata)

    train_adata, valid_adata, test_adata = train_valid_test(
        lincs_adata, split_key=config["train"]["split"]
    )
    print(f"Train: {len(train_adata)}, Valid: {len(valid_adata)}, Test: {len(test_adata)}")

    myDataset_dosage, myDataloader = Choose_dataset_loader(config["drug_encoder"], DEVICE)
    train_dataset = myDataset_dosage(
        train_adata, pert_smiles_emb, dosage_prompt_emb, DEVICE,
        cfg=config["using_cfg"], cfg_prob=config["diffusion"]["cfg_prob"],
        FC=config["using_FC"],
    )
    train_loader = myDataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Choose_model(config)
    model.init_weights()
    model = model.to(DEVICE)
    model.train()

    sampler = Diffusion_Sampler(
        sampler_type=config["diffusion"]["sampler_type"],
        num_train_timesteps=config["diffusion"]["train_steps"],
        timesteps=config["diffusion"]["num_steps"],
        start=config["diffusion"]["beta_start"],
        end=config["diffusion"]["beta_end"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        device=DEVICE,
        guidance_scale=config["diffusion"]["guidance_scale"],
        uncond=None,
        loss_ratio=None,
    )

    loss_func, _ = choose_loss_generator(config)

    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        config["scheduler"]["lr_max"],
        config["scheduler"]["warmup_n_steps"],
        config["scheduler"]["lr_start"],
        config["scheduler"]["T_max"],
        config["scheduler"]["lr_min"],
    )

    print("Pre-fetching batches...")
    batches = []
    for batch in train_loader:
        batches.append(batch)
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        batch = batches[idx]
        loss = loss_func(sampler, model, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    run_bench_loop("PertDiT", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
