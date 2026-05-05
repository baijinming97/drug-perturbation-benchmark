"""Micro-benchmark — CIGER (upstream code, original cold-drug split)."""
import os
import sys
import time
import warnings
import numpy as np
import torch
import random

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
# _profiling/ -> original_reproduction/ -> experiments/ -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
CIGER_DIR = os.path.join(REPO_ROOT, "reference", "CIGER", "CIGER")
sys.path.insert(0, CIGER_DIR)
sys.path.insert(0, os.path.join(CIGER_DIR, "models"))
sys.path.insert(0, os.path.join(CIGER_DIR, "utils"))

from models import CIGER
from utils import DataReader
from bench_common import run_bench_loop, WARMUP, TOTAL, DEVICE
SEED = 343
BATCH_SIZE = 64
FOLD = 0


def main():
    np.random.seed(seed=SEED)
    random.seed(a=SEED)
    torch.manual_seed(SEED)

    drug_file = os.path.join(CIGER_DIR, "data", "drug_smiles.csv")
    drug_id_file = os.path.join(CIGER_DIR, "data", "drug_id.csv")
    gene_file = os.path.join(CIGER_DIR, "data", "gene_feature.csv")
    data_file = os.path.join(CIGER_DIR, "data", "chemical_signature.csv")

    fp_type = "neural"
    label_type = "real"
    loss_type = "list_wise_rankcosine"

    data = DataReader(drug_file, drug_id_file, gene_file, data_file, fp_type, DEVICE, FOLD)

    model = CIGER(
        drug_input_dim={"atom": 62, "bond": 6},
        gene_embed=data.gene,
        gene_input_dim=data.gene.size()[1],
        encode_dim=512,
        fp_type=fp_type,
        loss_type=loss_type,
        label_type=label_type,
        device=DEVICE,
        initializer=None,
    )
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("Pre-fetching batches...")
    batches = []
    for batch in data.get_batch_data(dataset="train", batch_size=BATCH_SIZE, shuffle=True):
        batches.append(batch)
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        ft, lb = batches[idx]
        drug = ft["drug"]
        gene = ft["gene"]
        pert_type = ft.get("pert_type", None)
        cell_id = ft.get("cell_id", None)
        pert_idose = ft.get("pert_idose", None)
        label = lb["real"]

        optimizer.zero_grad()
        predict = model(drug, gene, pert_type, cell_id, pert_idose)
        loss = model.loss(label, predict)
        loss.backward()
        optimizer.step()

    run_bench_loop("CIGER", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
