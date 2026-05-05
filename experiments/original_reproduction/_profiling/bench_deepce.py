"""Micro-benchmark — DeepCE (upstream code, original cold-drug split)."""
import os
import sys
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
DEEPCE_DIR = os.path.join(REPO_ROOT, "reference", "DeepCE", "DeepCE")
sys.path.insert(0, DEEPCE_DIR)
sys.path.insert(0, os.path.join(DEEPCE_DIR, "models"))
sys.path.insert(0, os.path.join(DEEPCE_DIR, "utils"))

from models import DeepCE
from utils import DataReader
from bench_common import run_bench_loop, WARMUP, TOTAL, DEVICE
BATCH_SIZE = 16


def main():
    drug_file = os.path.join(DEEPCE_DIR, "data", "drugs_smiles.csv")
    gene_file = os.path.join(DEEPCE_DIR, "data", "gene_vector.csv")
    train_file = os.path.join(DEEPCE_DIR, "data", "signature_train.csv")
    dev_file = os.path.join(DEEPCE_DIR, "data", "signature_dev.csv")
    test_file = os.path.join(DEEPCE_DIR, "data", "signature_test.csv")

    DATA_FILTER = {
        "time": "24H",
        "pert_id": ["BRD-U41416256", "BRD-U60236422"],
        "pert_type": ["trt_cp"],
        "cell_id": ["A375", "A549", "HA1E", "HCC515", "HEPG2", "HT29", "MCF7", "PC3", "YAPC"],
        "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"],
    }

    data = DataReader(drug_file, gene_file, train_file, dev_file, test_file, DATA_FILTER, DEVICE)

    model = DeepCE(
        drug_input_dim={"atom": 62, "bond": 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_input_dim=np.shape(data.gene)[1],
        gene_emb_dim=128,
        num_gene=np.shape(data.gene)[0],
        hid_dim=128,
        dropout=0.1,
        loss_type="point_wise_mse",
        device=DEVICE,
        initializer=torch.nn.init.xavier_uniform_,
        pert_type_input_dim=len(DATA_FILTER["pert_type"]),
        cell_id_input_dim=len(DATA_FILTER["cell_id"]),
        pert_idose_input_dim=len(DATA_FILTER["pert_idose"]),
        pert_type_emb_dim=4,
        cell_id_emb_dim=4,
        pert_idose_emb_dim=4,
        use_pert_type=data.use_pert_type,
        use_cell_id=data.use_cell_id,
        use_pert_idose=data.use_pert_idose,
    )
    model.to(DEVICE)
    model = model.double()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

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
        mask = ft["mask"]
        pert_type = ft["pert_type"] if data.use_pert_type else None
        cell_id = ft["cell_id"] if data.use_cell_id else None
        pert_idose = ft["pert_idose"] if data.use_pert_idose else None

        optimizer.zero_grad()
        predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose)
        loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()

    run_bench_loop("DeepCE", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
