"""Micro-benchmark — MultiDCP (upstream code, original cold-cell split, perturbed track)."""
import os
import sys
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "disabled"

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
MULTIDCP_DIR = os.path.join(REPO_ROOT, "reference", "MultiDCP", "MultiDCP")
sys.path.insert(0, MULTIDCP_DIR)
sys.path.insert(0, os.path.join(MULTIDCP_DIR, "models"))
sys.path.insert(0, os.path.join(MULTIDCP_DIR, "utils"))

from models import multidcp
import datareader
from collections import defaultdict
from bench_common import run_bench_loop, WARMUP, TOTAL, DEVICE
BATCH_SIZE = 64


def main():
    DATA_FILTER = {
        "time": "24H",
        "pert_id": ["BRD-U41416256", "BRD-U60236422"],
        "pert_type": ["trt_cp"],
        "cell_id": ["A375", "A549", "HA1E", "HCC515", "HEPG2", "HT29", "MCF7", "PC3", "YAPC"],
        "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"],
    }

    # MultiDCP's bundled data is shipped via tarball; prepare.py extracts it
    # into data/MultiDCP/extracted/data/. Fall back to the upstream-clone copy
    # if the tarball hasn't been extracted yet.
    EXTRACTED = os.path.join(REPO_ROOT, "data", "MultiDCP", "extracted", "data")
    DATA_BASE = EXTRACTED if os.path.isdir(EXTRACTED) else os.path.join(MULTIDCP_DIR, "data")

    class Args:
        drug_file = os.path.join(DATA_BASE, "all_drugs_l1000.csv")
        gene_file = os.path.join(DATA_BASE, "gene_vector.csv")
        train_file = os.path.join(DATA_BASE, "pert_transcriptom", "signature_train_cell_2.csv")
        dev_file = os.path.join(DATA_BASE, "pert_transcriptom", "signature_dev_cell_2.csv")
        test_file = os.path.join(DATA_BASE, "pert_transcriptom", "signature_test_cell_2.csv")
        cell_ge_file = os.path.join(DATA_BASE, "adjusted_ccle_tcga_ad_tpm_log2.csv")
        linear_encoder_flag = True
        batch_size = BATCH_SIZE

    args = Args()

    data = datareader.PerturbedDataLoader(DATA_FILTER, DEVICE, args)
    data.setup()
    print(f"Train: {len(data.train_data)}, Dev: {len(data.dev_data)}, Test: {len(data.test_data)}")

    model_param_registry = defaultdict(
        drug_input_dim={"atom": 62, "bond": 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_emb_dim=128,
        gene_input_dim=128,
        cell_id_input_dim=978,
        cell_feature_emb_dim=32,
        pert_idose_emb_dim=4,
        hid_dim=128,
        num_gene=978,
        loss_type="point_wise_mse",
        initializer=torch.nn.init.kaiming_uniform_,
    )
    model_param_registry.update({
        "num_gene": np.shape(data.gene)[0],
        "pert_idose_input_dim": len(DATA_FILTER["pert_idose"]),
        "dropout": 0.1,
        "linear_encoder_flag": args.linear_encoder_flag,
    })

    model = multidcp.MultiDCPEhillPretraining(device=DEVICE, model_param_registry=model_param_registry)
    model = model.double()
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    print("Pre-fetching batches...")
    batches = []
    for i, (ft, lb, cell_type) in enumerate(data.train_dataloader()):
        batches.append((ft, lb, cell_type))
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        ft, lb, cell_type = batches[idx]
        drug = ft["drug"]
        mask = ft["mask"]
        cell_feature = ft["cell_id"]
        pert_idose = ft["pert_idose"]

        optimizer.zero_grad()
        predict, cell_hidden_ = model(
            drug, data.gene.to(DEVICE), mask, cell_feature, pert_idose,
            job_id="perturbed", epoch=0,
        )
        loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()

    run_bench_loop("MultiDCP", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
