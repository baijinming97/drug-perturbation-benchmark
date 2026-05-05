"""Micro-benchmark — drug-blind MLP baseline (3-layer, 2048 hidden)."""
import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import scanpy as sc

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))

from bench_common import run_bench_loop, WARMUP, TOTAL, DEVICE
BATCH_SIZE = 512
SEED = 2024


class MLP(nn.Module):
    def __init__(self, input_dim=978, hidden_dim=2048, output_dim=978, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_path = os.path.join(REPO_ROOT, "data", "XPert", "processed_data", "l1000_sdst_78453.h5ad")
    print(f"Loading {data_path}...")
    data = sc.read_h5ad(data_path)

    # Use the first available cold-drug split column. The XPert h5ad ships
    # with `split_1`...`split_5`; dpb's prepare.py adds `nm_drug_blind_1`...
    split_col = next((c for c in ["split_1", "nm_drug_blind_1"] if c in data.obs.columns), None)
    if split_col is None:
        raise RuntimeError(
            f"No drug-blind split column in {data_path}; tried split_1 / nm_drug_blind_1."
        )
    tr_data = data[data.obs[split_col] == "train"]
    print(f"Split: {split_col} | Train samples: {tr_data.n_obs}")

    trt_expr = torch.tensor(tr_data.X.toarray() if hasattr(tr_data.X, "toarray") else np.array(tr_data.X), dtype=torch.float32)
    ctl_expr = torch.tensor(tr_data.obsm["X_ctl"].toarray() if hasattr(tr_data.obsm["X_ctl"], "toarray") else np.array(tr_data.obsm["X_ctl"]), dtype=torch.float32)
    deg_true = trt_expr - ctl_expr

    dataset = torch.utils.data.TensorDataset(ctl_expr, deg_true)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = MLP(input_dim=978, hidden_dim=2048, output_dim=978)
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Pre-fetching batches...")
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        ctl, deg = batches[idx]
        ctl = ctl.to(DEVICE)
        deg = deg.to(DEVICE)

        optimizer.zero_grad()
        pred = model(ctl)
        loss = criterion(pred, deg)
        loss.backward()
        optimizer.step()

    run_bench_loop("MLP", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
