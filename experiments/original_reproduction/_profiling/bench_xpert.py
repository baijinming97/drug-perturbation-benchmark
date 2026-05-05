"""Micro-benchmark — XPert (vendored code, AMP)."""
import os
import sys
import time
import warnings
import logging
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
import yaml
import scanpy as sc

warnings.filterwarnings("ignore")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH_DIR)))
XPERT_ROOT = os.path.join(REPO_ROOT, "models", "XPert")
sys.path.insert(0, XPERT_ROOT)

from utils import mse_loss_ls_sum, pcc_loss_sum, load_dataloader
from models.model_XPert import XPertNet
from bench_common import run_bench_loop, WARMUP, TOTAL, DEVICE
SEED = 2024


class Args:
    mode = "train"
    nfold = "split_1"
    drug_feat = "unimol"
    device = str(DEVICE)
    model = "XPert"
    config = "config_l1000"
    seed = SEED
    dataset = "l1000_sdst"
    pretrained_mode = "global"
    include_cell_idx = False
    wo_HG = False
    wo_atom = False
    wo_atom_HG = False
    wo_unimol = False
    wo_ppi = False
    use_gene_pos_emed = False
    use_gradscaler = True
    lr_scheduler = False
    resume_from = None
    saved_model_path = None
    saved_model = None
    pretrained_model = None
    output_profile = False
    output_attention = False
    output_cls_embed = False
    weighted_loss = False
    kl_loss = False
    expt_dir = None


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    args = Args()
    logger = logging.getLogger("bench_xpert")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    orig_cwd = os.getcwd()
    os.chdir(XPERT_ROOT)

    config_path = os.path.join(XPERT_ROOT, "configs", "config_l1000.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tr_dataloader, _, _ = load_dataloader(args, config, logger, args.nfold)

    model = XPertNet(args, config, DEVICE, logger)
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["train_lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    a, b, c, d = config["train"]["loss_weight"]
    init_epoch = config["train"]["init_epoch"]

    os.chdir(orig_cwd)

    print("Pre-fetching batches...")
    batches = []
    for batch in tr_dataloader:
        batches.append(batch)
        if len(batches) >= TOTAL + 5:
            break
    print(f"Pre-fetched {len(batches)} batches")

    def step_fn(i):
        idx = i % len(batches)
        data = batches[idx]

        model.zero_grad()
        with autocast(enabled=use_amp):
            trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, _, cell_class_true, cell_class_predict = model(data)
            deg_true = trt_raw_data - ctl_raw_data
            num_samples = trt_raw_data.shape[0]

            loss1 = mse_loss_ls_sum(trt_output, trt_raw_data)
            if cell_class_predict is not None:
                cell_class_predict_1, cell_class_predict_2 = cell_class_predict
                MultiClassLoss = torch.nn.CrossEntropyLoss(reduction="sum")
                loss2 = MultiClassLoss(cell_class_predict_1, cell_class_true) + MultiClassLoss(cell_class_predict_2, cell_class_true)
            else:
                loss2 = mse_loss_ls_sum(ctl_output, ctl_raw_data)
            loss3 = mse_loss_ls_sum(deg_output, deg_true)
            loss4 = pcc_loss_sum(deg_output, deg_true)

            batch_weighted_loss = (
                torch.sqrt(loss1 / num_samples) * a
                + torch.sqrt(loss2 / num_samples) * b
                + torch.sqrt(loss3 / num_samples) * c
                + (loss4 / num_samples) * d
            )

        scaler.scale(batch_weighted_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    run_bench_loop("XPert", model, step_fn, num_batches=len(batches))


if __name__ == "__main__":
    main()
