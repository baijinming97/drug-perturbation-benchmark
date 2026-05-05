"""
Task 1: Checkpoint sweep for random_split_0
Evaluate best and latest checkpoints on BOTH valid and test sets.
Also evaluate training-time validation metrics from saved mse_predict.
Since only best/latest checkpoints are saved, we extract training-curve info
from the checkpoint's stored mse_predict list and from the training log.
"""
import os, sys, yaml, torch, logging
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from scipy.stats import pearsonr
from utils.seed_everything import seed_everything
from utils.utils import train_valid_test, calculate_correlation_coefficients, mse, cal_r2
from dataset.my_Dataset import Choose_dataset_loader
from model.model_factory import Choose_model
from trainer.lossfunc_and_generator import choose_loss_generator
from sampler.Sampler import Diffusion_Sampler
import torch.nn as nn

SPLIT = "random_split_0"
RESULT_DIR = f"data/result/{SPLIT}/CrossDiT_repro_s117"
OUTPUT_CSV = f"data/result/{SPLIT}/checkpoint_sweep.csv"

def eval_on_dataset(model, generator_func, sampler, loader, limit, drug_adata_obs, relu, scale_params=None):
    """Evaluate model on a loader, return dict of metrics."""
    model.eval()
    total_y_pred = torch.Tensor()
    total_y_true = torch.Tensor()
    total_x = torch.Tensor()
    total_mse_list = torch.Tensor()
    total_r2_list = torch.Tensor()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=limit, desc="eval"):
            if i == limit:
                break
            y_pred = generator_func(sampler, model, batch)
            y_true = batch[0].cpu()
            x = batch[1].cpu()
            y_pred = relu(y_pred)
            total_y_pred = torch.cat((total_y_pred, y_pred), 0)
            total_y_true = torch.cat((total_y_true, y_true), 0)
            total_x = torch.cat((total_x, x), 0)
            total_mse_list = torch.cat((total_mse_list, mse(y_true, y_pred)), 0)
            total_r2_list = torch.cat((total_r2_list, cal_r2(y_true, y_pred)), 0)

    avg_mse = total_mse_list.mean().item()
    avg_r2 = total_r2_list.mean().item()

    # per-sample FC PCC
    fc_true = (total_y_true - total_x).numpy()
    fc_pred = (total_y_pred - total_x).numpy()
    sample_pccs = []
    for i in range(min(5000, fc_true.shape[0])):
        r, _ = pearsonr(fc_true[i], fc_pred[i])
        if not np.isnan(r):
            sample_pccs.append(r)
    per_sample_fc_pcc = np.mean(sample_pccs)

    # drug-level FC PCC
    coeff_drug, _ = calculate_correlation_coefficients(drug_adata_obs, 'condition', total_x, total_y_true, total_y_pred)
    drug_fc_pcc = coeff_drug.mean()

    coeff_cov, _ = calculate_correlation_coefficients(drug_adata_obs, 'cov_drug_name', total_x, total_y_true, total_y_pred)
    cov_fc_pcc = coeff_cov.mean()

    # slope(y_pred vs x) — shrinkage indicator
    slopes = []
    for i in range(min(2000, total_y_pred.shape[0])):
        coeffs = np.polyfit(total_x[i].numpy(), total_y_pred[i].numpy(), 1)
        slopes.append(coeffs[0])
    avg_slope = np.mean(slopes)

    return {
        'MSE': avg_mse, 'R2': avg_r2,
        'per_sample_FC_PCC': per_sample_fc_pcc,
        'drug_FC_PCC': drug_fc_pcc,
        'cov_drug_FC_PCC': cov_fc_pcc,
        'slope_ypred_vs_x': avg_slope
    }

def main():
    seed_everything(117)
    config_path = f"{RESULT_DIR}/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load data
    print("Loading data...")
    pert_smiles_emb = torch.load('data/pert_smiles_emb.pkl')
    dosage_prompt_emb = torch.load('data/dosage_prompt_emb_lincs.pkl')
    lincs_adata = sc.read('data/lincs_adata.h5ad')
    lincs_adata.X = np.clip(lincs_adata.X, 0, 1e3)
    sc.pp.normalize_total(lincs_adata)
    sc.pp.log1p(lincs_adata)

    train_adata, valid_adata, test_adata = train_valid_test(lincs_adata, split_key=SPLIT)
    device = torch.device(config['device'])
    myDataset, myDataloader = Choose_dataset_loader(config['drug_encoder'], device)
    valid_dataset = myDataset(valid_adata, pert_smiles_emb, dosage_prompt_emb, device, FC=config['using_FC'])
    test_dataset = myDataset(test_adata, pert_smiles_emb, dosage_prompt_emb, device, FC=config['using_FC'])
    valid_loader = myDataloader(valid_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = myDataloader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # Model
    model = Choose_model(config)
    model.init_weights()

    uncond = pert_smiles_emb['negative_ctrl'].mean(dim=0) if config['using_cfg'] else None
    sampler = Diffusion_Sampler(
        sampler_type=config['diffusion']['sampler_type'],
        num_train_timesteps=config['diffusion']['train_steps'],
        timesteps=config['diffusion']['num_steps'],
        start=config['diffusion']['beta_start'],
        end=config['diffusion']['beta_end'],
        beta_schedule=config['diffusion']['beta_schedule'],
        device=device, guidance_scale=config['diffusion']['guidance_scale'], uncond=uncond
    )
    _, generator_func = choose_loss_generator(config)
    relu = nn.ReLU(inplace=True)

    valid_limit = len(valid_loader)  # full validation
    test_limit = len(test_loader)

    results = []
    checkpoints = [
        ("best", f"{RESULT_DIR}/PertDit_best.pth"),
        ("latest", f"{RESULT_DIR}/PertDit_latest.pth"),
    ]

    for ckpt_name, ckpt_path in checkpoints:
        print(f"\n=== Evaluating {ckpt_name}: {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        epoch = ckpt['epoch']
        best_mse = ckpt['best_mse']
        print(f"  Epoch: {epoch}, train best_mse: {best_mse:.6f}")

        seed_everything(117)
        print("  Evaluating on valid set (full)...")
        valid_metrics = eval_on_dataset(model, generator_func, sampler, valid_loader, valid_limit,
                                         valid_dataset.drug_adata.obs, relu)

        seed_everything(117)
        print("  Evaluating on test set...")
        test_metrics = eval_on_dataset(model, generator_func, sampler, test_loader, test_limit,
                                        test_dataset.drug_adata.obs, relu)

        row = {
            'checkpoint': ckpt_name,
            'epoch': epoch,
            'train_best_mse': best_mse,
            'valid_MSE': valid_metrics['MSE'],
            'valid_per_sample_FC_PCC': valid_metrics['per_sample_FC_PCC'],
            'valid_drug_FC_PCC': valid_metrics['drug_FC_PCC'],
            'valid_slope': valid_metrics['slope_ypred_vs_x'],
            'test_MSE': test_metrics['MSE'],
            'test_per_sample_FC_PCC': test_metrics['per_sample_FC_PCC'],
            'test_drug_FC_PCC': test_metrics['drug_FC_PCC'],
            'test_cov_drug_FC_PCC': test_metrics['cov_drug_FC_PCC'],
            'test_slope': test_metrics['slope_ypred_vs_x'],
        }
        results.append(row)
        print(f"  Valid: MSE={valid_metrics['MSE']:.4f} drug_FC_PCC={valid_metrics['drug_FC_PCC']:.4f}")
        print(f"  Test:  MSE={test_metrics['MSE']:.4f} drug_FC_PCC={test_metrics['drug_FC_PCC']:.4f}")

    # Also extract per-epoch validation MSE from checkpoint
    ckpt_latest = torch.load(f"{RESULT_DIR}/PertDit_latest.pth", map_location='cpu')
    if 'mse_predict' in ckpt_latest and len(ckpt_latest['mse_predict']) > 0:
        print("\n=== Per-epoch validation MSE from checkpoint ===")
        for ep_idx, mse_tensor in enumerate(ckpt_latest['mse_predict']):
            ep_mse = mse_tensor.mean().item()
            results.append({
                'checkpoint': f'epoch_{ep_idx}',
                'epoch': ep_idx,
                'train_best_mse': np.nan,
                'valid_MSE': ep_mse,
                'valid_per_sample_FC_PCC': np.nan,
                'valid_drug_FC_PCC': np.nan,
                'valid_slope': np.nan,
                'test_MSE': np.nan,
                'test_per_sample_FC_PCC': np.nan,
                'test_drug_FC_PCC': np.nan,
                'test_cov_drug_FC_PCC': np.nan,
                'test_slope': np.nan,
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
