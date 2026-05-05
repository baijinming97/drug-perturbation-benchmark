"""
Task 2: Conditioning ablation
For random_split_0 and drug_split_0 best checkpoints:
  a) Normal drug embedding
  b) Zero/negative_ctrl conditioning (replace drug embedding with negative_ctrl)
  c) Shuffle drug embedding within test batch

Compare per-sample FC PCC and drug-level FC PCC.
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

OUTPUT_CSV = "data/result/conditioning_ablation.csv"

def eval_with_condition(model, sampler, loader, limit, drug_adata_obs, relu, device,
                        pert_smiles_emb, mode='normal'):
    """
    Evaluate model with different conditioning modes.
    mode: 'normal', 'zero', 'shuffle'
    """
    model.eval()
    total_y_pred = torch.Tensor()
    total_y_true = torch.Tensor()
    total_x = torch.Tensor()

    neg_ctrl_embed = pert_smiles_emb['negative_ctrl']  # negative control embedding

    with torch.no_grad():
        all_batches = list(enumerate(loader))
        for i, batch in tqdm(all_batches, total=min(limit, len(all_batches)), desc=f"eval_{mode}"):
            if i == limit:
                break
            treated, ctrl, drug_embed, mask = batch

            if mode == 'zero':
                # Replace drug embedding with negative_ctrl (= no drug)
                batch_size = treated.shape[0]
                # negative_ctrl embed has shape (n_tokens, d_model)
                neg = neg_ctrl_embed.unsqueeze(0).expand(batch_size, -1, -1).to(device)
                drug_embed = neg
                mask = [neg_ctrl_embed.shape[0]] * batch_size

            elif mode == 'shuffle':
                # Shuffle drug embeddings across samples within batch
                perm = torch.randperm(treated.shape[0])
                drug_embed = drug_embed[perm]
                mask = [mask[j] for j in perm.tolist()]

            batch_modified = (treated, ctrl, drug_embed, mask)

            # Generate using Cross generator
            y_pred = sampler.ddim_sample(model, treated.shape, ctrl, drug_embed, mask).cpu()
            y_pred = relu(y_pred)
            y_true = treated.cpu()
            x = ctrl.cpu()

            total_y_pred = torch.cat((total_y_pred, y_pred), 0)
            total_y_true = torch.cat((total_y_true, y_true), 0)
            total_x = torch.cat((total_x, x), 0)

    # per-sample FC PCC
    fc_true = (total_y_true - total_x).numpy()
    fc_pred = (total_y_pred - total_x).numpy()
    sample_pccs = []
    for j in range(min(5000, fc_true.shape[0])):
        r, _ = pearsonr(fc_true[j], fc_pred[j])
        if not np.isnan(r):
            sample_pccs.append(r)
    per_sample_fc_pcc = np.mean(sample_pccs)

    # drug-level FC PCC
    coeff_drug, _ = calculate_correlation_coefficients(drug_adata_obs, 'condition', total_x, total_y_true, total_y_pred)
    drug_fc_pcc = np.nanmean(coeff_drug)

    # MSE
    avg_mse = ((total_y_true - total_y_pred)**2).mean().item()

    # slope
    slopes = []
    for j in range(min(2000, total_y_pred.shape[0])):
        c = np.polyfit(total_x[j].numpy(), total_y_pred[j].numpy(), 1)
        slopes.append(c[0])
    avg_slope = np.mean(slopes)

    return {
        'MSE': avg_mse,
        'per_sample_FC_PCC': per_sample_fc_pcc,
        'drug_FC_PCC': drug_fc_pcc,
        'slope': avg_slope
    }


def run_for_split(split_key, results_list):
    result_dir = f"data/result/{split_key}/CrossDiT_repro_s117"
    config_path = f"{result_dir}/config.yaml"
    ckpt_path = f"{result_dir}/PertDit_best.pth"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Split: {split_key}")
    print(f"{'='*60}")

    # Load data
    pert_smiles_emb = torch.load('data/pert_smiles_emb.pkl')
    dosage_prompt_emb = torch.load('data/dosage_prompt_emb_lincs.pkl')
    lincs_adata = sc.read('data/lincs_adata.h5ad')
    lincs_adata.X = np.clip(lincs_adata.X, 0, 1e3)
    sc.pp.normalize_total(lincs_adata)
    sc.pp.log1p(lincs_adata)

    _, _, test_adata = train_valid_test(lincs_adata, split_key=split_key)
    device = torch.device(config['device'])
    myDataset, myDataloader = Choose_dataset_loader(config['drug_encoder'], device)
    test_dataset = myDataset(test_adata, pert_smiles_emb, dosage_prompt_emb, device, FC=config['using_FC'])
    test_loader = myDataloader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # Model
    model = Choose_model(config)
    model.init_weights()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

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
    relu = nn.ReLU(inplace=True)
    test_limit = len(test_loader)

    for mode in ['normal', 'zero', 'shuffle']:
        print(f"\n--- Mode: {mode} ---")
        seed_everything(117)
        metrics = eval_with_condition(
            model, sampler, test_loader, test_limit,
            test_dataset.drug_adata.obs, relu, device,
            pert_smiles_emb, mode=mode
        )
        row = {
            'split': split_key,
            'mode': mode,
            **metrics
        }
        results_list.append(row)
        print(f"  MSE={metrics['MSE']:.4f} per_sample_FC_PCC={metrics['per_sample_FC_PCC']:.4f} "
              f"drug_FC_PCC={metrics['drug_FC_PCC']:.4f} slope={metrics['slope']:.4f}")


def main():
    results = []
    run_for_split("random_split_0", results)
    run_for_split("drug_split_0", results)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
