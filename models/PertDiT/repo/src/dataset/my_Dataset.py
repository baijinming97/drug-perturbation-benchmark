import os
import numpy as np 
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial
from dataset.drug_dose_encoder import Drug_dose_encoder

# There classes are partially cited from https://github.com/Perturbation-Response-Prediction/PRnet
class myDataset_LLM(Dataset):
    def __init__(self, adata, pert_smiles_emb, dosage_prompt_emb, device, cfg=False, cfg_prob=0.1, FC=False, scale=False):
        super(Dataset, self).__init__()
        # Process adata
        self.device = device
        self.dense_adata = adata
        if scale:
            self.mean_value = np.mean(self.dense_adata.X)
            self.std_value = np.std(self.dense_adata.X)
            self.dense_adata.X = (self.dense_adata.X - self.mean_value) * (0.5 / self.std_value)
            print("Scaling data to mean=0 and std=0.5")
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=0.0] 
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()
        self.control_index_dict = {index: i for i, index in enumerate(self.dense_adata_index)}
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose_val_4f'].to_list()
        # Load embedding
        self.pert_smiles_emb = pert_smiles_emb
        self.dosage_prompt_emb = dosage_prompt_emb
        self.cfg = cfg
        self.cfg_prob = cfg_prob
        self.FC = FC
        if self.cfg and self.cfg_prob>0:
            print("cfg activated in training")
        
    def __len__(self):
        return len(self.drug_adata)
        
    def __getitem__(self, index):
        #assert 0 <= index <= len(self.treated_exp)
        treated_exp = self.data[index, :]
        # control_index = self.dense_adata_index.index(self.paired_control_index[index])   
        control_index = self.control_index_dict[self.paired_control_index[index]]   
        ctrl_exp = self.dense_data[control_index,:]
        uniform_var = np.random.uniform(0, 1)
        if self.cfg and uniform_var < self.cfg_prob:
            mix_text_embed=self.pert_smiles_emb['negative_ctrl']
        else:
            mix_text_embed = torch.cat((self.pert_smiles_emb[self.drug_type_list[index]],self.dosage_prompt_emb[self.dose_list[index]]),dim=0)
        if self.FC:
            return treated_exp-ctrl_exp, ctrl_exp, mix_text_embed, mix_text_embed.shape[0]
        else:
            return treated_exp, ctrl_exp, mix_text_embed, mix_text_embed.shape[0]
    
    def scale_data(self, mean, scaler):
        self.data = (self.data - mean) * scaler
        self.dense_data = (self.dense_data - mean) * scaler
        print("Scaling data using mean and scaler")
     

class collater():
    def __init__(self, device):
        self.device = device
    
    def __call__(self, data):
        control_input_list, text_emb_list, treated_input_list, mask_list = [], [], [], []
        device = self.device
        for (treated_input, control_input, drug_input, mask) in data:
            control_input_list.append(control_input.to(device))
            text_emb_list.append(drug_input.to(device))
            treated_input_list.append(treated_input.to(device))
            mask_list.append(mask)
        return torch.stack(treated_input_list, 0), torch.stack(control_input_list, 0), torch.nn.utils.rnn.pad_sequence(text_emb_list).transpose(1, 0), mask_list

class myDataset_dosage_LLM_mean(Dataset):
    def __init__(self, adata, pert_smiles_emb, dosage_prompt_emb, device, cfg=False, cfg_prob=0.1, FC=False, scale=False):
        super(Dataset, self).__init__()
        # Process adata
        self.dense_adata = adata
        if scale:
            self.mean_value = np.mean(self.dense_adata.X)
            self.std_value = np.std(self.dense_adata.X)
            self.dense_adata.X = (self.dense_adata.X - self.mean_value) * (0.5 / self.std_value)
            print("Scaling data to mean=0 and std=0.5")
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=0.0] 
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()
        self.control_index_dict = {index: i for i, index in enumerate(self.dense_adata_index)}
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose_val_4f'].to_list()
        # Load embedding
        self.pert_smiles_emb = pert_smiles_emb
        self.dosage_prompt_emb = dosage_prompt_emb
        self.cfg = cfg
        self.cfg_prob = cfg_prob
        self.device = device
        self.FC = FC
        if self.cfg and self.cfg_prob>0:
            print("cfg activated in training")
    
    def __len__(self):
        return len(self.drug_adata)
        
    def __getitem__(self, index):
        #assert 0 <= index <= len(self.treated_exp)
        treated_exp = self.data[index, :]
        # control_index = self.dense_adata_index.index(self.paired_control_index[index])   
        control_index = self.control_index_dict[self.paired_control_index[index]]   
        ctrl_exp = self.dense_data[control_index,:]
        uniform_var = np.random.uniform(0, 1)
        if self.cfg and uniform_var < self.cfg_prob:
            mix_text_embed=self.pert_smiles_emb['negative_ctrl'].mean(dim=0)
        else:
            mix_text_embed = torch.cat((self.pert_smiles_emb[self.drug_type_list[index]],self.dosage_prompt_emb[self.dose_list[index]]),dim=0).mean(dim=0)
        if self.FC:
            return (treated_exp-ctrl_exp).to(self.device), ctrl_exp.to(self.device), mix_text_embed.to(self.device)
        else:
            return treated_exp.to(self.device), ctrl_exp.to(self.device), mix_text_embed.to(self.device)

    def scale_data(self, mean, scaler):
        self.data = (self.data - mean) * scaler
        self.dense_data = (self.dense_data - mean) * scaler
        print("Scaling data using mean and scaler")

class myDataset_rdkit(Dataset):
    def __init__(self, adata, pert_smiles_emb, dosage_prompt_emb, device, cfg=False, cfg_prob=0.1, FC=False):
        super(Dataset, self).__init__()
        # Process adata
        self.dense_adata = adata
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=0.0] 
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()
        self.control_index_dict = {index: i for i, index in enumerate(self.dense_adata_index)}
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose'].to_list()
        self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list)
        self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        # Load embedding
        self.pert_smiles_emb = pert_smiles_emb
        self.dosage_prompt_emb = dosage_prompt_emb
        self.cfg = cfg
        self.cfg_prob = cfg_prob
        self.device = device
        
    def __len__(self):
        return len(self.drug_adata)
        
    def __getitem__(self, index):
        #assert 0 <= index <= len(self.treated_exp)
        treated_exp = self.data[index, :]
        # control_index = self.dense_adata_index.index(self.paired_control_index[index])   
        control_index = self.control_index_dict[self.paired_control_index[index]]   
        ctrl_exp = self.dense_data[control_index,:]
        drug_dose_embed = self.encode_drug_doses[index, :]
        return treated_exp.to(self.device), ctrl_exp.to(self.device), drug_dose_embed.to(self.device)

# ["LLM", "LLM_mean", "Rdkit"]
def Choose_dataset_loader(drug_encoder, device):
    if drug_encoder == "LLM":
        return myDataset_LLM, partial(DataLoader, collate_fn = collater(device))
    elif drug_encoder == "Rdkit":
        return myDataset_rdkit, DataLoader
    elif drug_encoder == "LLM_mean":
        return myDataset_dosage_LLM_mean, DataLoader
    else:
        raise NotImplementedError