"""
Phase 6 PertDiT Dataset — extends NM dataset with per-sample dose support.

Adds dose_list parameter for multi-dose data (original Lincs has variable doses).
When dose_list is None, falls back to single dose_emb tensor (Phase 1/2 behavior).
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BenchPertDiTDataset(Dataset):
    """
    Args:
        x_ctrl:       (N, 978) control expression
        x_pert:       (N, 978) treated expression
        pert_idx:     (N,) int drug indices
        drug_emb_dict: dict[int → Tensor(L, 1024)] + 'negative_ctrl'
        dose_emb:     Tensor(L_dose, 1024) for dose=10.0
        cfg:          enable classifier-free guidance dropout
        cfg_prob:     probability of replacing drug with negative_ctrl
    """

    def __init__(self, x_ctrl, x_pert, pert_idx, drug_emb_dict, dose_emb,
                 dose_list=None, cfg=False, cfg_prob=0.1):
        self.x_ctrl = torch.tensor(x_ctrl, dtype=torch.float32)
        self.x_pert = torch.tensor(x_pert, dtype=torch.float32)
        self.pert_idx = pert_idx.astype(int)
        self.drug_emb_dict = drug_emb_dict
        self.dose_emb = dose_emb
        self.dose_list = dose_list
        self.neg_ctrl = drug_emb_dict['negative_ctrl']
        self.cfg = cfg
        self.cfg_prob = cfg_prob

    def __getitem__(self, index):
        x_pert = self.x_pert[index]    # (978,)
        x_ctrl = self.x_ctrl[index]    # (978,)

        # CFG dropout: with cfg_prob, replace drug embedding with negative_ctrl
        # Matches upstream preprocessing.ipynb / my_Dataset.py line 50
        if self.cfg and random.random() < self.cfg_prob:
            mix_emb = self.neg_ctrl                     # (2, 1024)
        else:
            drug_emb = self.drug_emb_dict[self.pert_idx[index]]  # (L_drug, 1024)
            dose_emb = self.dose_emb[self.dose_list[index]] if self.dose_list is not None else self.dose_emb
            mix_emb = torch.cat([drug_emb, dose_emb], dim=0)  # (L_drug+L_dose, 1024)

        return x_pert, x_ctrl, mix_emb, mix_emb.shape[0]

    def __len__(self):
        return len(self.x_pert)


class BenchCollater:
    """
    Pad variable-length drug embeddings to max length in batch.
    Matches PertDiT's original collater (my_Dataset.py lines 64-76).
    """

    def __call__(self, data):
        treated_list = []
        control_list = []
        drug_list = []
        mask_list = []

        for (treated, control, drug, seq_len) in data:
            treated_list.append(treated)
            control_list.append(control)
            drug_list.append(drug)
            mask_list.append(seq_len)

        return (
            torch.stack(treated_list, 0),                                    # (B, 978)
            torch.stack(control_list, 0),                                    # (B, 978)
            torch.nn.utils.rnn.pad_sequence(drug_list).transpose(1, 0),     # (B, max_L, 1024)
            mask_list                                                        # list[int]
        )


def build_dataloaders(train_ds, valid_ds, test_ds, batch_size=64, num_workers=0):
    collate_fn = BenchCollater()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader
