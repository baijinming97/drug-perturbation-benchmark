"""
NM-compatible dataset for TranSiGen.
Drop-in replacement for the original dataset.py, reading from numpy arrays
instead of TranSiGen's monolithic HDF5 format.

Returns the same 6-tuple as the original TranSiGenDataset:
    (x1, x2, mol_feature, mol_id, cid, sig)
"""

import numpy as np
from torch.utils.data import Dataset


class BenchTranSiGenDataset(Dataset):
    """TranSiGen-compatible dataset reading from NM unified format.

    Args:
        x_ctrl:    (N, 978) basal expression (x1)
        x_pert:    (N, 978) perturbed expression (x2)
        pert_idx:  (N,) drug identifiers (int)
        cell_idx:  (N,) cell identifiers (int)
        kpgt_dict: {int: ndarray(2304,)} pert_idx -> KPGT embedding
    """

    def __init__(self, x_ctrl, x_pert, pert_idx, cell_idx, kpgt_dict):
        self.x_ctrl = x_ctrl.astype(np.float32)
        self.x_pert = x_pert.astype(np.float32)
        self.pert_idx = pert_idx.astype(int)
        self.cell_idx = cell_idx.astype(int)
        self.kpgt_dict = kpgt_dict

    def __getitem__(self, index):
        x1 = self.x_ctrl[index]                                      # (978,) float32
        x2 = self.x_pert[index]                                      # (978,) float32
        mol_id = int(self.pert_idx[index])                            # int
        mol_feature = self.kpgt_dict[mol_id].astype(np.float32)      # (2304,) float32
        cid = str(self.cell_idx[index])                               # str
        sig = str(index)                                              # str
        return x1, x2, mol_feature, mol_id, cid, sig

    def __len__(self):
        return len(self.pert_idx)
