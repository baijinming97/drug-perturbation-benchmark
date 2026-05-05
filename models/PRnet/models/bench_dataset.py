"""
NM-compatible dataset for PRnet.
Returns (x_ctrl, x_pert, drug_encoding) tuples.
Drug encoding = FCFP4 (1024-bit) * log10(dose + 1).
"""

import numpy as np
from torch.utils.data import Dataset


class BenchPRnetDataset(Dataset):
    """PRnet-compatible dataset reading from NM unified format.

    Args:
        x_ctrl:     (N, 978) float32 — basal/control expression
        x_pert:     (N, 978) float32 — perturbed expression
        fcfp4_feat: (N, 1024) float32 — precomputed FCFP4 * log10(dose+1)
    """

    def __init__(self, x_ctrl, x_pert, fcfp4_feat):
        self.x_ctrl = x_ctrl.astype(np.float32)
        self.x_pert = x_pert.astype(np.float32)
        self.fcfp4 = fcfp4_feat.astype(np.float32)

    def __getitem__(self, index):
        return self.x_ctrl[index], self.x_pert[index], self.fcfp4[index]

    def __len__(self):
        return len(self.x_ctrl)
