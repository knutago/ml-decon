"""torch Dataset over the observed/ideal patch arrays written by gen_data.

No transforms: the patches are already normalized and (for now) un-augmented, so
__getitem__ just hands back tensors. The whole split fits in memory.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, data_dir, split):
        data_dir = Path(data_dir)
        self.observed = np.load(data_dir / f"{split}_observed.npy")
        self.ideal = np.load(data_dir / f"{split}_ideal.npy")

    def __len__(self):
        return len(self.observed)

    def __getitem__(self, index):
        return torch.from_numpy(self.observed[index]), torch.from_numpy(self.ideal[index])
