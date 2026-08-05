"""Shared runtime helpers: seeding and device selection.

Seeding covers Python/NumPy/Torch so the data split, weight init, and shuffle order are
fixed by config.seed. We deliberately do NOT enable
torch.use_deterministic_algorithms / cudnn-deterministic: those buy byte-for-byte
identity at a real speed cost, and we only need parameter-level reproducibility.

Metrics live in `evaluation/`, not here — this module stays free of anything that scores
a result.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
