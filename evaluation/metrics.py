"""Metrics for scoring a deconvolution result.

Every function here takes arrays and returns a number — never a model. That is what
keeps the evaluation workflow model-agnostic: a classical baseline (Richardson-Lucy,
Wiener) and a trained network are scored through the identical path.

Metrics operate in NORMALIZED space (see core.normalize), where the data is roughly
[0, 1]. Passing raw FITS flux gives a meaningless number.
"""

import math

import torch


def psnr(predicted, ideal):
    """PSNR in dB for data in [0, 1], so peak = 1. Returns a Python float.

    PSNR = 10 * log10(peak^2 / MSE); with peak = 1 this reduces to -10 * log10(MSE).
    """
    mse = torch.mean((predicted - ideal) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)
