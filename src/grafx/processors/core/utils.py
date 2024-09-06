import numpy as np
import torch
import torch.fft
import torch.nn.functional as F


def rms_difference(X, Y, eps=1e-7):
    X_rms = torch.log(X.square().mean((-1, -2)) + eps)
    Y_rms = torch.log(Y.square().mean((-1, -2)) + eps)
    diff = (X_rms - Y_rms).abs().sum()
    return diff


def normalize_impulse(ir, eps=1e-12):
    assert ir.ndim == 3
    e = ir.square().sum(2, keepdim=True).mean(1, keepdim=True)
    ir = ir / torch.sqrt(e + eps)
    return ir
