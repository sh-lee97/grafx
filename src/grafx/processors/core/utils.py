import torch
import torch.nn.functional as F
import torch.fft
import numpy as np


def rms_difference(X, Y, eps=1e-7):
    X_rms = torch.log(X.square().mean((-1, -2)) + eps)
    Y_rms = torch.log(Y.square().mean((-1, -2)) + eps)
    diff = (X_rms - Y_rms).abs().sum()
    return diff


def normalize_impulse(ir):
    if ir.ndim != 3:
        raise Exception("An input impulse response must has shape of (b, c, t)")
    e = ir.square().sum(2, keepdim=True).mean(1, keepdim=True)
    ir = ir / torch.sqrt(e + 1e-12)
    return ir