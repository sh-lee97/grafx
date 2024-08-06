import torch
import torch.nn.functional as F
import torch.fft
import numpy as np


def rms_difference(X, Y, eps=1e-7):
    X_rms = torch.log(X.square().mean((-1, -2)) + eps)
    Y_rms = torch.log(Y.square().mean((-1, -2)) + eps)
    diff = (X_rms - Y_rms).abs().sum()
    return diff


def compute_pad_len(x, y, pad_mode="pow2"):
    pad_len = x.shape[-1] + y.shape[-1] - 1
    match pad_mode:
        case "pow2":
            pad_len_log2 = np.ceil(np.log2(pad_len))
            pad_len = int(2**pad_len_log2)
        case "min":
            return pad_len


def convolve(x, h, mode="zerophase", pad_mode="min"):
    pad_len = compute_pad_len(x, h, pad_mode)
    x_pad = F.pad(x, (0, pad_len - x.shape[-1]))
    h_pad = F.pad(h, (0, pad_len - h.shape[-1]))
    X_PAD = torch.fft.rfft(x_pad)
    H_PAD = torch.fft.rfft(h_pad)
    Y_PAD = X_PAD * H_PAD
    y_pad = torch.fft.irfft(Y_PAD)
    match mode:
        case "zerophase":
            y = y_pad[..., h.shape[-1] // 2 : h.shape[-1] // 2 + x.shape[-1]]
        case "causal":
            y = y_pad[..., : x.shape[-1]]
        case _:
            y = y_pad
    return y


def normalize_impulse(ir):
    if ir.ndim != 3:
        raise Exception("An input impulse response must has shape of (b, c, t)")
    e = ir.square().sum(2, keepdim=True).mean(1, keepdim=True)
    ir = ir / torch.sqrt(e + 1e-12)
    return ir
