import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grafx.processors.core.convolution import CausalConvolution
from torchcomp import compressor_core


class TruncatedOnePoleIIRFilter(nn.Module):
    def __init__(
        self,
        iir_len=16384,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        arange = torch.arange(iir_len)[None, :]
        self.register_buffer("arange", arange)

        self.conv = CausalConvolution(
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

    def forward(self, signal, z_alpha):
        h = self.compute_impulse(z_alpha)
        smoothed = self.conv(signal, h)
        smoothed = F.relu(smoothed)
        return smoothed

    def compute_impulse(self, z_alpha):
        alpha = torch.sigmoid(z_alpha)
        # alpha = torch.clamp(alpha, min=1e-5, max=1 - 1e-5)
        alpha = torch.clamp(alpha, max=1 - 1e-5)
        log_alpha = torch.log(alpha)
        log_decay = self.arange * log_alpha
        decay = torch.exp(log_decay)
        h = (1 - alpha) * decay
        return h

    def parameter_size(self):
        return {"z_alpha": 1}


class Ballistics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, signal, z_alpha):
        ts = torch.sigmoid(z_alpha)
        zi = torch.ones(signal.shape[0], device=signal.device)
        at, rt = ts[..., 0], ts[..., 1]
        smoothed = compressor_core(signal, zi, at, rt)
        # smoothed = signal
        return smoothed

    def parameter_size(self):
        return {"z_alpha": 2}


# class FramewiseBallistics(nn.Module):
#    r"""
#    """
#
#    def __init__(self, frame_len=1024):
#        super().__init__()
#
#    def forward(self, signal, z_ts):
#        ts = torch.sigmoid(z_ts)
#        ts_sample, ts_frame = ts.chunk(2, dim=-1)
#        ts_sample = ts_sample.clamp(min=1e-3, max=1 - 1e-3)
#        ts_frame = ts_frame.clamp(min=1e-5, max=1 - 1e-5)
#
#        zi = torch.ones(signal.shape[0], device=signal.device)
#        at_sample, rt_sample,  = ts.chunk(2, dim=-1)
#        #signal = compressor_core(signal, zi, at, rt)
#        return signal
#
#    def parameter_size(self):
#        return {"z_ts": 2}
