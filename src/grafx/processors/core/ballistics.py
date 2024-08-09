import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcomp import compressor_core
from grafx.processors.core.convolution import CausalConvolution


class IIRBallistics(nn.Module):
    r"""
    """

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
        alpha = torch.sigmoid(z_alpha)
        alpha = torch.clamp(alpha, min=1e-5, max=1 - 1e-5)
        log_alpha = torch.log(alpha)
        log_decay = self.arange * log_alpha
        decay = torch.exp(log_decay)
        h = (1 - alpha) * decay
        signal = self.conv(signal, h)
        signal = F.relu(signal)
        return signal

    def parameter_size(self):
        return {"z_alpha": 1}

class TrueBallistics(nn.Module):
    r"""
    """

    def __init__(self):
        super().__init__()

    def forward(self, signal, z_ts):
        ts = torch.sigmoid(z_ts).clamp(min=1e-5, max=1 - 1e-5)
        zi = torch.ones(signal.shape[0], device=signal.device)
        at, rt = ts.chunk(2, dim=-1)
        signal = compressor_core(signal, zi, at, rt)
        return signal

    def parameter_size(self):
        return {"z_ts": 2}


class FramewiseBallistics(nn.Module):
    r"""
    """

    def __init__(self, frame_len=1024):
        super().__init__()

    def forward(self, signal, z_ts):
        ts = torch.sigmoid(z_ts)
        ts_sample, ts_frame = ts.chunk(2, dim=-1)
        ts_sample = ts_sample.clamp(min=1e-3, max=1 - 1e-3) 
        ts_frame = ts_frame.clamp(min=1e-5, max=1 - 1e-5)   

        zi = torch.ones(signal.shape[0], device=signal.device)
        at_sample, rt_sample,  = ts.chunk(2, dim=-1)
        #signal = compressor_core(signal, zi, at, rt)
        return signal

    def parameter_size(self):
        return {"z_ts": 2}



class IIREnvelopeFollower(nn.Module):
    """
    A class representing an Infinite Impulse Response (IIR) Envelope Follower.

    Args:
        iir_len (int): The length of the IIR filter. Default is 16384.
        flashfftconv (bool): Whether to use flashfftconv. Default is True.
        max_input_len (int): The maximum input length. Default is 2**17.

    Attributes:
        arange (torch.Tensor): A tensor representing the range of values from 0 to iir_len.

    Methods:
        forward(signal, z_alpha): Computes the envelope of the input signal using the IIR filter.
        apply_ballistics(energy, alpha): Applies the ballistics to the energy signal.

    """

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
        """
        Computes the envelope of the input signal using the IIR filter.

        Args:
            signal (torch.Tensor): The input signal.
            z_alpha (torch.Tensor): The parameter controlling the decay rate of the envelope.

        Returns:
            torch.Tensor: The computed envelope of the input signal.

        """
        alpha = torch.sigmoid(z_alpha)
        alpha = torch.clamp(alpha, min=1e-5, max=1 - 1e-5)
        energy = signal.square().mean(-2)
        envelope = self.apply_ballistics(energy, alpha)
        envelope = torch.log(envelope + 1e-5)
        return envelope

    def apply_ballistics(self, energy, alpha):
        log_alpha = torch.log(alpha)
        log_decay = self.arange * log_alpha
        decay = torch.exp(log_decay)
        h = (1 - alpha) * decay
        envelope = self.conv(energy, h)
        envelope = F.relu(envelope)
        return envelope