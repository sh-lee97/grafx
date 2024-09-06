import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcomp import compressor_core

from grafx.processors.core.convolution import FIRConvolution


class TruncatedOnePoleIIRFilter(nn.Module):
    r"""
    A one-pole IIR filter with a truncated impulse response.

        The true one-pole IIR filter is defined as a recursive filter with a coefficient $\alpha$.
        Here, for the speed-up, we calculate its truncated impulse response analytically and convolve it to the input signal.
        $$
        y[n] \approx u[n] * (1-\alpha)\alpha^n.
        $$

        The length of the truncated FIR, $N$, is given as an argument :python:`iir_len`.
    """

    def __init__(
        self,
        iir_len=16384,
        **backend_kwargs,
    ):
        super().__init__()
        arange = torch.arange(iir_len)[None, :]
        self.register_buffer("arange", arange)

        self.conv = FIRConvolution(mode="causal", **backend_kwargs)

    def forward(self, input_signals, z_alpha):
        r"""
        Processes input audio with the processor and given coefficients.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of one-pole coefficients.

        Returns:
            :python:`FloatTensor`: A batch of smoothed signals of shape :math:`B \times L`.
        """
        h = self.compute_impulse(z_alpha)
        smoothed = self.conv(input_signals, h)
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


class Ballistics(nn.Module):
    r"""
    A ballistics processor that smooths the input signal with a recursive filter.

        An input signal $u[n]$ is smoothed with recursively, with a different coefficient for an "attack" and "release".
        $$
        y[n] = \begin{cases}
        \alpha_\mathrm{R} y[n-1]+(1-\alpha_\mathrm{R}) u[n] & u[n] < y[n-1], \\
        \alpha_\mathrm{A} y[n-1]+(1-\alpha_\mathrm{A}) u[n] & u[n] \geq y[n-1]. \\
        \end{cases}
        $$

        We calculate the coefficients from the inputs with the sigmoid function, i.e., 
        $\alpha_\mathrm{A} = \sigma(z_{\mathrm{A}})$ and $\alpha_\mathrm{R} = \sigma(z_{\mathrm{R}})$.
        We use :python:`diffcomp` for the optimized forward and backward computation :cite:`yu2024differentiable`.

    """

    def __init__(self):
        super().__init__()

    def forward(self, input_signals, z_alpha):
        r"""
        Processes input audio with the processor and given coefficients.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`B \times 2`):
                A batch of attack and release coefficients stacked in the last dimension.

        Returns:
            :python:`FloatTensor`: A batch of smoothed signals of shape :math:`B \times L`.
        """
        ts = torch.sigmoid(z_alpha)
        zi = torch.ones(input_signals.shape[0], device=input_signals.device)
        at, rt = ts[..., 0], ts[..., 1]
        smoothed = compressor_core(input_signals, zi, at, rt)
        return smoothed


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
