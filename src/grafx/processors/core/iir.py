"""
IIR and frequency-sampled IIR filters
"""

import math
import warnings
from functools import partial

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import lfilter

from grafx.processors.core.convolution import CausalConvolution

TORCHAUDIO_VERSION = torchaudio.__version__


if TORCHAUDIO_VERSION < "2.4.0":
    warnings.warn(
        f"The current version of torchaudio ({TORCHAUDIO_VERSION}) provides lfilter that could be either slow or faulty. For example, see https://github.com/pytorch/audio/releases/tag/v2.4.0. We recommend using torchaudio>=2.4.0 or using the frequency-sampled version instead."
    )


def delay(delay_length, fir_length):
    r"""

    Args:
        delay_length (:python:`FloatTensor` or :python:`LongTensor`): Delay lengths
        fir_length (int): length of the FIR filter
    """
    ndim = delay_length.ndim
    arange = torch.arange(fir_length // 2 + 1, device=delay_length.device)
    arange = arange.view((1,) * ndim + (-1,))
    phase = delay_length.unsqueeze(-1) * arange / fir_length * 2 * np.pi
    delay = torch.exp(-1j * phase)
    return delay


def svf_to_biquad(twoR, G, c_hp, c_bp, c_lp):
    G_square = G.square()

    b0 = c_hp + c_bp * G + c_lp * G_square
    b1 = -c_hp * 2 + c_lp * 2 * G_square
    b2 = c_hp - c_bp * G + c_lp * G_square

    a0 = 1 + G_square + twoR * G
    a1 = 2 * G_square - 2
    a2 = 1 + G_square - twoR * G

    Bs = torch.stack([b0, b1, b2], -1)
    As = torch.stack([a0, a1, a2], -1)

    return Bs, As


def lowpass_to_biquad(G, twoR):
    return svf_to_biquad(twoR, G, 0, 0, 1)


def highpass_to_biquad(G, twoR):
    return svf_to_biquad(twoR, G, 1, 0, 0)


def bandpass_to_biquad(G, twoR):
    return svf_to_biquad(twoR, G, 0, twoR, 0)


def lowshelf_to_biquad(G, c, twoR):
    c_hp, c_bp, c_lp = c, twoR * torch.sqrt(c), 1
    return svf_to_biquad(twoR, G, c_hp, c_bp, c_lp)


def highshelf_to_biquad(G, c, twoR):
    c_hp, c_bp, c_lp = 1, twoR * torch.sqrt(c), c
    return svf_to_biquad(twoR, G, c_hp, c_bp, c_lp)


def peak_to_biquad(G, c, twoR):
    c_hp, c_bp, c_lp = 1, twoR * c, 1
    return svf_to_biquad(twoR, G, c_hp, c_bp, c_lp)


def iir_fsm(Bs, As, delays, eps=1e-10):
    Bs, As = Bs.unsqueeze(-1), As.unsqueeze(-1)
    biquad_response = torch.sum(Bs * delays, -2) / (torch.sum(As * delays, -2) + eps)
    return biquad_response


class FrequencySampledStateVariableFilter(nn.Module):
    def __init__(self, fir_length=4000):
        super().__init__()

        arange = torch.arange(3)
        delays = delay(arange, fir_length=fir_length)
        self.register_buffer("delays", delays)
        # self.svf_to_biquad = torch.compile(svf_to_biquad)

    def forward(self, twoR, G, c_hp, c_bp, c_lp):
        Bs, As = svf_to_biquad(twoR, G, c_hp, c_bp, c_lp)
        svf = iir_fsm(Bs, As, delays=self.delays)
        return svf


if __name__ == "__main__":
    svf = FrequencySampledStateVariableFilter()
    twoR = torch.rand(16, 1)
    G = torch.rand(16, 1)
    c_hp = torch.rand(16, 1)
    c_bp = torch.rand(16, 1)
    c_lp = torch.rand(16, 1)
    response = svf(twoR, G, c_hp, c_bp, c_lp)
    print(response.shape)
