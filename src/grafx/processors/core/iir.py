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


class BiquadFilterBackend(nn.Module):
    def __init__(
        self,
        backend="fsm",
        fsm_fir_len=4000,
    ):
        super().__init__()
        self.backend = backend
        self.fsm_fir_len = fsm_fir_len

        match backend:
            case "fsm":
                self.svf = FrequencySampledStateVariableFilter(fir_len=fsm_fir_len)
                self.conv = CausalConvolution(fsm_fir_len)
            case "lfilter":
                pass
            case _:
                raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, input_signal, Bs, As):
        match self.backend:
            case "fsm":
                fsm_response = self.fsm(input_signal, twoR, G, c_hp, c_bp, c_lp)
                fsm_response = fsm_response.prod(-2)
                fsm_fir = torch.fft.irfft(fsm_response, dim=-1, n=self.fsm_fir_len)
                output_signal = self.conv(input_signal, fsm_fir)
            case "lfilter":
                output_signal = input
                num_filters = Bs.shape[-2]
                for i in range(num_filters):
                    output_signal = lfilter(
                        output_signal,
                        b_coeffs=Bs[..., i, :],
                        a_coeffs=As[..., i, :],
                        batching=True,
                    )

        return output_signal


def peaking(gain_db, cutoff_freq, q_factor, sample_rate=44100):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + (alpha / A)
    a1 = -2 * cos_w0
    a2 = 1 - (alpha / A)

    Bs = torch.stack([b0, b1, b2], -1)
    As = torch.stack([a0, a1, a2], -1)
    return Bs, As


def get_magnitude_resposne(Bs, As):
    arange = torch.arange(3)
    fir_length = 2**15
    faxis = torch.linspace(0, 22050, fir_length // 2 + 1)
    delays = delay(arange, fir_length=fir_length)
    fsm = iir_fsm(Bs, As, delays=delays)
    fsm_magnitude = torch.abs(fsm)
    fsm_db = torch.log(fsm_magnitude + 1e-7)
    return faxis, fsm_db


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from grafx.processors.core.iir import delay, iir_fsm

    sr = 44100
    gain_db = torch.tensor([6.0])
    cutoff_freq = torch.tensor([1000.0])
    q_factor = torch.tensor([3.0])
    G = torch.tan(math.pi * cutoff_freq / sr)
    twoR = 1 / q_factor / np.sqrt(2)
    c = 10 ** (gain_db / 20)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    Bs, As = peak_to_biquad(G, c, twoR)
    faxis, peak_1 = get_magnitude_resposne(Bs, As)
    ax.plot(faxis, peak_1[0])
    Bs, As = peaking(gain_db, cutoff_freq, q_factor, sample_rate=sr)
    faxis, peak_2 = get_magnitude_resposne(Bs, As)
    ax.plot(faxis, peak_2[0])

    ax.set_xlim(10, 22050)
    ax.set_xscale("symlog", linthresh=10, linscale=0.1)

    fig.savefig("peak.pdf", bbox_inches="tight")


# if __name__ == "__main__":
#    svf = FrequencySampledStateVariableFilter()
#    twoR = torch.rand(16, 1)
#    G = torch.rand(16, 1)
#    c_hp = torch.rand(16, 1)
#    c_bp = torch.rand(16, 1)
#    c_lp = torch.rand(16, 1)
#    response = svf(twoR, G, c_hp, c_bp, c_lp)
#    print(response.shape)
