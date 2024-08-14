import math

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import lfilter

from grafx.processors.core.convolution import CausalConvolution
from grafx.processors.core.iir import FrequencySampledStateVariableFilter, svf_to_biquad

HALF_PI = math.pi / 2
TWOR_SCALE = 2 / math.log(2)


class SVFFilter(nn.Module):
    r"""
    A bank of second-order filters (biquads) with the state variable filter (SVF) parameters.
    Note that we are not using the exact time-domain implementation of the SVF, 
    but rather its parameterization that allows better optimization than the direct prediction of biquad coefficients 
    (empirically observed in :cite:`kuznetsov2020differentiable, nercessian2021lightweight, lee2022differentiable`).
    $$
    b_{i, 0} &= f^2_i m^{\text{LP}}_i+f_i m^{\text{BP}}_i+m^{\text{HP}}_i, \label{svf_to_biquad_1} \\
    b_{i, 1} &= 2f^2_i m^{\text{LP}}_i - 2m^{\text{HP}}_i, \\
    b_{i, 2} &= f^2_i m^{\text{LP}}_i-f_i m^{\text{BP}}_i+m^{\text{HP}}_i, \\
    a_{i, 0} &= f^2_i + 2R_if_i + 1, \\
    a_{i, 1} &= 2f^2_i-2, \\
    a_{i, 2} &= f^2_i - 2R_if_i + 1. \label{svf_to_biquad_2} 
    $$

    $$
    \tilde{H}[k] = \prod_i (H_i)_N[k] = \prod_i \left(\frac{\sum_{m=0}^2 b_{i, m} z^{-m}}{\sum_{m=0}^2 a_{i, m} z^{-m}}\right)_N[k].    
    $$

    Args:
        num_svfs (:python:`int`, *optional*):
            Number of SVFs to use (default: :python:`1`).
        pre_bilinear_warp (:python:`bool`, *optional*):
            Whether to pre-warp the cutoff frequency so that the filter has a accurate response near the Nyquist frequency.
            When set to :python:`False`, the biquad coefficients are identical to ones from :python:`torchaudio` (default: :python:`True`).
    """

    def __init__(
        self,
        num_svfs=1,
        backend="fsm",
        fsm_fir_len=4000,
    ):
        super().__init__()
        self.num_svfs = num_svfs
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

    def forward(self, input_signal, twoR, G, c_hp, c_bp, c_lp):
        G = torch.tan(HALF_PI * torch.sigmoid(G))
        twoR = TWOR_SCALE * F.softplus(twoR) + 1e-2

        match self.backend:
            case "fsm":
                fsm_response = self.fsm(input_signal, twoR, G, c_hp, c_bp, c_lp)
                fsm_response = fsm_response.prod(-2)
                fsm_fir = torch.fft.irfft(fsm_response, dim=-1, n=self.fsm_fir_len)
                output_signal = self.conv(input_signal, fsm_fir)
            case "lfilter":
                Bs, As = svf_to_biquad(twoR, G, c_hp, c_bp, c_lp)
                output_signal = input
                for i in range(self.num_svfs):
                    output_signal = lfilter(
                        output_signal, b_coeffs=Bs[i], ba_coeffs=As[i], atching=True
                    )

        return output_signal

    def parameter_size(self):
        return {
            "twoR": self.num_svfs,
            "G": self.num_svfs,
            "c_hp": self.num_svfs,
            "c_bp": self.num_svfs,
            "c_lp": self.num_svfs,
        }
