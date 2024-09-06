import math

import torch
import torch.nn as nn

# Third-Octave Design (31 bands)
FC_THIRD_OCTAVE = torch.tensor(
    [
        19.69,
        24.80,
        31.25,
        39.37,
        49.61,
        62.50,
        78.75,
        99.21,
        125.0,
        157.5,
        198.4,
        250.0,
        315.0,
        396.9,
        500.0,
        630.0,
        793.7,
        1000.0,
        1260.0,
        1587.0,
        2000.0,
        2520.0,
        3175.0,
        4000.0,
        5040.0,
        6350.0,
        8000.0,
        10080.0,
        12700.0,
        16000.0,
        20160.0,
    ]
)

FB_THIRD_OCTAVE = torch.tensor(
    [
        9.178,
        11.56,
        14.57,
        18.36,
        23.13,
        29.14,
        36.71,
        46.25,
        58.28,
        73.43,
        92.51,
        116.6,
        146.9,
        185.0,
        233.1,
        293.7,
        370.0,
        466.2,
        587.4,
        740.1,
        932.4,
        1175,
        1480,
        1865,
        2350,
        2846,
        3502,
        4253,
        5038,
        5689,
        5573,
    ]
)

# Bark Scale Design (24 bands)
FC_BARK = torch.tensor(
    [
        50,
        150,
        250,
        350,
        450,
        570,
        700,
        840,
        1000,
        1170,
        1370,
        1600,
        1850,
        2150,
        2500,
        2900,
        3400,
        4000,
        4800,
        5800,
        7000,
        8500,
        10500,
        13500,
    ]
)

FB_BARK = torch.tensor(
    [
        133.3,
        160.0,
        171.4,
        177.8,
        214.7,
        235.9,
        256.7,
        294.4,
        315.5,
        370.8,
        426.9,
        466.2,
        558.1,
        651.0,
        744.8,
        926.5,
        1110.0,
        1467.0,
        1828.0,
        2194.0,
        2735.0,
        3619.0,
        5333.0,
        6000.0,
    ]
)


class GraphicEqualizerBiquad(nn.Module):
    r"""
    :cite:`liski2017quest`
    """

    def __init__(self, scale="bark", sr=44100):
        super().__init__()

        match scale:
            case "bark":
                fc = FC_BARK
                fB = FB_BARK
                # c = [0.36] + [0.42] * 23
                c = [0.4] * 24
            case "third_octave":
                fc = FC_THIRD_OCTAVE
                fB = FB_THIRD_OCTAVE
                c = [0.4] * 31
            case _:
                raise ValueError(f"Unsupported scale: {scale}")

        fc = fc[fc < sr / 2]
        fB = fB[: len(fc)]
        wc = 2 * math.pi * fc / sr
        m2_cos_wc = -2 * torch.cos(wc)
        B_half = math.pi * fB / sr
        tan_B_half = torch.tan(B_half)

        self.register_buffer("fc", fc)
        self.register_buffer("fB", fB)
        self.register_buffer("m2_cos_wc", m2_cos_wc)
        self.register_buffer("tan_B_half", tan_B_half)
        self.register_buffer("c", torch.tensor(c))

        self.num_bands = len(fc)

    def forward(self, log_gains):
        # gains = torch.exp(log_gains)
        gains = torch.exp(log_gains)
        gains_square = gains.square()

        # neighbor_gains = torch.exp(neighbor_log_gains)
        neighbor_log_gains = log_gains * self.c
        neighbor_gains = torch.exp(neighbor_log_gains)
        neighbor_gains_square = neighbor_gains.square()

        beta = torch.ones_like(gains) * self.tan_B_half
        nonzero_gain_mask = log_gains.abs() >= 1e-3
        nonzero_gain_beta_mult = torch.sqrt(
            ((1 - neighbor_gains_square).abs() + 1e-7)
            / ((gains_square - neighbor_gains_square).abs() + 1e-7)
        )
        beta[nonzero_gain_mask] = (
            beta[nonzero_gain_mask] * nonzero_gain_beta_mult[nonzero_gain_mask]
        )
        # beta = self.tan_B_half * nonzero_gain_beta_mult
        gbeta = gains * beta

        # repeat to make the same shape as b0, i.e., from [N] to [..., N]
        repeat_shape = log_gains.shape[:-1]
        m2_cos_wc = self.m2_cos_wc.repeat(*repeat_shape, 1)
        b0 = 1 + gbeta
        b1 = m2_cos_wc
        b2 = 1 - gbeta
        a0 = 1 + beta
        a1 = m2_cos_wc
        a2 = 1 - beta

        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As
