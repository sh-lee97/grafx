import math

import torch


# for reference
def _biquad(
    gain_db: torch.Tensor,
    cutoff_freq: torch.Tensor,
    q_factor: torch.Tensor,
    filter_type: str = "peaking",
    sample_rate: int = 44100,
):

    bs = gain_db.size(0)
    # reshape params
    gain_db = gain_db.view(bs, -1)
    cutoff_freq = cutoff_freq.view(bs, -1)
    q_factor = q_factor.view(bs, -1)

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = torch.stack([b0, b1, b2], dim=1).view(bs, -1)
    a = torch.stack([a0, a1, a2], dim=1).view(bs, -1)

    # normalize
    b = b.type_as(gain_db) / a0
    a = a.type_as(gain_db) / a0

    return b, a


def _get_magnitude_resposne(Bs, As):
    arange = torch.arange(3)
    fir_length = 2**15
    faxis = torch.linspace(0, 22050, fir_length // 2 + 1)
    delays = delay(arange, fir_length=fir_length)
    fsm = iir_fsm(Bs, As, delays=delays)
    fsm_magnitude = torch.abs(fsm)
    fsm_db = torch.log(fsm_magnitude + 1e-7)
    return faxis, fsm_db
