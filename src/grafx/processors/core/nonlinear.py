import torch
import torch.nn.functional as F


def compressor_gain_hard_knee(log_energy, log_threshold, log_ratio, log_knee=None):
    ratio = 1 + torch.exp(log_ratio)
    log_energy_out = torch.minimum(
        log_energy, log_threshold + (log_energy - log_threshold) / ratio
    )
    log_gain = log_energy_out - log_energy
    return log_gain


def compressor_gain_quad_knee_original(log_energy, log_threshold, log_ratio, log_knee):
    ratio = 1 + torch.exp(log_ratio)
    log_knee = torch.exp(log_knee)

    below_mask = log_energy < (log_threshold - log_knee / 2)
    above_mask = log_energy > (log_threshold + log_knee / 2)
    middle_mask = (~below_mask) * (~above_mask)

    below = log_energy
    above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)
    middle = log_energy + (1 / (ratio + 1e-3) - 1) * (
        log_energy - log_threshold + log_knee / 2
    ) ** 2 / 2 / (log_knee + 1e-3)

    log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
    log_gain = log_energy_out - log_energy
    return log_gain


def compressor_gain_quad_knee(log_energy, log_threshold, log_ratio, log_knee):
    ratio = 1 + torch.exp(log_ratio)
    log_knee = torch.exp(log_knee) / 2

    below_mask = log_energy < (log_threshold - log_knee)
    above_mask = log_energy > (log_threshold + log_knee)
    middle_mask = (~below_mask) * (~above_mask)

    below = log_energy
    above = log_threshold + (log_energy - log_threshold) / ratio
    middle = log_energy + (1 / ratio - 1) * (
        log_energy - log_threshold + log_knee
    ).square() / (4 * log_knee)

    log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
    log_gain = log_energy_out - log_energy
    return log_gain


def compressor_gain_exp_knee(log_energy, log_threshold, log_ratio, log_knee):
    ratio = 1 + torch.exp(log_ratio)
    log_knee = torch.exp(log_knee - 1)
    log_gain = (
        (1 / ratio - 1) * F.softplus(log_knee * (log_energy - log_threshold)) / log_knee
    )
    return log_gain
