import torch


def ms_to_lr(x):
    mid, side = torch.split(x, (1, 1), -2)
    left, right = mid + side, mid - side
    x = torch.cat([left, right], -2)
    return x


def lr_to_ms(x, mult=0.5):
    left, right = torch.split(x, (1, 1), -2)
    mid, side = left + right, left - right
    x = torch.cat([mid, side], -2)
    if mult is not None:
        x = x * mult
    return x
