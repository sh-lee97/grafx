import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

try:
    from flashfftconv import FlashFFTConv

    FLASHFFTCONV_AVAILABLE = True
except:
    FLASHFFTCONV_AVAILABLE = False


def compute_pad_len(x, y, pad_mode="pow2"):
    pad_len = x.shape[-1] + y.shape[-1] - 1
    match pad_mode:
        case "pow2":
            pad_len_log2 = np.ceil(np.log2(pad_len))
            pad_len = int(2**pad_len_log2)
        case "min":
            return pad_len


def convolve(x, h, mode="zerophase", pad_mode="min"):
    pad_len = compute_pad_len(x, h, pad_mode)
    x_pad = F.pad(x, (0, pad_len - x.shape[-1]))
    h_pad = F.pad(h, (0, pad_len - h.shape[-1]))
    X_PAD = torch.fft.rfft(x_pad)
    H_PAD = torch.fft.rfft(h_pad)
    Y_PAD = X_PAD * H_PAD
    y_pad = torch.fft.irfft(Y_PAD)
    match mode:
        case "zerophase":
            y = y_pad[..., h.shape[-1] // 2 : h.shape[-1] // 2 + x.shape[-1]]
        case "causal":
            y = y_pad[..., : x.shape[-1]]
        case _:
            y = y_pad
    return y


class CausalConvolution(nn.Module):
    """
    CausalConvolution module that performs convolution operation in a causal manner.

    Args:
        flashfftconv (bool): Flag indicating whether to use FlashFFTConv for convolution.
        max_input_len (int): Maximum input length for FlashFFTConv.

    Attributes:
        flashfftconv (bool): Flag indicating whether to use FlashFFTConv for convolution.
        conv (FlashFFTConv): FlashFFTConv instance for convolution.

    """

    def __init__(
        self,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        self.flashfftconv = flashfftconv
        if self.flashfftconv:
            flashfftconv_len = 2 ** int(np.ceil(np.log2(max_input_len)))
            self.conv = FlashFFTConv(flashfftconv_len, dtype=torch.bfloat16)

    def forward(self, x, h):
        """
        Forward pass of the CausalConvolution module.

        Args:
            x (torch.Tensor): Input tensor.
            h (torch.Tensor): Convolution kernel.

        Returns:
            torch.Tensor: Output tensor.

        """
        if self.flashfftconv:
            return self.flashfftconv_forward(x, h)
        else:
            return convolve(x, h, mode="causal")

    def flashfftconv_forward(self, x, h):
        x_shape, h_shape = x.shape, h.shape
        x = x.view(1, -1, x_shape[-1])
        x = x.type(torch.bfloat16)
        h = h.view(-1, h_shape[-1])
        y = self.conv(x, h)
        y = y.view(*x_shape[:-1], -1)
        return y