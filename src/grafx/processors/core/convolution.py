import warnings

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

try:
    from flashfftconv import FlashFFTConv

    FLASHFFTCONV_AVAILABLE = True
except:
    FLASHFFTCONV_AVAILABLE = False


class FIRConvolution(nn.Module):
    r"""
    A FIR convolution backend, which can use either native FFT-based convolution or :python:`FlashFFTConv` :cite:`fu2023flashfftconv`.
    Allows for causal and zero-phase convolution modes.

        For an input $\mathbf{U}\in\mathbb{R}^{B\times C_{\mathrm{in}} \times L_{\mathrm{in}}}$
        and a filter $\mathbf{U}\in\mathbb{R}^{B\times C_{\mathrm{filter}} \times L_{\mathrm{filter}}}$
        the operation is defined as a usual convolution. However, the output length will be the one of the input
        and the number of the output channels will be determined by broadcasting.

    Args:
        mode (:python:`str`, *optional*):
            The convolution mode, either :python:`"causal"` or :python:`"zerophase"` (default: :python:`"causal"`).
        flashfftconv (:python:`bool`, *optional*):
            An option to use :python:`FlashFFTConv` as a backend
            (default: :python:`True`).
        max_input_len (:python:`int`, *optional*):
            When :python:`flashfftconv` is set to :python:`True`,
            the max input length must be also given (default: :python:`2**17`).
    """

    def __init__(
        self,
        mode="causal",
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        self.mode = mode

        if not FLASHFFTCONV_AVAILABLE and flashfftconv:
            warnings.warn(
                f"FlashFFTConv is not available. Using native convolution instead."
            )
            flashfftconv = False

        self.flashfftconv = flashfftconv

        if self.flashfftconv and mode == "zerophase":
            warnings.warn(
                f"When using FlashFFTConv with zerophase mode, make sure that the sum of the input and kernel lengths is less than or equal to max_input_len."
            )

        if self.flashfftconv:
            flashfftconv_len = 2 ** int(np.ceil(np.log2(max_input_len)))
            self.conv = FlashFFTConv(flashfftconv_len, dtype=torch.bfloat16)
            self._forward = self._flashfftconv_forward
        else:
            self._forward = self._native_forward

    def forward(self, input_signals, fir):
        r"""
        Performs the convolution operation.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C_\mathrm{in} \times L_\mathrm{in}`):
                A batch of input audio signals.
            fir (:python:`FloatTensor`, :math:`B \times C_\mathrm{filter} \times L_\mathrm{filter}`):
                A batch of FIR filters.

        Returns:
            :python:`FloatTensor`: A batch of convolved signals of shape :math:`B \times C_\mathrm{out} \times L_\mathrm{in}` where :math:`C_\mathrm{out} = \max (C_\mathrm{in}, C_\mathrm{filter})`.
        """
        return self._forward(input_signals, fir)

    def _native_forward(self, x, h):
        return convolve(x, h, mode=self.mode)

    def _flashfftconv_forward(self, x, h):
        if self.mode == "zerophase":
            assert (
                False
            ), "We currently do not support zerophase mode with FlashFFTConv."
        x_shape, h_shape = x.shape, h.shape

        if x_shape[-2] == 1 and h_shape[-2] != 1:
            x = x.repeat(1, h_shape[-2], 1)
        elif x_shape[-2] != 1 and h_shape[-2] == 1:
            h = h.repeat(1, x_shape[-2], 1)

        x_shape, h_shape = x.shape, h.shape

        x = x.view(1, -1, x_shape[-1])
        x = x.type(torch.bfloat16)
        h = h.view(-1, h_shape[-1])

        y = self.conv(x, h)
        y = y.view(*x_shape[:-1], -1)
        y = y.type(h.dtype)  #####################################
        return y


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
