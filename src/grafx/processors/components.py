import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grafx.processors.functional import convolve

try:
    from flashfftconv import FlashFFTConv

    FLASHFFTCONV_AVAILABLE = True
except:
    FLASHFFTCONV_AVAILABLE = False


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


class IIREnvelopeFollower(nn.Module):
    """
    A class representing an Infinite Impulse Response (IIR) Envelope Follower.

    Args:
        iir_len (int): The length of the IIR filter. Default is 16384.
        flashfftconv (bool): Whether to use flashfftconv. Default is True.
        max_input_len (int): The maximum input length. Default is 2**17.

    Attributes:
        arange (torch.Tensor): A tensor representing the range of values from 0 to iir_len.

    Methods:
        forward(signal, z_alpha): Computes the envelope of the input signal using the IIR filter.
        apply_ballistics(energy, alpha): Applies the ballistics to the energy signal.

    """

    def __init__(
        self,
        iir_len=16384,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        arange = torch.arange(iir_len)[None, :]
        self.register_buffer("arange", arange)

        self.conv = CausalConvolution(
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

    def forward(self, signal, z_alpha):
        """
        Computes the envelope of the input signal using the IIR filter.

        Args:
            signal (torch.Tensor): The input signal.
            z_alpha (torch.Tensor): The parameter controlling the decay rate of the envelope.

        Returns:
            torch.Tensor: The computed envelope of the input signal.

        """
        alpha = torch.sigmoid(z_alpha)
        alpha = torch.clamp(alpha, min=1e-5, max=1 - 1e-5)
        energy = signal.square().mean(-2)
        envelope = self.apply_ballistics(energy, alpha)
        envelope = torch.log(envelope + 1e-5)
        return envelope

    def apply_ballistics(self, energy, alpha):
        log_alpha = torch.log(alpha)
        log_decay = self.arange * log_alpha
        decay = torch.exp(log_decay)
        h = (1 - alpha) * decay
        envelope = self.conv(energy, h)
        envelope = F.relu(envelope)
        return envelope


class NormalizedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input / (1e-7 + grad_input.abs())


class SurrogateDelay(nn.Module):
    """
    A module that represents a surrogate delay line.

    Args:
        N (int): The size of the delay line. Default is 2048.
        straight_through (bool): Whether to use straight-through gradient estimator during training. Default is True.
    """

    def __init__(self, N=2048, straight_through=True):
        super().__init__()

        self.straight_through = straight_through

        sin_N = N // 2 + 1
        arange_sin = torch.arange(sin_N)
        self.register_buffer("arange_sin", arange_sin[None, :])

    def forward(self, z):
        """
        Forward pass of the surrogate delay line.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing the delay line.
            torch.Tensor: The radii loss.
        """
        assert z.dtype == torch.cfloat

        shape = z.shape
        z = z.view(-1)

        radii_loss = self.calculate_radii_loss(z)

        z = NormalizedGradient.apply(z)
        mag = torch.abs(z)
        z = z * torch.tanh(mag) / (mag + 1e-7)

        sins = z[:, None] ** self.arange_sin
        irs = torch.fft.irfft(sins)

        if self.straight_through:
            irs = self.apply_straight_through(irs)

        irs = irs.view(*shape, -1)
        return irs, radii_loss

    def calculate_radii_loss(self, z):
        mag = torch.abs(z)
        mag = torch.tanh(mag)
        radii_losses = (1 - mag).square()
        return radii_losses.sum()

    def apply_straight_through(self, irs):
        hard_irs = self.get_hard_irs(irs)
        irs = irs + (hard_irs - irs).detach()
        return irs

    @torch.no_grad()
    def get_hard_irs(self, irs):
        """
        Get the hard impulse responses.

        Args:
            irs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The hard impulse responses.
        """
        hard_irs = torch.zeros_like(irs)
        onset = torch.argmax(irs, -1)
        arange_ir = torch.arange(len(hard_irs), device=irs.device)
        hard_irs[arange_ir, onset] = 1
        return hard_irs


class ZeroPhaseFIR(nn.Module):
    """
    ZeroPhaseFIR module that performs zero-phase filtering using the Fast Fourier Transform (FFT).

    Args:
        num_magnitude_bins (int): Number of magnitude bins for the FIR filter. Default is 1024.

    Attributes:
        num_magnitude_bins (int): Number of magnitude bins for the FIR filter.
        fir_len (int): Length of the FIR filter.
        window (torch.Tensor): Hann window used for the FIR filter.

    """

    def __init__(
        self,
        num_magnitude_bins=1024,
    ):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.fir_len = 2 * num_magnitude_bins - 1

        window = torch.hann_window(self.fir_len)
        window = window.view(1, -1)
        self.register_buffer("window", window)

    def forward(self, log_mag):
        """
        Forward pass of the ZeroPhaseFIR module.

        Args:
            log_mag (torch.Tensor): Log-magnitude spectrogram.

        Returns:
            torch.Tensor: Zero-phase filtered output.

        """
        shape = log_mag.shape
        shape, f = shape[:-1], shape[-1]
        log_mag = log_mag.view(-1, f)
        mag = torch.exp(log_mag)
        ir = torch.fft.irfft(mag, n=self.fir_len)
        ir = torch.roll(ir, shifts=self.num_magnitude_bins - 1, dims=-1)
        ir = ir * self.window
        ir = ir.view(*shape, -1)
        return ir
