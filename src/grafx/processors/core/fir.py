import torch
import torch.nn as nn

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