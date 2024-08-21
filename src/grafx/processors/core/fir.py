import torch
import torch.nn as nn

from grafx.processors.core.fft_filterbank import TriangularFilterBank


def get_window(window_type, window_length, **kwargs):
    match window_type:
        case "hann":
            return torch.hann_window(window_length, **kwargs)
        case "hamming":
            return torch.hamming_window(window_length, **kwargs)
        case "blackman":
            return torch.blackman_window(window_length, **kwargs)
        case "bartlett":
            return torch.bartlett_window(window_length, **kwargs)
        case "kaiser":
            return torch.kaiser_window(window_length, **kwargs)
        case "rectangular" | "none" | "boxcar" | None:
            return None
        case _:
            raise ValueError(f"Unsupported window type: {window_type}")


def log_magnitude_to_zerophase_fir(
    log_magnitude,
    fir_len,
    window=None,
):
    shape = log_magnitude.shape
    shape, f = shape[:-1], shape[-1]
    log_magnitude = log_magnitude.view(-1, f)
    magnitude = torch.exp(log_magnitude)
    ir = torch.fft.irfft(magnitude, n=fir_len)
    shifts = fir_len // 2
    ir = torch.roll(ir, shifts=shifts, dims=-1)
    if window is not None:
        ir = ir * window[None, :]
    ir = ir.view(*shape, -1)
    return ir


class ZeroPhaseFIR(nn.Module):
    r"""
    Creates a simple zero-phase FIR from a log-magnitude response.
    """

    def __init__(
        self,
        num_magnitude_bins=1024,
        window="hann",
        **window_kwargs,
    ):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.fir_len = 2 * num_magnitude_bins - 1

        if isinstance(window, torch.Tensor):
            self.register_buffer("window", window)
        else:
            match window:
                case "rectangular" | "none" | "boxcar" | None:
                    self.window = None
                case _:
                    window = get_window(
                        window_type=window, window_length=self.fir_len, **window_kwargs
                    )
                    self.register_buffer("window", window)

    def forward(self, log_magnitude):
        return log_magnitude_to_zerophase_fir(
            log_magnitude, fir_len=self.fir_len, window=self.window
        )


class ZeroPhaseFilterBankFIR(nn.Module):
    def __init__(
        self,
        num_frequency_bins=1024,
        use_filterbank=False,
        filterbank_kwargs={},
        window="hann",
        window_kwargs={},
        eps=1e-7,
    ):
        super().__init__()

        self.num_frequency_bins = num_frequency_bins
        self.fir_len = 2 * num_frequency_bins - 1
        self.eps = eps

        self.use_filterbank = use_filterbank
        if self.use_filterbank:
            self.filterbank = TriangularFilterBank(
                num_frequency_bins=num_frequency_bins, **filterbank_kwargs
            )

        if isinstance(window, torch.Tensor):
            self.register_buffer("window", window)
        else:
            window = get_window(
                window_type=window, window_length=self.fir_len, **window_kwargs
            )
            self.register_buffer("window", window)

    def forward(self, log_magnitude):
        shape = log_magnitude.shape
        shape, f = shape[:-1], shape[-1]
        log_magnitude = log_magnitude.view(-1, f)

        magnitude = torch.exp(log_magnitude)
        if self.use_filterbank:
            energy = magnitude.square()
            energy = self.filterbank(energy)
            magnitude = torch.sqrt(energy + self.eps)
        ir = torch.fft.irfft(magnitude, n=self.fir_len)
        shifts = self.fir_len // 2
        ir = torch.roll(ir, shifts=shifts, dims=-1)
        if self.window is not None:
            ir = ir * self.window[None, :]

        ir = ir.view(*shape, -1)
        return ir
