import torch
import torch.nn as nn

from grafx.processors.core.fft_filterbank import fft_triangular_filterbank


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


def log_energy_to_zerophase_fir_with_filterbank(
    log_energy,
    fir_len,
    filterbank,
    window=None,
    eps=1e-7,
):
    shape = log_energy.shape
    shape, f = shape[:-1], shape[-1]
    log_energy = log_energy.view(-1, f)
    energy = torch.exp(log_energy)
    magnitude = torch.matmul(filterbank, energy)
    magnitude = torch.sqrt(magnitude + eps)
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
        num_energy_bins=1024,
        scale="bark_traunmuller",
        n_filters=80,
        f_min=40,
        f_max=None,
        sr=None,
        eps=1e-7,
        window="hann",
        **window_kwargs,
    ):
        super().__init__()

        assert (
            f_max is not None or sr is not None
        ), "Either f_max or sr must be provided."

        self.num_energy_bins = num_energy_bins
        self.fir_len = 2 * num_energy_bins - 1
        self.eps = eps

        filterbank = fft_triangular_filterbank(
            n_freqs=num_energy_bins,
            f_min=f_min,
            f_max=f_max,
            n_filters=n_filters,
            scale=scale,
            attach_remaining=True,
        )
        filterbank = filterbank.T
        self.register_buffer("filterbank", filterbank)

        if isinstance(window, torch.Tensor):
            self.register_buffer("window", window)
        else:
            match window:
                case "rectangular" | "none" | "boxcar" | None:
                    self.window = None
                case _:
                    window = get_window(
                        window_type=window, fir_len=self.fir_len, **window_kwargs
                    )
                    self.register_buffer("window", window)

    def forward(self, filterbank_log_energy):
        return log_energy_to_zerophase_fir_with_filterbank(
            filterbank_log_energy,
            fir_len=self.fir_len,
            window=self.window,
            filterbank=self.filterbank,
            eps=self.eps,
        )
