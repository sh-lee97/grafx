import torch.nn as nn

from grafx.processors.core.convolution import convolve
from grafx.processors.core.fir import ZeroPhaseFilterBankFIR, ZeroPhaseFIR


class ZeroPhaseFIREqualizer(nn.Module):
    r"""
    A single-channel zero-phase finite impulse response (FIR) filter.

        From the input log-magnitude $H_{\mathrm{log}}$,
        we compute inverse FFT (IFFT) of the magnitude response
        and multiply it with a zero-centered window $v[n]$.
        Each input channel is convolved with the following FIR.
        $$
        h[n] = v[n] \cdot \frac{1}{N} \sum_{k=0}^{N-1} \exp H_{\mathrm{log}}[k] \cdot  w_{N}^{kn}.
        $$

        Here, $-(N+1)/2 \leq n \leq (N+1)/2$ and $w_{N} = \exp(j\cdot 2\pi/N)$.
        This equalizer's learnable parameter is $p = \{ H_{\mathrm{log}} \}$.

    Args:
        num_magnitude_bins (:python:`int`, *optional*):
            The number of FFT magnitude bins (default: :python:`1024`).
        window (:python:`str` or :python:`FloatTensor`, *optional*):
            The window function to use for the FIR filter.
            If :python:`str` is given, we create the window internally.
            It can be: :python:`"hann"`, :python:`"hamming"`, :python:`"blackman"`, :python:`"bartlett"`, and :python:`"kaiser"`.
            If :python:`FloatTensor` is given, we use it as a window (default: :python:`"hann"`).
        **window_kwargs (:python:`Dict[str, Any]`, *optional*):
            Additional keyword arguments for the window function.
    """

    def __init__(self, num_magnitude_bins=1024):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.fir = ZeroPhaseFIR(num_magnitude_bins)

    def forward(self, input_signals, log_magnitude):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_magnitude (:python:`FloatTensor`, :math:`B \times K \:\!`):
                A batch of log-magnitude vectors of the FIR filter.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        fir = self.fir(log_magnitude)[:, None, :]
        output_signals = convolve(input_signals, fir, mode="zerophase")
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"log_magnitude": self.num_magnitude_bins}


class ZeroPhaseFilterBankFIREqualizer(nn.Module):
    r"""
    A zero-phase FIR filter parameterized with a filterbank log-magnitudes.

        From the input log-energy $H_{\mathrm{fb}} \in \mathbb{R}^{K_{\mathrm{fb}}}$,
        we compute the FFT magnitudes as
        $$
        H_{\mathrm{log}} = \sqrt { M \exp (H_{\mathrm{fb}}) + \epsilon}
        $$

        where $M \in \mathbb{R}^{K \times K_{\mathrm{fb}}}$ is the filterbank matrix
        ($K$ and $K_{\mathrm{fb}}$ are the number of FFT magnitude bins and filterbank bins, respectively).
        We use the standard triangular filterbank.
        After obtaining the log-magnitude response, the remaining process is the same as :class:`~grafx.processors.eq.ZeroPhaseFIREqualizer`.
        This equalizer's learnable parameter is $p = \{ H_{\mathrm{fb}} \}$.

    Args:
        num_energy_bins (:python:`int`, *optional*):
            The number of FFT energy bins (default: :python:`1024`).
        f_min (:python:`float`, *optional*):
            Minimum frequency in Hz. (default: :python:`40`).
        f_max (:python:`float` or :python:`None`, *optional*):
            Maximum frequency in Hz.
            If :python:`None`, the sampling rate :python:`sr` must be provided
            and we use the half of the sampling rate (default: :python:`None`).
        sr (:python:`float` or :python:`None`, *optional*):
            The underlying sampling rate of the input signal (default: :python:`None`).
        n_filters (:python:`int`, *optional*):
            Number of filterbank bins (default: :python:`80`).
        scale (:python:`str`, *optional*):
            The frequency scale to use, which can be:
            :python:`"bark_traunmuller"`, :python:`"bark_schroeder"`, :python:`"bark_wang"`,
            :python:`"mel_htk"`, :python:`"mel_slaney"`, :python:`"linear"`, and :python:`"log"`
            (default: :python:`"bark_traunmuller"`).
        window (:python:`str` or :python:`FloatTensor`, *optional*):
            The window function to use for the FIR filter.
            If :python:`str` is given, we create the window internally.
            It can be: :python:`"hann"`, :python:`"hamming"`, :python:`"blackman"`, :python:`"bartlett"`, and :python:`"kaiser"`.
            If :python:`FloatTensor` is given, we use it as a window (default: :python:`"hann"`).
        **window_kwargs (:python:`Dict[str, Any]`, *optional*):
            Additional keyword arguments for the window function.
    """

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
        self.num_energy_bins = num_energy_bins
        self.eps = eps
        self.fir = ZeroPhaseFilterBankFIR(
            num_energy_bins=num_energy_bins,
            scale=scale,
            n_filters=n_filters,
            f_min=f_min,
            f_max=f_max,
            sr=sr,
            window=window,
            **window_kwargs,
        )

    def forward(self, input_signals, log_magnitude):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_magnitude (:python:`FloatTensor`, :math:`B \times K_{\mathrm{fb}}`):
                A batch of log-magnitude vectors of the FIR filter.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        fir = self.fir(log_magnitude)[:, None, :]
        output_signals = convolve(input_signals, fir, mode="zerophase")
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"log_energy": self.num_energy_bins}
