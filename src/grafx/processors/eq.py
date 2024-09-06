import math

import torch
import torch.nn as nn

from grafx.processors.core.convolution import convolve
from grafx.processors.core.fir import ZeroPhaseFilterBankFIR, ZeroPhaseFIR
from grafx.processors.core.geq import GraphicEqualizerBiquad
from grafx.processors.core.iir import IIRFilter
from grafx.processors.core.midside import lr_to_ms, ms_to_lr
from grafx.processors.filter import (
    BaseParametricEqualizerFilter,
    HighShelf,
    LowShelf,
    PeakingFilter,
)

PI = math.pi
TWO_PI = 2 * math.pi
HALF_PI = math.pi / 2
TWOR_SCALE = 1 / math.log(2)
ALPHA_SCALE = 1 / 2


class ZeroPhaseFIREqualizer(nn.Module):
    r"""
    A single-channel zero-phase finite impulse response (FIR) filter :cite:`smith2007introduction, smith2011spectral, engel2020ddsp`.

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


class NewZeroPhaseFIREqualizer(nn.Module):
    r"""
    A single-channel zero-phase finite impulse response (FIR) filter :cite:`smith2007introduction, smith2011spectral, engel2020ddsp`.

        From the input log-magnitude $H_{\mathrm{log}}$,
        we compute inverse FFT (IFFT) of the magnitude response
        and multiply it with a zero-centered window $w[n]$.
        Each input channel is convolved with the following FIR.
        $$
        h[n] = w[n] \cdot \frac{1}{N} \sum_{k=0}^{N-1} \exp H_{\mathrm{log}}[k] \cdot  z_{N}^{kn}.
        $$

        Here, $-(N+1)/2 \leq n \leq (N+1)/2$ and $z_{N} = \exp(j\cdot 2\pi/N)$.
        This equalizer's learnable parameter is $p = \{ H_{\mathrm{log}} \}$.

        From the input log-energy $H_{\mathrm{fb}} \in \mathbb{R}^{K_{\mathrm{fb}}}$,
        we compute the FFT magnitudes as
        $$
        H_{\mathrm{log}} = \sqrt { M \exp (H_{\mathrm{fb}}) + \epsilon}
        $$

        where $M \in \mathbb{R}^{K \times K_{\mathrm{fb}}}$ is the filterbank matrix
        ($K$ and $K_{\mathrm{fb}}$ are the number of FFT magnitude bins and filterbank bins, respectively).
        We use the standard triangular filterbank.
        This equalizer's learnable parameter is $p = \{ H_{\mathrm{fb}} \}$.

    Args:
        num_frequency_bins (:python:`int`, *optional*):
            The number of FFT energy bins (default: :python:`1024`).
        processor_channel (:python:`str`, *optional*):
            The channel configuration of the equalizer,
            which can be :python:`"mono"`, :python:`"stereo"`, :python:`"midside"`, or :python:`"pseudo_midside"`
            (default: :python:`"mono"`).
        filterbank (:python:`bool`, *optional*):
            Whether to use the filterbank (default: :python:`False`).
        scale (:python:`str`, *optional*):
            The frequency scale to use, which can be:
            :python:`"bark_traunmuller"`, :python:`"bark_schroeder"`, :python:`"bark_wang"`,
            :python:`"mel_htk"`, :python:`"mel_slaney"`, :python:`"linear"`, and :python:`"log"`
            (default: :python:`"bark_traunmuller"`).
        n_filters (:python:`int`, *optional*):
            Number of filterbank bins (default: :python:`80`).
        f_min (:python:`float`, *optional*):
            Minimum frequency in Hz. (default: :python:`40`).
        f_max (:python:`float` or :python:`None`, *optional*):
            Maximum frequency in Hz.
            If :python:`None`, the sampling rate :python:`sr` must be provided
            and we use the half of the sampling rate (default: :python:`None`).
        sr (:python:`float` or :python:`None`, *optional*):
            The underlying sampling rate. Only used when using the filterbank
            (default: :python:`None`).
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
        num_frequency_bins=1024,
        processor_channel="mono",
        use_filterbank=False,
        filterbank_kwargs={},
        window="hann",
        window_kwargs={},
        eps=1e-7,
        flashfftconv=False,
    ):
        super().__init__()
        self.num_frequency_bins = num_frequency_bins
        self.processor_channel = processor_channel
        self.fir = ZeroPhaseFilterBankFIR(
            num_frequency_bins=num_frequency_bins,
            use_filterbank=use_filterbank,
            filterbank_kwargs=filterbank_kwargs,
            window=window,
            window_kwargs=window_kwargs,
            eps=eps,
        )
        self.use_filterbank = use_filterbank

        match self.processor_channel:
            case "mono" | "stereo":
                self.process = self._process_mono_stereo
            case "midside":
                self.process = self._process_midside
            case _:
                raise ValueError(f"Invalid processor_channel: {self.processor_channel}")

    def forward(self, input_signals, log_magnitude):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_magnitude (:python:`FloatTensor`, :math:`B \times C_\mathrm{eq} \times K` *or* :math:`B \times C_\mathrm{eq} \times K_\mathrm{fb}`):
                A batch of log-magnitude vectors of the FIR filter.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        fir = self.fir(log_magnitude)
        output_signals = self.process(input_signals, fir)
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        n_bins = (
            self.fir.filterbank.num_filters
            if self.use_filterbank
            else self.num_frequency_bins
        )
        match self.processor_channel:
            case "mono":
                n_channels = 1
            case "stereo" | "midside":
                n_channels = 2
        return {"log_magnitude": (n_channels, n_bins)}

    def _process_mono_stereo(self, input_signals, fir):
        return convolve(input_signals, fir, mode="zerophase")

    def _process_midside(self, input_signals, fir):
        input_signals = lr_to_ms(input_signals)
        output_signals = convolve(input_signals, fir, mode="zerophase")
        return ms_to_lr(output_signals)


class ParametricEqualizer(nn.Module):
    r"""
    A parametric equalizer (PEQ) based on second-order filters.

        We cascade $K$ biquad filters to form a parametric equalizer,
        $$
        H(z) = \prod_{k=1}^{K} H_k(z)
        $$

        By default, $k=1$ and $k=K$ are low-shelf and high-shelf filters, respectively, and the remainings are peaking filters.
        See :class:`~grafx.processors.filter.LowShelf`, :class:`~grafx.processors.filter.PeakingFilter`, and :class:`~grafx.processors.filter.HighShelf` for the filter details.

    Args:
        num_filters (:python:`int`, *optional*):
            The number of filters to use (default: :python:`10`).
        processor_channel (:python:`str`, *optional*):
            The channel configuration of the equalizer,
            which can be :python:`"mono"`, :python:`"stereo"`, or :python:`"midside"` (default: :python:`"mono"`).
        use_shelving_filters (:python:`bool`, *optional*):
            Whether to use a low-shelf and high-shelf filter.
            If false, we use only peaking filters (default: :python:`True`)
            (default: :python:`True`).
        **backend_kwargs (:python:`Dict[str, Any]`, *optional*):
            Additional keyword arguments for the backend.
    """

    def __init__(
        self,
        num_filters=10,
        processor_channel="mono",
        use_shelving_filters=True,
        **backend_kwargs,
    ):
        super().__init__()

        self.num_filters = num_filters
        self.use_shelving_filters = use_shelving_filters
        if self.use_shelving_filters:
            self.split = [1, self.num_filters - 2, 1]
            self.get_biquad_coefficients = (
                self.get_biquad_coefficients_with_shelving_filters
            )
        else:
            self.get_biquad_coefficients = PeakingFilter.get_biquad_coefficients

        self.biquad = IIRFilter(order=2, **backend_kwargs)
        self.processor_channel = processor_channel

        match self.processor_channel:
            case "mono" | "stereo":
                self.process = self._process_mono_stereo
            case "midside":
                self.process = self._process_midside
            case _:
                raise ValueError(f"Invalid processor_channel: {self.processor_channel}")

    def forward(self, input_signals, w0, q_inv, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times K`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times K`):
                A batch of quality factors (or resonance).
            log_gain (:python:`FloatTensor`, :math:`B \times K`):
                A batch of log-gains.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0, q_inv, A = BaseParametricEqualizerFilter.filter_parameter_activations(
            w0, q_inv, log_gain
        )
        cos_w0, alpha = BaseParametricEqualizerFilter.compute_common_filter_parameters(
            w0, q_inv
        )
        Bs, As = self.get_biquad_coefficients(cos_w0, alpha, A)
        output_signal = self.process(input_signals, Bs, As)
        return output_signal

    def get_biquad_coefficients_with_shelving_filters(self, cos_w0, alpha, A):
        cos_w0_ls, cos_w0_peak, cos_w0_hs = torch.split(cos_w0, self.split, dim=2)
        alpha_ls, alpha_peak, alpha_hs = torch.split(alpha, self.split, dim=2)
        A_ls, A_peak, A_hs = torch.split(A, self.split, dim=2)

        Bs_ls, As_ls = LowShelf.get_biquad_coefficients(cos_w0_ls, alpha_ls, A_ls)
        Bs_peak, As_peak = PeakingFilter.get_biquad_coefficients(
            cos_w0_peak, alpha_peak, A_peak
        )
        Bs_hs, As_hs = HighShelf.get_biquad_coefficients(cos_w0_hs, alpha_hs, A_hs)

        Bs = torch.cat([Bs_ls, Bs_peak, Bs_hs], dim=2)
        As = torch.cat([As_ls, As_peak, As_hs], dim=2)

        return Bs, As

    def _process_mono_stereo(self, input_signals, Bs, As):
        return self.biquad(input_signals, Bs, As)

    def _process_midside(self, input_signals, Bs, As):
        input_signals = lr_to_ms(input_signals)
        output_signals = self.biquad(input_signals, Bs, As)
        return ms_to_lr(output_signals)

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        match self.processor_channel:
            case "mono":
                n_channels = 1
            case "stereo" | "midside":
                n_channels = 2

        size = (n_channels, self.num_filters)
        return {k: size for k in ["w0", "q_inv", "log_gain"]}


class GraphicEqualizer(nn.Module):
    r"""
    A graphic equalizer (GEQ) based on second-order peaking filters :cite:`liski2017quest`.

        We cascade $K$ biquad filters to form a graphic equalizer,
        whose transfer function is given as $H(z) = \prod_{k=1}^{K} H_k(z)$ where each biquad $H_k(z)$ is as follows,
        $$
        H_k(z)=\frac{1+g_k \beta_k-2 \cos (\omega_k) z^{-1}+(1-g_k \beta_k) z^{-2}}{1+\beta_k-2 \cos (\omega_k) z^{-1}+(1-\beta_k) z^{-2}}.
        $$

        Here, $g_k$ is the linear gain and $\omega_k$ is the center frequency.
        $\beta_k$ is given as
        $$
        \beta_k = \sqrt{\frac{\left|\tilde{g}_k^2-1\right|}{\left|g_k^2-\tilde{g}_k^2\right|}} \tan {\frac{B_k}{2}}
        $$

        where $B_k$ is the bandwidth frequency and $\tilde{g}_k$ is the gain at the neighboring band frequency,
        pre-determined to be $\tilde{g}_k = g_k^{0.4}$.
        The frequency values ($\omega_k$ and $B_k$) and the number of bands $K$ are also determined by the frequency scale.
        The learnable parameter is a concatenation of the log-magnitudes, i.e., $\smash{p = \{ \mathbf{g}^{\mathrm{log}} \}}$ where $\smash{g_k = \exp g_k^{\mathrm{log}}}$.

        Note that the log-gain parameters are different to the equalizer's log-magnitude response values at the center frequencies known as "control points".
        To set the log-gains to match the control points, we can use least-square optimization methods :cite:`liski2017quest, valimaki2019neurally`.


    Args:
        scale (:python:`str`, *optional*):
            The frequency scale to use, which can be:
            24-band :python:`"bark"` and 31-band :python:`"third_oct"` (default: :python:`"bark"`).
        sr (:python:`int`, *optional*):
            The underlying sampling rate of the input signal (default: :python:`44100`).
        backend (:python:`str`, *optional*):
            The backend to use for the filtering, which can either be the frequency-sampling method
            :python:`"fsm"` or exact time-domain filter :python:`"lfilter"` (default: :python:`"fsm"`).
        fsm_fir_len (:python:`int`, *optional*):
            The length of FIR approximation when :python:`backend == "fsm"` (default: :python:`8192`).
    """

    def __init__(
        self,
        processor_channel="mono",
        scale="bark",
        sr=44100,
        **backend_kwargs,
    ):
        super().__init__()
        self.geq = GraphicEqualizerBiquad(scale=scale, sr=sr)
        self.biquad = IIRFilter(**backend_kwargs)

        self.processor_channel = processor_channel

        match self.processor_channel:
            case "mono" | "stereo":
                self.process = self._process_mono_stereo
            case "midside":
                self.process = self._process_midside
            case _:
                raise ValueError(f"Invalid processor_channel: {self.processor_channel}")

    def forward(self, input_signals, log_gains):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_gains (:python:`FloatTensor`, :math:`B \times K \:\!`):
                A batch of log-gain vectors of the GEQ.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        Bs, As = self.geq(log_gains)
        output_signal = self.process(input_signals, Bs, As)
        return output_signal

    def _process_mono_stereo(self, input_signals, Bs, As):
        return self.biquad(input_signals, Bs, As)

    def _process_midside(self, input_signals, Bs, As):
        input_signals = lr_to_ms(input_signals)
        output_signals = self.biquad(input_signals, Bs, As)
        return ms_to_lr(output_signals)

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        num_bands = self.geq.num_bands

        match self.processor_channel:
            case "mono":
                n_channels = 1
            case "stereo" | "midside":
                n_channels = 2

        return {"log_gains": (n_channels, num_bands)}
