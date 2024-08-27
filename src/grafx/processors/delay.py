import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from grafx.processors.core.convolution import FIRConvolution, convolve
from grafx.processors.core.delay import SurrogateDelay
from grafx.processors.core.fir import ZeroPhaseFIR
from grafx.processors.core.utils import normalize_impulse


class MultitapDelay(nn.Module):
    r"""
    A stereo delay module comprising feed-forward delays, each with learnable delay length.

        Simliar to the other LTI processors, we first compute the FIR of the processor and convolve it with the input.
        The multitap delay's FIR is given as
        $$
        h[n] = \sum_{m=1}^{M} \underbrace{c_m[n]}_{\mathrm{optional}}*\delta[n-d_m]
        $$

        where $\delta[n]$ denotes a unit impulse and $c_m[n]$ is an optional coloration filter.
        The delays lengths are optimized with the surrogate delay lines: see :class:`~grafx.processors.core.delay.SurrogateDelay`.
        Instead of allowing the delays to have the full range (from $0$ to $N-1$),
        we can restrict them to have a smaller range and then concatenate them to form a multitap delay;
        see the arguments :python:`segment_len` and :python:`num_segments` below.
        This multitap delay's learnable parameter is $p = \{\mathbf{z}, \mathbf{H}\}$ where the latter is optional
        log-magnitude responses of the coloration filters.

    Args:
        segment_len (:python:`int`, *optional*):
            The length of the segment for each delay
            (default: :python:`3000`).
        num_segments (:python:`int`, *optional*):
            The number of segments for each channel
            (default: :python:`20`).
        num_delay_per_segment (:python:`int`, *optional*):
            The number of delay taps per segment
            (default: :python:`1`).
        stereo (:python:`bool`, *optional*):
            Use two independent delays for left and right.
            (default: :python:`True`).
        zp_filter_per_tap (:python:`bool`, *optional*):
            Use a :class:`~grafx.processors.eq.ZeroPhaseFIREqualizer` for each tap
            (default: :python:`True`).
        zp_filter_bins (:python:`int`, *optional*):
            The number of bins for each equalizer
            (default: :python:`20`).
        flashfftconv (:python:`bool`, *optional*):
            An option to use :python:`FlashFFTConv` :cite:`fu2023flashfftconv` as a backend
            to perform the causal convolution efficiently (default: :python:`True`).
        max_input_len (:python:`int`, *optional*):
            When :python:`flashfftconv` is set to :python:`True`,
            the max input length must be also given (default: :python:`2**17`).
        **surrgate_delay_kwargs (*optional*):
            Additional arguments for the :class:`~grafx.processors.core.delay.SurrogateDelay` module.
    """

    def __init__(
        self,
        segment_len=3000,
        num_segments=20,
        num_delay_per_segment=1,
        processor_channel="stereo",
        zp_filter_per_tap=True,
        zp_filter_bins=20,
        flashfftconv=True,
        max_input_len=2**17,
        pre_delay=0,
        **surrogate_delay_kwargs,
    ):
        super().__init__()
        self.segment_len = segment_len
        self.num_segments = num_segments
        self.num_delay_per_segment = num_delay_per_segment

        self.zp_filter_per_tap = zp_filter_per_tap
        if self.zp_filter_per_tap:
            self.zp_filter = ZeroPhaseFIR(zp_filter_bins)

        self.zp_filter_bins = zp_filter_bins
        self.zp_filter_len = zp_filter_bins * 2 - 1
        window = torch.hann_window(self.zp_filter_len)
        window = window.view(1, 1, -1)
        self.register_buffer("window", window)

        self.delay = SurrogateDelay(
            N=segment_len,
            **surrogate_delay_kwargs,
        )

        self.num_channelsonv = FIRConvolution(
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

        self.pre_delay = pre_delay

        self.processor_channel = processor_channel
        match self.processor_channel:
            case "mono":
                self.process = self._process_mono_stereo
                self.num_channels = 1
            case "stereo":
                self.process = self._process_mono_stereo
                self.num_channels = 2
            case "midside":
                self.process = self._process_midside
                self.num_channels = 2

    def forward(self, input_signals, delay_z, log_fir_magnitude=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times 2 \times L`):
                A batch of input audio signals.
            delay_z (:python:`FloatTensor`, :math:`B \times M \times 2`):
                A log-magnitude vector of the FIR filter.
            log_fir_magnitude (:python:`FloatTensor`, :math:`B \times M \times P`, *optional*):
                A log-magnitude vector of the FIR filter.
                Must be given when :python:`zp_filter_per_tap` is set to :python:`True`.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times 2 \times L`.
        """
        ir, radii_loss = self.get_ir(delay_z, log_fir_magnitude)
        output_signals = self.num_channelsonv(input_signals, ir)
        if self.pre_delay != 0:
            output_signals = F.pad(output_signals, (self.pre_delay, 0))
            output_signals = output_signals[:, :, : -self.pre_delay]
        return output_signals, radii_loss

    def get_ir(self, delay_z, log_fir_magnitude):
        z_c = torch.view_as_complex(delay_z)
        irs, radii_loss = self.delay(z_c)

        if self.zp_filter_per_tap:
            color_firs = self.zp_filter(log_fir_magnitude)
            irs = convolve(irs, color_firs, mode="zerophase")

        irs = rearrange(
            irs,
            "b (c m p) t -> b c m p t",
            c=self.num_channels,
            m=self.num_segments,
            p=self.num_delay_per_segment,
        )
        irs = irs.sum(-2)
        irs = rearrange(
            irs,
            "b c m t -> b c (m t)",
        )
        irs = normalize_impulse(irs)
        loss = {"radii_reg": radii_loss}
        return irs, loss

    def _process_mono_stereo(self, input_signals, fir):
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)

    def _process_midside(self, input_signals, fir):
        fir = normalize_impulse(fir)
        input_signals = lr_to_ms(input_signals)
        output_signals = self.conv(input_signals, fir)
        return ms_to_lr(output_signals)

    def parameter_size(self):
        """
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        num_delay = self.num_segments * self.num_delay_per_segment * self.num_channels
        size = {"delay_z": (num_delay, 2)}
        if self.zp_filter_per_tap:
            size["log_fir_magnitude"] = (num_delay, self.zp_filter_bins)
        return size
