import torch
import torch.nn as nn
from einops import rearrange
from torch.fft import irfft

from grafx.processors.components import CausalConvolution, SurrogateDelay
from grafx.processors.functional import convolve, normalize_impulse


class StereoMultitapDelay(nn.Module):
    r"""
    A stereo delay module comprising feed-forward delays, each with learnable delay length.

        Simliar to the other LTI processors, we first compute the FIR of the processor and convolve it with the input.
        The multitap delay's FIR is given as
        $$
        h[n] = \sum_{m=1}^{M} \underbrace{c_m[n]}_{\mathrm{optional}}*\delta[n-d_m]
        $$

        where $\delta[n]$ denotes a unit impulse and $c_m[n]$ is an optional coloration filter.

        Here, we aim to optimize each *discrete* delay length $d_m \in \mathbb{N}$ using gradient descent.
        To this end, we exploit the fact that each delay $\delta[n-d_m]$ corresponds to a complex sinusoid in the frequency domain.
        Such a sinusoid's angular frequency $z_m \in \mathbb{C}$ can be optimized with the gradient descent if we allow it to be inside the unit disk, i.e., $|z_m| \leq 1$ :cite:`hayes2023sinusoidal`.
        Hence, for each delay, we compute a damped sinusoid with the *continuous* frequency $z_m$ then use its IFFT as a surrogate soft delay.
        $$
        \delta[n-d_m] \approx \frac{1}{N} \sum_{k=0}^{N-1} z_m^k w_N^{kn}.
        $$

        where $0 \leq n < N$ and $w_{N} = \exp(j\cdot 2\pi/N)$.
        Note that, instead of allowing the delays to have the full range (from $0$ to $N-1$),
        we can restrict them to have a smaller range and then concatenate them to form a longer multitap delay
        (see the arguments, e.g., :python:`segment_len` and :python:`num_segments` below).
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
        straight_through (:python:`bool`, *optional*):
            Use hard delays for the forward passes and surrogate soft delays for the backward passes
            with straight-through estimation :cite:`bengio2013estimating`
            (default: :python:`True`).
        flashfftconv (:python:`bool`, *optional*):
            An option to use :python:`FlashFFTConv` :cite:`fu2023flashfftconv` as a backend 
            to perform the causal convolution efficiently (default: :python:`True`).
        max_input_len (:python:`int`, *optional*):
            When :python:`flashfftconv` is set to :python:`True`, 
            the max input length must be also given (default: :python:`2**17`).
    """

    def __init__(
        self,
        segment_len=3000,
        num_segments=20,
        num_delay_per_segment=1,
        stereo=True,
        zp_filter_per_tap=True,
        zp_filter_bins=20,
        straight_through=True,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        self.segment_len = segment_len
        self.num_segments = num_segments
        self.num_delay_per_segment = num_delay_per_segment
        self.stereo = stereo

        self.zp_filter_per_tap = zp_filter_per_tap
        self.zp_filter_bins = zp_filter_bins
        self.zp_filter_len = zp_filter_bins * 2 - 1
        window = torch.hann_window(self.zp_filter_len)
        window = window.view(1, 1, -1)
        self.register_buffer("window", window)

        self.delay = SurrogateDelay(
            N=segment_len,
            straight_through=straight_through,
        )

        self.conv = CausalConvolution(
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

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
        output_sinals = self.conv(input_signals, ir)
        return output_sinals, radii_loss

    def get_ir(self, delay_z, log_fir_magnitude):
        z_c = torch.view_as_complex(delay_z)
        irs, radii_loss = self.delay(z_c)

        if self.zp_filter_per_tap:
            color_firs = self.get_color_fir(log_fir_magnitude)
            irs = convolve(irs, color_firs, mode="zerophase")

        irs = rearrange(
            irs,
            "b (c m p) t -> b c m p t",
            c=2 if self.stereo else 1,
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

    def get_color_fir(self, fir_db):
        eq = torch.exp(fir_db)
        eq = irfft(eq, n=self.zp_filter_len)
        eq = torch.roll(eq, shifts=self.zp_filter_bins - 1, dims=-1)
        eq = self.window * eq
        return eq

    def parameter_size(self):
        """
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        num_delay = self.num_segments * self.num_delay_per_segment
        if self.stereo:
            num_delay *= 2
        size = {"delay_z": (num_delay, 2)}
        if self.zp_filter_per_tap:
            size["log_fir_magnitude"] = (num_delay, self.zp_filter_bins)
        return size
