import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from grafx.processors.core.convolution import FIRConvolution
from grafx.processors.core.midside import lr_to_ms, ms_to_lr
from grafx.processors.core.noise import get_filtered_noise
from grafx.processors.core.utils import normalize_impulse


class STFTMaskedNoiseReverb(nn.Module):
    r"""
    A filtered noise model :cite:`engel2020ddsp` (or pseudo-random noise method) with mid/side controls.

        We employ two fixed-length uniform noise signal, $v_{\mathrm{m}}[n]$ and $v_{\mathrm{s}}[n] \sim \mathcal{U}[-1, 1)$, that correpond to a mid and side chananels, respoenctively.
        Next, we apply a magnitude mask $M_{\mathrm{x}}[k, m] \in \mathbb{R}^{K\times M}$ to each noise's short-time Fourier transform (STFT) $V_{\mathrm{x}}[k, m] \in \mathbb{C}^{K\times M}$.
        $$
        H_{\mathrm{x}}[k, m] = V_{\mathrm{x}}[k, m] \odot M_{\mathrm{x}}[k, m] \quad (\mathrm{x} \in \{\mathrm{m}, \mathrm{s}\}).
        $$

        Here, $k$ and $m$ denote frequency and time frame index, respectively.
        Each mask is parameterized with an initial $H^0_{\mathrm{x}}[k] \in \mathbb{R}^K$ and an absorption filter $H^\Delta_{\mathrm{x}}[k] \in \mathbb{R}^K$ both in log-magnitudes.
        Also, a frequency-independent gain enevelope $G_{\mathrm{x}}[m] \in \mathbb{R}^{M}$ can be optionally added.
        $$
        M_{\mathrm{x}}[k, m] = \exp ({H^0_{\mathrm{x}}[k] + (m-1) H^\Delta_{\mathrm{x}}[k]} + \underbrace{G_{\mathrm{x}}[m]}_{\mathrm{optional}}).
        $$
        Next, we convert the masked noises to the time-domain responses, $h_\mathrm{m}[n]$ and $h_\mathrm{s}[n]$, via inverse STFT.
        We obtain the desired FIR $h[n]$ by converting the mid/side to stereo.
        Finally, we apply channel-wise convolutions (not a full 2-by-2 stereo convolution) to the input $u[n]$ and obtain the wet output $y[n]$.
        Hence, the learnable parameter is
        $p = \{ H^0_{\mathrm{m}}, H^0_{\mathrm{s}}, H^\Delta_{\mathrm{m}}, H^\Delta_{\mathrm{s}}, G_{\mathrm{m}}, G_{\mathrm{s}} \}$
        where the latter two are optional.

    Args:
        ir_len (:python:`int`, *optional*):
            The length of the impulse response (default: :python:`60000`).
        n_fft (:python:`int`, *optional*):
            FFT size of the STFT (default: :python:`384`).
        hop_length (:python:`int`, *optional*):
            Hop length of the STFT (default: :python:`192`).
        fixed_noise (:python:`bool`, *optional*):
            If set to :python:`True`, we use fixed-seed random noises ($v_{\mathrm{m}}[n]$ and $v_{\mathrm{s}}[n]$) for every forward pass.
            If set to :python:`False`, we create different uniform noises for every forward pass (default: :python:`True`).
        gain_envelope (:python:`bool`, *optional*):
            If set to :python:`True`, we use the log-magnitude gain envelope $G[m]$ (default: :python:`False`).
        flashfftconv (:python:`bool`, *optional*):
            An option to use :python:`FlashFFTConv` :cite:`fu2023flashfftconv` as a backend to perform the causal convolution efficiently (default: :python:`True`).
        max_input_len (:python:`int`, *optional*):
            When :python:`flashfftconv` is set to :python:`True`, the max input length must be also given (default: :python:`2**17`).

    """

    def __init__(
        self,
        ir_len=60000,
        processor_channel="pseudo_midside",
        n_fft=384,
        hop_length=192,
        fixed_noise=True,
        gain_envelope=False,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        self.ir_len = ir_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = 1 + (ir_len // hop_length)
        self.num_bins = 1 + n_fft // 2

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        arange = torch.arange(self.num_frames).view(1, 1, 1, -1)
        self.register_buffer("arange", arange)

        self.fixed_noise = fixed_noise
        if self.fixed_noise:
            self.get_fixed_noise()

        self.gain_envelope = gain_envelope

        self.conv = FIRConvolution(
            mode="causal",
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

        self.processor_channel = processor_channel
        match self.processor_channel:
            case "mono" | "stereo":
                self.process = self._process_mono_stereo
            case "midside":
                self.process = self._process_midside
            case "pseudo_midside":
                self.process = self._process_pseudo_midside

    def get_fixed_noise(self):
        rng = np.random.RandomState(0)
        noise = rng.uniform(size=(2, self.ir_len))
        noise = noise * 2 - 1
        noise = torch.tensor(noise).float()
        noise_stft = torch.stft(
            noise,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        noise_stft = rearrange(noise_stft, "c f t -> 1 c f t")
        self.register_buffer("noise_stft", noise_stft)

    def sample_noise(self, num_noises, device):
        noise = torch.rand(num_noises * 2, self.ir_len, device=device) * 2 - 1

        noise_stft = torch.stft(
            noise,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        noise_stft = rearrange(noise_stft, "(b c) f t -> b c f t", c=2)
        return noise_stft

    def forward(
        self,
        input_signals,
        init_log_magnitude,
        delta_log_magnitude,
        gain_env_log_magnitude=None,
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B\times 2\times L`):
                A batch of input audio signals.
            init_log_magnitude (:python:`FloatTensor`, :math:`B\times 2\times K \:\!`):
                A batch of log-magnitudes of the initial filters.
                We assume that the mid- and side-channel responses,
                :math:`H^0_{\mathrm{m}}` and :math:`H^0_{\mathrm{s}}` repectively,
                are stacked together (the same applies to the remaining tensors).
            delta_log_magnitude (:python:`FloatTensor`, :math:`B\times 2\times K \:\!`):
                A batch of log-magnitudes of the absorption filters.
            gain_env_log_magnitude (:python:`FloatTensor`, :math:`B\times 2\times M`, *optional*):
                A batch of log-gain envelopes.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        ir = self.compute_ir(
            init_log_magnitude, delta_log_magnitude, gain_env_log_magnitude
        )
        output_signals = self.process(input_signals, ir)
        return output_signals

    def compute_ir(
        self, init_log_magnitude, delta_log_magnitude, gain_env_log_magnitude=None
    ):

        if self.fixed_noise:
            noise_stft = self.noise_stft
        else:
            b, device = init_log_magnitude.shape[0], init_log_magnitude.device
            noise_stft = self.sample_noise(b, device)

        mask = self.compute_stft_mask(
            init_log_magnitude, delta_log_magnitude, gain_env_log_magnitude
        )
        ir_stft = noise_stft * mask

        ir_stft = rearrange(ir_stft, "b c f t -> (b c) f t")
        ir = torch.istft(
            ir_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=self.ir_len,
        )
        ir = rearrange(ir, "(b c) t -> b c t", c=2)

        # ir = self.ms_to_lr(ir)
        return ir

    def compute_stft_mask(
        self, init_log_magnitude, delta_log_magnitude, gain_env_log_magnitude=None
    ):
        init_log_magnitude = init_log_magnitude[:, :, :, None]
        delta_log_magnitude = -F.softplus(delta_log_magnitude)[:, :, :, None]
        mask_log_magnitude = init_log_magnitude + delta_log_magnitude * self.arange
        if self.gain_envelope:
            mask_log_magnitude = (
                mask_log_magnitude + gain_env_log_magnitude[:, :, None, :]
            )
        mask = torch.exp(mask_log_magnitude / 8)
        return mask

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {
            "init_log_magnitude": (2, self.num_bins),
            "delta_log_magnitude": (2, self.num_bins),
        }
        if self.gain_envelope:
            size["gain_env_log_magnitude"] = (2, self.num_frames)
        return size

    def _process_mono_stereo(self, input_signals, fir):
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)

    def _process_midside(self, input_signals, fir):
        fir = normalize_impulse(fir)
        input_signals = lr_to_ms(input_signals)
        output_signals = self.conv(input_signals, fir)
        return ms_to_lr(output_signals)

    def _process_pseudo_midside(self, input_signals, fir):
        fir = ms_to_lr(fir)
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)


class FilteredNoiseShapingReverb(nn.Module):
    r"""
    A time-domain FIR filter based on the envelope "shaping" of filterbank noise signals
    :cite:`steinmetz2021filtered`.

        From a noise signal $v[n] \sim \mathcal{U}[-1, 1)$,
        we apply a $K$-band filterbank to obtain a set of filtered noise signals $v_1[n], \cdots, v_K[n]$.
        Then, we apply a time-domain envelope shaping, $a_i[n]$, to each filtered noise signal as follows,
        $$
        h[n] = \sum_{i=1}^K a_i[n] v_i[n].
        $$

        Each envelope shaping is parameterized by a decay $r_i$ and an initial gain $g_i$.
        Furthermore, we can set a fade-in envelope to the shaping which is set to be shorter than the decay time.



    Args:
        ir_len (:python:`int`, *optional*):
            The length of the impulse response (default: :python:`60000`).
        num_bands (:python:`int`, *optional*):
            The number of frequency bands (default: :python:`12`).
        processor_channel (:python:`str`, *optional*):
            The channel type of the processor, either :python:`"midside"`, :python:`"stereo"`, or :python:`"mono"` (default: :python:`"midside"`
        f_min (:python:`float`, *optional*):
            The minimum frequency of the filtered noise (default: :python:`31.5`).
        f_max (:python:`float`, *optional*):
            The maximum frequency of the filtered noise (default: :python:`15000`).
        scale (:python:`str`, *optional*):
            Frequency scale to use: :python:`"bark_traunmuller"`, :python:`"bark_schroeder"`, :python:`"bark_wang"`, :python:`"mel_htk"`, :python:`"mel_slaney"`, :python:`"linear"`, :python:`"log"` 
            (default: :python:`"log"`).
        sr (:python:`int`, *optional*):
            The sample rate of the filtered noise (default: :python:`30000`).
        zerophase (:python:`bool`, *optional*):
            If set to :python:`True`, we use a zero-phase crossover filter (default: :python:`True`).
        order (:python:`int`, *optional*):
            The order of the crossover filter (default: :python:`2`).
        noise_randomness (:python:`str`, *optional*):
            The randomness of the filtered noise, either :python:`"pseudo-random"`, :python:`"fixed"`, or :python:`"random"` (default: :python:`"pseudo-random"`).
        use_fade_in (:python:`bool`, *optional*):
            If set to :python:`True`, we use a fade-in envelope (default: :python:`False`).
        min_decay_ms (:python:`float`, *optional*):
            The minimum decay time in milliseconds (default: :python:`50`).
        max_decay_ms (:python:`float`, *optional*):
            The maximum decay time in milliseconds (default: :python:`2000`).
        flashfftconv (:python:`bool`, *optional*):
            An option to use :python:`FlashFFTConv` :cite:`fu2023flashfftconv` as a backend to perform the causal convolution efficiently (default: :python:`True`).
        max_input_len (:python:`int`, *optional*):
            When :python:`flashfftconv` is set to :python:`True`, the max input length must be also given (default: :python:`2**17`).
    """

    def __init__(
        self,
        ir_len=60000,
        num_bands=12,
        processor_channel="midside",
        f_min=31.5,
        f_max=15000,
        scale="log",
        sr=30000,
        zerophase=True,
        order=2,
        noise_randomness="pseudo-random",
        use_fade_in=False,
        min_decay_ms=50,
        max_decay_ms=2000,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.processor_channel = processor_channel

        match self.processor_channel:
            case "midside":
                self.num_channels = 2
                self.process = self._process_midside
            case "stereo":
                self.num_channels = 2
                self.process = self._process_mono_stereo
            case "mono":
                self.num_channels = 1
                self.process = self._process_mono_stereo
            case _:
                raise ValueError(f"Unknown channel type: {self.channel}")

        self.ir_len = ir_len
        self.noise_randomness = noise_randomness
        match self.noise_randomness:
            case "pseudo-random" | "fixed":
                noise_len = (
                    self.ir_len if self.noise_randomness == "fixed" else self.ir_len * 5
                )
                filtered_noise = get_filtered_noise(
                    noise_len,
                    num_channels=self.num_channels,
                    num_bands=self.num_bands,
                    f_min=f_min,
                    f_max=f_max,
                    scale=scale,
                    sr=sr,
                    zerophase=zerophase,
                    order=order,
                )
                filtered_noise = filtered_noise.unsqueeze(0)
                self.register_buffer("filtered_noise", filtered_noise)
            case "random":
                assert False  # TODO
            case _:
                raise ValueError(
                    f"Invalid filtered_noise argument: {self.filtered_noise}"
                )

        self.conv = FIRConvolution(
            mode="causal",
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

        min_decay_sample = min_decay_ms * sr / 1000
        min_decay_db = -60 / min_decay_sample
        self.min_decay = min_decay_db / 20 * math.log(10)

        max_decay_sample = max_decay_ms * sr / 1000
        max_decay_db = -60 / max_decay_sample
        self.max_decay = max_decay_db / 20 * math.log(10)

        self.use_fade_in = use_fade_in

        arange = torch.arange(self.ir_len)[None, None, None, :]
        self.register_buffer("arange", arange)

    def forward(
        self, input_signals, log_decay, log_gain, log_fade_in=None, z_fade_in_gain=None
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B\times C\times L`):
                A batch of input audio signals.
            log_decay (:python:`FloatTensor`, :math:`B\times C_{\mathrm{rev}}\times K \:\!`):
                A batch of log-decay values.
            log_gain (:python:`FloatTensor`, :math:`B\times C_{\mathrm{rev}}\times K \:\!`):
                A batch of log-gain values.
            log_fade_in (:python:`FloatTensor`, :math:`B\times C_{\mathrm{rev}}\times K`, *optional*):
                A batch of log-fade-in values (default: :python:`None`).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.

        """

        log_decay = (
            torch.sigmoid(log_decay) * (self.max_decay - self.min_decay)
            + self.min_decay
        )
        envelope = torch.exp(self.arange * log_decay.unsqueeze(-1))

        if self.use_fade_in:
            log_fade_in = (
                torch.sigmoid(log_fade_in) * (log_decay - self.min_decay)
                + self.min_decay
            )
            fade_in = torch.exp(self.arange * log_fade_in.unsqueeze(-1))
            fade_in_gain = torch.sigmoid(z_fade_in_gain).unsqueeze(-1)

            envelope = envelope - fade_in * fade_in_gain

        envelope = envelope * log_gain.unsqueeze(-1)

        filtered_noise = self.get_filtered_noise()
        ir = filtered_noise * envelope
        ir = ir.sum(2)

        output_signals = self.process(input_signals, ir)
        return output_signals

    def get_filtered_noise(self):
        match self.noise_randomness:
            case "pseudo-random":
                start = torch.randint(
                    0, self.filtered_noise.shape[-1] - self.ir_len, (1,)
                )
                return self.filtered_noise[..., start : start + self.ir_len]
            case "fixed":
                return self.filtered_noise
            case "random":
                assert False

    def _process_mono_stereo(self, input_signals, fir):
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)

    def _process_midside(self, input_signals, fir):
        fir = normalize_impulse(fir)
        input_signals = lr_to_ms(input_signals)
        output_signals = self.conv(input_signals, fir)
        return ms_to_lr(output_signals)

    def _process_pseudo_midside(self, input_signals, fir):
        fir = ms_to_lr(fir)
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        shape = (self.num_channels, self.num_bands)
        size = {"log_decay": shape, "log_gain": shape}
        if self.use_fade_in:
            size["log_fade_in"] = shape
            size["z_fade_in_gain"] = shape
        return size


# class FeedbackDelayNetwork(nn.Module):
#    r"""
#    A frequency-sampled feedback delay network (FDN).
#    """
#
#    def __init__(self):
#        super().__init__()
#
#    def forward(self):
#        pass
#
