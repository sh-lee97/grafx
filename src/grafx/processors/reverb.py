import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from grafx.processors.components import CausalConvolution
from grafx.processors.functional import normalize_impulse


class MidSideFilteredNoiseReverb(nn.Module):
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

        self.conv = CausalConvolution(
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

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
        noise = torch.FloatTensor(num_noises * 2, self.ir_len, device=device).uniform_(
            -1, 1
        )
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
        output_signals = self.conv(input_signals, ir)
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

        ir = self.ms_to_lr(ir)
        ir = normalize_impulse(ir)
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

    def ms_to_lr(self, ir):
        mid, side = torch.split(ir, (1, 1), -2)
        left, right = mid + side, mid - side
        ir = torch.cat([left, right], -2)
        return ir

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
