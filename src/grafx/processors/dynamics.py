import torch
import torch.nn as nn
import torch.nn.functional as F

from grafx.processors.core.convolution import CausalConvolution
from grafx.processors.core.envelope import Ballistics, TruncatedOnePoleIIRFilter


class ApproxCompressor(nn.Module):
    r"""
    A feed-forward dynamic range compressor :cite:`giannoulis2012digital`. 

        First, for a given input audio, we sum the left and right channels to obtain a mid $u_\mathrm{m}[n]$. 
        Then, we calculate its energy envelope $G_u[n]$ as follows,
        $$
        G_u[n] &= \log g_u[n], \\
        g_u[n] &= \alpha[n] g_u[n-1]+(1-\alpha[n]) u_{\mathrm{m}}^2[n].
        $$

        Here, the coefficient $\alpha[n]$ is typically set to a different constant for an "attack" 
        (where $g_u[n]$ increases) and "release" (where $g_u[n]$ decreases) phase.
        As this part (also known as ballistics) bottlenecks the computation speed in GPU,
        we follow the recent work :cite:`steinmetz2022style` and 
        restrict the coefficients to the same value $\alpha$. 
        By doing so, the above equation simplifies to a one-pole IIR filter, 
        whose impulse response $h^\mathrm{env}[n]$ can be approximated to a certain length $N$.
        $$
        h^\mathrm{env}[n] = (1-\alpha) \alpha^n.
        $$

        Then, the approximated loudness envelope is $G_u[n] \approx \log (u_{\mathrm{m}}^2 * h^\mathrm{env})[n]$.

        Next, we compute the output (compressed) envelope $G_y[n]$ as follows,
        $$
        G_y[n] = \begin{cases}
        G_y^\mathrm{above}[n] & G_u[n] \geq T+W,  \\
        G_y^\mathrm{mid}[n]   & T-W \leq G_u[n] < T+W, \\
        G_y^\mathrm{below}[n] & G_u[n] < T-W
        \end{cases}
        $$

        where $T$ and $W$ is a threshold and knee width (both in the log domain), respectively. 
        We use a quadratic knee, which gives us the following formula.
        $$
        G_y^\mathrm{above}[n] &= T+\frac{G_u[n]-T}{R}, \\
        G_y^\mathrm{mid}[n]   &= G_u[n] + \Big(\frac{1}{R}-1\Big)\frac{(G_u[n]-T+W)^2}{4W}, \\
        G_y^\mathrm{below}[n] &= G_u[n].
        $$

        Finally, we compute the gain reduction curve and multiply it to all channels.
        $$
        y_{\mathrm{x}}[n] = \exp(G_y[n]-G_u[n]) \cdot u_{\mathrm{x}}[n] \quad (\mathrm{x} \in \{\mathrm{l}, \mathrm{r}\}).
        $$

        This compressor's learnable parameter is $p = \{ z_{\alpha}, T, \bar{R}, W_{\mathrm{log}} \}$.
        The IIR filter coefficient is recovered with a logistic sigmoid $\alpha = \sigma (z_{\alpha})$.
        The ratio is recovered with $R = 1 + \exp (\bar{R})$. 
        Finally, the knee width is obtained with $W = \exp (W_{\mathrm{log}})$.

    Args:
        iir_len (:python:`int`, *optional*): 
            The legnth of the smoothing FIR (default: :python:`16384`).
        flashfftconv (:python:`bool`, *optional*): 
            An option to use :python:`FlashFFTConv` :cite:`fu2023flashfftconv` as a backend 
            to perform the causal convolution in the gain smoothing stage efficiently (default: :python:`True`).
        max_input_len (:python:`int`, *optional*): 
            When :python:`flashfftconv` is set to :python:`True`, 
            the max input length must be also given (default: :python:`2**17`).


    """

    def __init__(
        self,
        iir_len=16384,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        self.env_follower = IIREnvelopeFollower(
            iir_len=iir_len,
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

    # @profile
    def forward(self, input_signals, z_alpha, log_threshold, log_ratio, log_knee=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`N \times C \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`N \times 1`):
                IIR coefficients before applying the sigmoid.
            log_threshold (:python:`FloatTensor`, :math:`N \times 1`):
                Compression threshold in log scale.
            log_ratio (:python:`FloatTensor`, :math:`N \times 1`):
                Unconstrained ratio values, which will be transformed into the range of :math:`[1, \infty)`.
            log_knee (:python:`FloatTensor`, :math:`N \times 1`, *optional*):
                Log of knee that operates on the log scale.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        log_energy = self.env_follower(input_signals, z_alpha)
        gain = self.compute_gain_exp(log_energy, log_threshold - 6, log_ratio, log_knee)
        output_signals = gain * input_signals
        return output_signals

    def compute_gain(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)
        middle = log_energy + (1 / (ratio + 1e-3) - 1) * (
            log_energy - log_threshold + log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def compute_gain_exp(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee - 1)
        log_gain = (
            (1 / ratio - 1)
            * F.softplus(log_knee * (log_energy - log_threshold))
            / log_knee
        )
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"z_alpha": 1, "log_threshold": 1, "log_ratio": 1, "log_knee": 1}


class ApproxNoiseGate(nn.Module):
    r"""
    A feed-forward noisegate :cite:`giannoulis2012digital`. 

        It is identical to the above :python:`ApproxCompressor` except for the output gain computation.
        Instead of compressing the signal above the threshold, it compresses below the threshold.
        $$
        G_y^\mathrm{above}[n] &= G_u[n], \\
        G_y^\mathrm{mid}[n]   &= G_u[n] + (1-R)\frac{(G_u[n]-T-W)^2}{4W}, \\
        G_y^\mathrm{below}[n] &= T+R(G_u[n]-T).
        $$

        Again, this processor's learnable parameter is $p = \{ z_{\alpha}, T, \bar{R}, W_{\mathrm{log}} \}$.

    Args:
        iir_len (:python:`int`, *optional*): 
            The legnth of the smoothing FIR (default: :python:`16384`).
        flashfftconv (:python:`bool`, *optional*): 
            An option to use :python:`FlashFFTConv` :cite:`fu2023flashfftconv` as a backend 
            to perform the causal convolution in the gain smoothing stage efficiently (default: :python:`True`).
        max_input_len (:python:`int`, *optional*): 
            When :python:`flashfftconv` is set to :python:`True`, 
            the max input length must be also given (default: :python:`2**17`).
    """

    def __init__(
        self,
        freq_sample_n=16384,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        self.env_follower = IIREnvelopeFollower(
            iir_len=freq_sample_n,
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )

    def forward(self, input_signals, z_alpha, log_threshold, log_ratio, log_knee):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`N \times C \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`N \times 1`):
                IIR coefficients before applying the sigmoid.
            log_threshold (:python:`FloatTensor`, :math:`N \times 1`):
                Gating threshold in log scale.
            log_ratio (:python:`FloatTensor`, :math:`N \times 1`):
                Unconstrained ratio values, which will be transformed into the range of :math:`[1, \infty)`.
            log_knee (:python:`FloatTensor`, :math:`N \times 1`, *optional*):
                Log of knee that operates on the log scale.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        log_energy = self.env_follower(input_signals, z_alpha)
        gain = self.compute_gain(log_energy, log_threshold - 6, log_ratio, log_knee)
        output_signals = gain * input_signals
        return output_signals

    def compute_gain(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = ratio * (log_energy - log_threshold) + log_threshold
        above = log_energy
        middle = log_energy + (1 - ratio) * (
            log_energy - log_threshold - log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"z_alpha": 1, "log_threshold": 1, "log_ratio": 1, "log_knee": 1}


class BaseEnvelopeFollower(nn.Module):
    def __init__(self, smoother, detect_with="energy"):
        super().__init__()
        self.detect_with = detect_with
        self.smoother = smoother

    def forward(self, signal, *args, **kwargs):
        match self.detect_with:
            case "energy":
                loudness = signal.square().mean(-2)
            case "amplitude":
                loudness = signal.abs().mean(-2)
            case "rms_channel":
                loudness = (self.eps + signal.square().mean(-2)).sqrt()

        envelope = self.smoother(loudness, *args, **kwargs)
        envelope = torch.log(envelope + 1e-5)
        return envelope

    def parameter_size(self):
        return self.smoother.parameter_size()


class IIREnvelopeFollower(BaseEnvelopeFollower):
    def __init__(
        self,
        detect_with="energy",
        iir_len=16384,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        smoother = TruncatedOnePoleIIRFilter(
            iir_len=iir_len,
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )
        super().__init__(smoother=smoother, detect_with=detect_with)


class BallisticsEnvelopeFollower(BaseEnvelopeFollower):
    def __init__(self, detect_with="energy"):
        smoother = Ballistics()
        super().__init__(smoother=smoother, detect_with=detect_with)


class BaseCompressor(nn.Module):
    def __init__(
        self,
        env_follower,
        with_knee=True,
    ):
        super().__init__()
        self.with_knee = with_knee
        self.env_follower = env_follower

    def forward(
        self, input_signals, log_threshold, log_ratio, log_knee=None, **env_params
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`N \times C \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`N \times 1`):
                IIR coefficients before applying the sigmoid.
            log_threshold (:python:`FloatTensor`, :math:`N \times 1`):
                Compression threshold in log scale.
            log_ratio (:python:`FloatTensor`, :math:`N \times 1`):
                Unconstrained ratio values, which will be transformed into the range of :math:`[1, \infty)`.
            log_knee (:python:`FloatTensor`, :math:`N \times 1`, *optional*):
                Log of knee that operates on the log scale.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        log_energy = self.env_follower(input_signals, **env_params)
        if self.with_knee:
            gain = self.compute_gain_with_knee(
                log_energy, log_threshold - 6, log_ratio, log_knee
            )
        else:
            gain = self.compute_gain_without_knee(
                log_energy, log_threshold - 6, log_ratio
            )
        output_signals = gain * input_signals
        return output_signals

    def compute_gain_with_knee(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)
        middle = log_energy + (1 / (ratio + 1e-3) - 1) * (
            log_energy - log_threshold + log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def compute_gain_without_knee(self, log_energy, log_threshold, log_ratio):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < log_threshold
        above_mask = log_energy >= log_threshold

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)

        log_energy_out = below * below_mask + above * above_mask
        log_gain = log_energy_out - log_energy
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"log_threshold": 1, "log_ratio": 1}
        if self.with_knee:
            size["log_knee"] = 1
        size = {**size, **self.env_follower.parameter_size()}
        return size


class BaseCompressor2(nn.Module):
    def __init__(
        self,
        gain_smoother=None,
        smooth_in_log=False,
        with_knee=True,
        smooth_energy=True,
    ):
        super().__init__()
        self.gain_smoother = gain_smoother
        self.with_knee = with_knee
        self.smooth_in_log = smooth_in_log

        self.smooth_energy = smooth_energy
        if self.smooth_energy:
            self.energy_smoother = TruncatedOnePoleIIRFilter()

    def forward(
        self,
        input_signals,
        z_alpha_pre,
        log_threshold,
        log_ratio,
        log_knee=None,
        **gain_params,
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`N \times C \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`N \times 1`):
                IIR coefficients before applying the sigmoid.
            log_threshold (:python:`FloatTensor`, :math:`N \times 1`):
                Compression threshold in log scale.
            log_ratio (:python:`FloatTensor`, :math:`N \times 1`):
                Unconstrained ratio values, which will be transformed into the range of :math:`[1, \infty)`.
            log_knee (:python:`FloatTensor`, :math:`N \times 1`, *optional*):
                Log of knee that operates on the log scale.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        energy = input_signals.square().mean(-2)
        if self.smooth_energy:
            energy = self.energy_smoother(energy, z_alpha=z_alpha_pre)
        log_energy = torch.log(energy + 1e-5)

        if self.with_knee:
            gain = self.compute_gain_with_knee(
                log_energy, log_threshold - 6, log_ratio, log_knee
            )
        else:
            gain = self.compute_gain_without_knee(
                log_energy, log_threshold - 6, log_ratio
            )

        if self.gain_smoother is not None:
            if self.smooth_in_log:
                gain = self.gain_smoother(gain, **gain_params)
                gain = torch.exp(gain)
            else:
                gain = torch.exp(gain)
                gain = self.gain_smoother(gain, **gain_params)
        else:
            gain = torch.exp(gain)
        output_signals = gain[:, None, :] * input_signals
        return output_signals

    def compute_gain_with_knee(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)
        middle = log_energy + (1 / (ratio + 1e-3) - 1) * (
            log_energy - log_threshold + log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        return log_gain

    def compute_gain_without_knee(self, log_energy, log_threshold, log_ratio):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < log_threshold
        above_mask = log_energy >= log_threshold

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)

        log_energy_out = below * below_mask + above * above_mask
        log_gain = log_energy_out - log_energy
        return log_gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"log_threshold": 1, "log_ratio": 1, "z_alpha_pre": 1}
        if self.with_knee:
            size["log_knee"] = 1
        size = {**size, **self.gain_smoother.parameter_size()}
        return size


class OnePoleIIRCompressor(BaseCompressor2):
    def __init__(
        self,
        iir_len=16384,
        flashfftconv=True,
        max_input_len=2**17,
        with_knee=True,
    ):
        env_follower = TruncatedOnePoleIIRFilter(
            iir_len=iir_len,
            flashfftconv=flashfftconv,
            max_input_len=max_input_len,
        )
        super().__init__(gain_smoother=env_follower, with_knee=with_knee)


class BallisticsCompressor(BaseCompressor2):
    def __init__(self, with_knee=True):
        env_follower = Ballistics()
        super().__init__(
            gain_smoother=env_follower,
            with_knee=with_knee,
            smooth_in_log=False,
        )


class BaseNoiseGate(nn.Module):
    def __init__(
        self,
        env_follower,
        with_knee=True,
    ):
        super().__init__()
        self.with_knee = with_knee
        self.env_follower = env_follower

    def forward(
        self, input_signals, log_threshold, log_ratio, log_knee=None, **env_params
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`N \times C \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`N \times 1`):
                IIR coefficients before applying the sigmoid.
            log_threshold (:python:`FloatTensor`, :math:`N \times 1`):
                Compression threshold in log scale.
            log_ratio (:python:`FloatTensor`, :math:`N \times 1`):
                Unconstrained ratio values, which will be transformed into the range of :math:`[1, \infty)`.
            log_knee (:python:`FloatTensor`, :math:`N \times 1`, *optional*):
                Log of knee that operates on the log scale.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        log_energy = self.env_follower(input_signals, **env_params)
        if self.with_knee:
            gain = self.compute_gain_with_knee(
                log_energy, log_threshold - 6, log_ratio, log_knee
            )
        else:
            gain = self.compute_gain_without_knee(
                log_energy, log_threshold - 6, log_ratio
            )
        output_signals = gain * input_signals
        return output_signals

    def compute_gain_with_knee(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = ratio * (log_energy - log_threshold) + log_threshold
        above = log_energy
        middle = log_energy + (1 - ratio) * (
            log_energy - log_threshold - log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def compute_gain_without_knee(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < log_threshold
        above_mask = log_energy > log_threshold

        below = ratio * (log_energy - log_threshold) + log_threshold
        above = log_energy

        log_energy_out = below * below_mask + above * above_mask
        log_gain = log_energy_out - log_energy
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"log_threshold": 1, "log_ratio": 1}
        if self.with_knee:
            size["log_knee"] = 1
        size = {**size, **self.env_follower.parameter_size()}
        return size


class Compressor(nn.Module):
    def __init__(
        self,
        energy_smoother="iir",
        gain_smoother=None,
        gain_smooth_in_log=False,
        with_knee=True,
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()
        assert energy_smoother in ["iir", "ballistics", None]
        assert gain_smoother in ["iir", "ballistics", None]
        assert (energy_smoother is not None) or (gain_smoother is not None)

        self.energy_smoother = energy_smoother
        match self.energy_smoother:
            case "iir":
                self.energy_smoother_module = TruncatedOnePoleIIRFilter(
                    flashfftconv=flashfftconv, max_input_len=max_input_len
                )
            case "ballistics":
                self.energy_smoother_module = Ballistics()
            case None:
                self.energy_smoother_module = nn.Identity()

        self.gain_smoother = gain_smoother
        match self.gain_smoother:
            case "iir":
                self.gain_smoother_module = TruncatedOnePoleIIRFilter(
                    flashfftconv=flashfftconv, max_input_len=max_input_len
                )
            case "ballistics":
                self.gain_smoother_module = Ballistics()
            case None:
                self.gain_smoother_module = nn.Identity()

        self.with_knee = with_knee
        self.gain_smooth_in_log = gain_smooth_in_log

    # @profile
    def forward(
        self, input_signals, log_threshold, log_ratio, log_knee=None, **smoother_params
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`N \times C \times L`):
                A batch of input audio signals.
            z_alpha (:python:`FloatTensor`, :math:`N \times 1`):
                IIR coefficients before applying the sigmoid.
            log_threshold (:python:`FloatTensor`, :math:`N \times 1`):
                Compression threshold in log scale.
            log_ratio (:python:`FloatTensor`, :math:`N \times 1`):
                Unconstrained ratio values, which will be transformed into the range of :math:`[1, \infty)`.
            log_knee (:python:`FloatTensor`, :math:`N \times 1`, *optional*):
                Log of knee that operates on the log scale.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        energy_params, gain_params = self.get_smoother_params(smoother_params)

        energy = input_signals.square().mean(-2)
        energy = self.energy_smoother_module(energy, **energy_params)
        log_energy = torch.log(energy + 1e-5)

        if self.with_knee:
            gain = self.compute_gain_with_knee(
                log_energy, log_threshold - 6, log_ratio, log_knee
            )
        else:
            gain = self.compute_gain_without_knee(
                log_energy, log_threshold - 6, log_ratio
            )

        if self.gain_smooth_in_log:
            gain = self.gain_smoother_module(gain, **gain_params)
            gain = torch.exp(gain)
        else:
            gain = torch.exp(gain)
            gain = self.gain_smoother_module(gain, **gain_params)

        output_signals = gain[:, None, :] * input_signals
        return output_signals

    def get_smoother_params(self, smoother_params):
        energy_params, gain_params = {}, {}
        for k, v in smoother_params.items():
            k1, k2 = k.split("_", 1)
            match k1:
                case "energy":
                    energy_params[k2] = v
                case "gain":
                    gain_params[k2] = v
        return energy_params, gain_params

    def compute_gain_with_knee(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)
        middle = log_energy + (1 / (ratio + 1e-3) - 1) * (
            log_energy - log_threshold + log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        return log_gain

    def compute_gain_without_knee(self, log_energy, log_threshold, log_ratio):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < log_threshold
        above_mask = log_energy >= log_threshold

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)

        log_energy_out = below * below_mask + above * above_mask
        log_gain = log_energy_out - log_energy
        return log_gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"log_threshold": 1, "log_ratio": 1}
        if self.with_knee:
            size["log_knee"] = 1
        if self.energy_smoother is not None:
            energy_size = self.energy_smoother_module.parameter_size()
            energy_size = {"energy_" + k: v for k, v in energy_size.items()}
            size = {**size, **energy_size}
        if self.gain_smoother is not None:
            gain_size = self.gain_smoother_module.parameter_size()
            gain_size = {"gain_" + k: v for k, v in gain_size.items()}
            size = {**size, **gain_size}
        return size


class FactorizedCompressor(nn.Module):
    def __init__(
        self,
        gain_smooth_in_log=False,
        with_knee=True,
        frame_len=1024,
    ):
        super().__init__()

        self.energy_smoother_module = Ballistics()
        self.gain_smooth_in_log = gain_smooth_in_log
        self.with_knee = with_knee
        self.frame_len = frame_len
        self.stride = frame_len // 2
        window = torch.hann_window(frame_len)
        self.register_buffer("window", window)

    # @profile
    def forward(
        self,
        input_signals,
        parameters,
    ):

        if self.with_knee:
            f_log_threshold, f_log_ratio, f_log_knee, f_z = parameters[:5].split(
                [1, 1, 1, 2], dim=-1
            )
            s_log_threshold, s_log_ratio, s_log_knee, s_z = parameters[5:].split(
                [1, 1, 1, 2], dim=-1
            )
        else:
            f_log_threshold, f_log_ratio, f_z = parameters[:4].split([1, 1, 2], dim=-1)
            s_log_threshold, s_log_ratio, s_z = parameters[4:].split([1, 1, 2], dim=-1)

        energy = input_signals.square().mean(-2, keepdim=True)
        energy_unfold = F.unfold(energy, kernel_size=self.frame_len, stride=self.stride)
        energy_unfold = energy_unfold.transpose(1, 2)  # (batch, frame, sample)

        b, f, s = energy_unfold.shape
        energy_unfold = energy_unfold * self.window[None, None, :]
        f_energy = energy_unfold.mean(-1)
        s_energy = energy_unfold.view(-1, s)

        # energy = self.energy_smoother_module(energy, **energy_params)
        log_energy = torch.log(energy + 1e-5)

        if self.with_knee:
            gain = self.compute_gain_with_knee(
                log_energy, log_threshold - 6, log_ratio, log_knee
            )
        else:
            gain = self.compute_gain_without_knee(
                log_energy, log_threshold - 6, log_ratio
            )

        if self.gain_smooth_in_log:
            gain = self.gain_smoother_module(gain, **gain_params)
            gain = torch.exp(gain)
        else:
            gain = torch.exp(gain)
            gain = self.gain_smoother_module(gain, **gain_params)

        output_signals = gain[:, None, :] * input_signals
        return output_signals

    def compute_gain_with_knee(self, log_energy, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < (log_threshold - log_knee / 2)
        above_mask = log_energy > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)
        middle = log_energy + (1 / (ratio + 1e-3) - 1) * (
            log_energy - log_threshold + log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_energy_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_energy_out - log_energy
        return log_gain

    def compute_gain_without_knee(self, log_energy, log_threshold, log_ratio):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_energy < log_threshold
        above_mask = log_energy >= log_threshold

        below = log_energy
        above = log_threshold + (log_energy - log_threshold) / (ratio + 1e-3)

        log_energy_out = below * below_mask + above * above_mask
        log_gain = log_energy_out - log_energy
        return log_gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        if self.with_knee:
            size = 5
        else:
            size = 10
        return {"parameters": size}
