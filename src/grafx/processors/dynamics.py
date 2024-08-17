import torch
import torch.nn as nn
import torch.nn.functional as F

from grafx.processors.core.envelope import Ballistics, TruncatedOnePoleIIRFilter
from grafx.processors.core.nonlinear import (
    compressor_gain_exp_knee,
    compressor_gain_hard_knee,
    compressor_gain_quad_knee,
)


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
        log_gain = compressor_gain_quad_knee(
            log_energy, log_threshold - 6, log_ratio, log_knee
        )
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        output_signals = gain * input_signals
        return output_signals

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


class Compressor(nn.Module):
    r"""
    A feed-forward dynamic range compressor :cite:`giannoulis2012digital`. 

        **(Pre-smoothing)** 
        First, for a given input audio, we calculate the mean energy across all channels as $e[n]$.
        Then, we optionally calculate its log-energy envelope $G_u[n] = \log g_u[n]$ where 
        the envelope can be either computed with a one-pole IIR filter or a ballistics model.
        $$
        g_u[n] = \alpha[n] g_u[n-1]+(1-\alpha[n]) e[n].
        $$

        If we use the ballistics model, the coefficient $\alpha[n]$ is set to a different constant for an "attack" 
        (where $g_u[n]$ increases) and "release" (where $g_u[n]$ decreases) phase.
        For such a case, we use an optimized backend :python:`diffcomp` :cite:`yu2024differentiable`.
        As this part is generally slow in GPU, we can follow the recent work :cite:`steinmetz2022style` and
        restrict the coefficients to the same value $\alpha$. 
        This gives a one-pole IIR filter, whose impulse response $h^\mathrm{env}[n] = (1-\alpha) \alpha^n$ 
        can be computed up to a length $N$ then used to obtain the approximate envelope $G_u[n] \approx \log (e * h^\mathrm{env})[n]$.
        Or, we can omit this part and simply use $G_u[n] = \log e[n]$.

        **(Gain computation)** 
        Next, we compute the output (compressed) envelope $G_y[n]$.
        We provide three options for the knee shape: hard, quadratic, and exponential.
        First, the simplest hard knee gives the following output envelope,
        $$
        G_y[n] = \begin{cases}
        G_y^\mathrm{above}[n] & G_u[n] \geq T,  \\
        G_y^\mathrm{below}[n] & G_u[n] < T
        \end{cases}
        $$
        where $T$ is a threshold in the log domain, and each condition is computed as

        $$
        G_y^\mathrm{above}[n] &= T+\frac{G_u[n]-T}{R}, \\
        G_y^\mathrm{below}[n] &= G_u[n].
        $$

        If we use the quadratic knee, 
        the above and below conditions become $G_u[n] \geq T+W$ and $G_u[n] < T-W$, respectively, 
        and in the middle, i.e., $T-W \leq G_u[n] < T+W$, the output envelope is computed as
        $$
        G_y^\mathrm{mid}[n]   &= G_u[n] + \Big(\frac{1}{R}-1\Big)\frac{(G_u[n]-T+W)^2}{4W}, \\
        $$
        where $W$ is a knee width in the log domain.
        If we use the exponential knee,
        there is no conditional branch, and the output envelope is given as
        $$
        G_y[n] = G_u[n] + \Big(\frac{1}{R} - 1\Big) \frac{\log (1 + \exp(W \cdot (G_u[n] - T))}{W}.
        $$
        
        **(Post-smoothing)** 
        Finally, we compute the gain reduction curve 
        $$
        g[n] = \exp(G_y[n] - G_u[n]).
        $$

        Before multiplying it to all channels, we can optionally smooth it (like the energy smoothing) 
        with a one-pole IIR or ballistics filter.
        This compressor's learnable parameter is 
        $p = \{ z_{\alpha}^{\mathrm{pre}}, z_{\alpha}^{\mathrm{post}}, T, \bar{R}, W_{\mathrm{log}} \}$.
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
        energy_smoother="iir",
        gain_smoother=None,
        gain_smooth_in_log=False,
        knee="quadratic",
        flashfftconv=True,
        max_input_len=2**17,
    ):
        super().__init__()

        self.energy_smoother = energy_smoother
        match self.energy_smoother:
            case "iir":
                self.energy_smoother_module = TruncatedOnePoleIIRFilter(
                    flashfftconv=flashfftconv, max_input_len=max_input_len
                )
            case "ballistics":
                self.energy_smoother_module = Ballistics()
            case None:
                pass
            case _:
                raise ValueError(f"Unknown energy_smoother: {self.energy_smoother}")

        self.gain_smoother = gain_smoother
        match self.gain_smoother:
            case "iir":
                self.gain_smoother_module = TruncatedOnePoleIIRFilter(
                    flashfftconv=flashfftconv, max_input_len=max_input_len
                )
            case "ballistics":
                self.gain_smoother_module = Ballistics()
            case None:
                pass
            case _:
                raise ValueError(f"Unknown gain_smoother: {self.gain_smoother}")

        self.knee = knee
        match self.knee:
            case "hard":
                self.compute_gain = compressor_gain_hard_knee
            case "quadratic":
                self.compute_gain = compressor_gain_quad_knee
            case "exponential":
                self.compute_gain = compressor_gain_exp_knee
            case _:
                raise ValueError(f"Unknown knee: {self.knee}")

        if gain_smooth_in_log:
            self.smooth = self.smooth_in_log
        else:
            self.smooth = self.smooth_in_linear

    # @profile
    def forward(
        self,
        input_signals,
        log_threshold,
        log_ratio,
        log_knee=None,
        z_alpha_pre=None,
        z_alpha_post=None,
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

        if self.energy_smoother is not None:
            energy = self.energy_smoother_module(energy, z_alpha=z_alpha_pre)

        log_energy = torch.log(energy + 1e-5) / 2
        gain = self.compute_gain(
            log_energy,
            log_threshold=log_threshold,
            log_ratio=log_ratio,
            log_knee=log_knee,
        )

        if self.gain_smoother is not None:
            gain = self.smooth(gain, z_alpha=z_alpha_post)

        output_signals = gain[:, None, :] * input_signals
        return output_signals

    def smooth_in_log(self, gain, **gain_smooth_params):
        gain = self.gain_smoother_module(gain, **gain_smooth_params)
        gain = torch.exp(gain)
        return gain

    def smooth_in_linear(self, gain, **gain_smooth_params):
        gain = torch.exp(gain)
        gain = self.gain_smoother_module(gain, **gain_smooth_params)
        return gain

    # def parameter_size(self):
    #    r"""
    #    Returns:
    #        :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
    #    """
    #    size = {}
    #    size["gain_params"] = {"log_threshold": 1, "log_ratio": 1}
    #    if self.knee != "hard":
    #        size["gain_params"]["log_knee"] = 1
    #    if self.energy_smoother is not None:
    #        size["energy_smoother_params"] = (
    #            self.energy_smoother_module.parameter_size()
    #        )
    #    if self.gain_smoother is not None:
    #        size["gain_smoother_params"] = self.gain_smoother_module.parameter_size()
    #    return size

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"log_threshold": 1, "log_ratio": 1}
        if self.knee != "hard":
            size["log_knee"] = 1
        if self.energy_smoother is not None:
            pre_size = self.energy_smoother_module.parameter_size()
            size["z_alpha_pre"] = pre_size["z_alpha"]
        if self.gain_smoother is not None:
            post_size = self.gain_smoother_module.parameter_size()
            size["z_alpha_post"] = post_size["z_alpha"]
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


def test_gains():
    from tqdm import tqdm

    B = 16
    T = 2**17
    log_energy = -50 * torch.rand(B, T).cuda()
    log_threshold = -50 * torch.rand(B, 1).cuda()
    log_ratio = torch.randn(B, 1).cuda()
    log_knee = 10 * torch.rand(B, 1).cuda()

    for _ in tqdm(range(1000)):
        compressor_gain(log_energy, log_threshold, log_ratio)

    # for _ in tqdm(range(1000)):
    #    compressor_gain_mask(log_energy, log_threshold, log_ratio)

    for _ in tqdm(range(1000)):
        compressor_gain_minimum(log_energy, log_threshold, log_ratio)

    for _ in tqdm(range(1000)):
        compressor_gain_quad_knee_original(
            log_energy, log_threshold, log_ratio, log_knee
        )

    for _ in tqdm(range(1000)):
        compressor_gain_quad_knee(log_energy, log_threshold, log_ratio, log_knee)

    # for _ in tqdm(range(1000)):
    #    compressor_gain_quad_knee_with_mask(
    #        log_energy, log_threshold, log_ratio, log_knee
    #    )

    for _ in tqdm(range(1000)):
        compressor_gain_exp_knee(log_energy, log_threshold, log_ratio, log_knee)


if __name__ == "__main__":
    test_gains()
    test_gains()
