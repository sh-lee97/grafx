import torch
import torch.nn as nn

from grafx.processors.components import IIREnvelopeFollower


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

        log_env = self.env_follower(input_signals, z_alpha)
        gain = self.compute_gain(log_env, log_threshold - 6, log_ratio, log_knee)
        output_signals = gain * input_signals
        return output_signals

    def compute_gain(self, log_env, log_threshold, log_ratio, log_knee):
        ratio = 1 + torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_env < (log_threshold - log_knee / 2)
        above_mask = log_env > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = log_env
        above = log_threshold + (log_env - log_threshold) / (ratio + 1e-3)
        middle = log_env + (1 / (ratio + 1e-3) - 1) * (
            log_env - log_threshold + log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_env_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_env_out - log_env
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
        log_env = self.env_follower(input_signals, z_alpha)
        gain = self.compute_gain(log_env, log_threshold - 6, log_ratio, log_knee)
        output_signals = gain * input_signals
        return output_signals

    def compute_gain(self, log_env, log_threshold, log_ratio, log_knee):
        ratio = torch.exp(log_ratio)
        log_knee = torch.exp(log_knee)

        below_mask = log_env < (log_threshold - log_knee / 2)
        above_mask = log_env > (log_threshold + log_knee / 2)
        middle_mask = (~below_mask) * (~above_mask)

        below = ratio * (log_env - log_threshold) + log_threshold
        above = log_env
        middle = log_env + (1 - ratio) * (
            log_env - log_threshold - log_knee / 2
        ) ** 2 / 2 / (log_knee + 1e-3)

        log_env_out = below * below_mask + above * above_mask + middle * middle_mask
        log_gain = log_env_out - log_env
        gain = torch.exp(log_gain)
        gain = gain[:, None, :]
        return gain

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"z_alpha": 1, "log_threshold": 1, "log_ratio": 1, "log_knee": 1}
