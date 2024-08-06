import torch
import torch.nn as nn


class StereoGain(nn.Module):
    r"""
    A simple stereo-to-stereo or mono-to-stereo gain.

        We use simple channel-wise constant multiplication with a gain vector.
        The gain is assumed to be in log scale, so we apply exponentiation before multiplying it to the stereo signal.
        $$
        y[n] = \exp (g_{\mathrm{log}}) \cdot u[n].
        $$

        Hence, the learnable parameter is $p = \{ g_{\mathrm{log}} \}.$
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_signals, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals, either mono or stereo.
            log_magnitude (:python:`FloatTensor`, :math:`B \times 2`):
                A log-magnitude vector of the FIR filter.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times 2 \times L`.
        """
        b, c, t = input_signals.shape
        assert c == 2
        gain = torch.exp(log_gain)
        output_signals = input_signals * gain[..., None]
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"log_gain": 2}


class SideGainImager(nn.Module):
    r"""
    Stereo imager with side-channel loudness control.

        We multiply the input's side signal $u_{\mathrm{s}}[n]$, 
        i.e., left $u_{\mathrm{l}}[n]$ minus right $u_{\mathrm{r}}[n]$, 
        with a gain parameter $g \in \mathbb{R}$ 
        to control the stereo width. The mid and side outputs are given as
        $$
        y_{\mathrm{m}}[n] &= u_{\mathrm{l}}[n] + u_{\mathrm{r}}[n], \\
        y_{\mathrm{s}}[n] &= \exp (g) \cdot (u_{\mathrm{l}}[n] - u_{\mathrm{r}}[n]).
        $$

        Hence, the learnable parameter is $p = \{ g_{\mathrm{log}} \}.$
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_signals, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals; must be stereo.
            log_magnitude (:python:`FloatTensor`, :math:`B \times P \:\!`):
                A log-magnitude vector of the FIR filter.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times 2 \times L`.
        """
        b, c, t = input_signals.shape
        assert c == 2

        left, right = input_signals[:, 0, :], input_signals[:, 1, :]
        mid, side = left + right, left - right
        gain = torch.exp(log_gain)
        side = gain * side
        left, right = (mid + side) / 2, (mid - side) / 2
        output_signals = torch.stack([left, right], 1)
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"log_gain": 1}
