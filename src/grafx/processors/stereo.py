import math

import torch
import torch.nn as nn

INV_SQRT_2 = 1 / math.sqrt(2)


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


class MonoToStereo(nn.Module):
    r"""
    A simple mono-to-stereo conversion.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_signals):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times 1 \times L`):
                A batch of input audio signals; must be mono.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times 2 \times L`.
        """
        b, c, t = input_signals.shape
        assert c == 1
        output_signals = input_signals.repeat(1, 2, 1)
        return output_signals

    def parameter_size(self):
        """
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {}


class StereoToMidSide(nn.Module):
    r"""
    A simple stereo-to-mid-side conversion.
    """

    def __init__(self, normalize=True):
        super().__init__()

        self.normalize = normalize

    def forward(self, input_signals):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times 2 \times L`):
                A batch of input audio signals; must be stereo.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times 2 \times L`.
        """
        _, c, _ = input_signals.shape
        assert c == 2
        if self.normalize:
            input_signals = input_signals * INV_SQRT_2
        left, right = input_signals[:, :1, :], input_signals[:, 1:, :]
        mid, side = left + right, left - right
        return mid, side

    def parameter_size(self):
        """
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {}


class MidSideToStereo(nn.Module):
    r"""
    A simple mid-side-to-stereo conversion.
    """

    def __init__(self, normalize=True):
        super().__init__()
        self.normalization_const = INV_SQRT_2 if normalize else 0.5

    def forward(self, mid, side):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            mid (:python:`FloatTensor`, :math:`B \times 1 \times L`):
                A batch of mid audio signals.
            side (:python:`FloatTensor`, :math:`B \times 1 \times L`):
                A batch of side audio signals.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times 2 \times L`.
        """
        b, c, t = mid.shape
        assert c == 1
        left, right = mid + side, mid - side
        output_signals = torch.cat([left, right], 1)
        output_signals = output_signals * self.normalization_const
        return output_signals

    def parameter_size(self):
        """
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {}
