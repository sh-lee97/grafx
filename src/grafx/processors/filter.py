import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from grafx.processors.core.convolution import FIRConvolution
from grafx.processors.core.iir import IIRFilter
from grafx.processors.core.midside import lr_to_ms, ms_to_lr
from grafx.processors.core.utils import normalize_impulse

PI = math.pi
TWO_PI = 2 * math.pi
HALF_PI = math.pi / 2
TWOR_SCALE = 1 / math.log(2)
ALPHA_SCALE = 1 / 2


class FIRFilter(nn.Module):
    r"""
    A time-domain filter with learnable finite impulse response (FIR) coefficients.
    It is implemented with the :class:`~grafx.processors.core.convolution.FIRConvolution` module.

    Args:
        fir_len (:python:`int`, *optional*):
            The length of the FIR filter (default: :python:`1023`).
        processor_channel (:python:`str`, *optional*):
            The channel type of the processor, which can be either :python:`"mono"`, :python:`"stereo"`, or :python:`"midside"` (default: :python:`"mono"`).
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.FIRConvolution`.
    """

    def __init__(self, fir_len=1023, processor_channel="mono", **backend_kwargs):
        super().__init__()
        self.fir_len = fir_len
        self.conv = FIRConvolution(fir_len=fir_len, **backend_kwargs)

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

    def forward(self, input_signals, fir):
        r"""
        Performs the convolution operation.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C_\mathrm{in} \times L_\mathrm{in}`):
                A batch of input audio signals.
            fir (:python:`FloatTensor`, :math:`B \times C_\mathrm{filter} \times L_\mathrm{filter}`):
                A batch of FIR filters.

        Returns:
            :python:`FloatTensor`: A batch of convolved signals of shape :math:`B \times C_\mathrm{out} \times L_\mathrm{in}` where :math:`C_\mathrm{out} = \max (C_\mathrm{in}, C_\mathrm{filter})`.
        """
        fir = torch.tanh(fir)
        output_signals = self.process(input_signals, fir)
        return output_signals

    def _process_mono_stereo(self, input_signals, fir):
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)

    def _process_midside(self, input_signals, fir):
        fir = normalize_impulse(fir)
        input_signals = lr_to_ms(input_signals)
        output_signals = self.conv(input_signals, fir)
        return ms_to_lr(output_signals)

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"fir": (self.num_channels, self.fir_len)}


class BiquadFilter(nn.Module):
    r"""
    A serial stack of second-order filters (biquads) with the given coefficients.

        To ensure the stability of each biquad, we restrict the normalized feedback coefficients with the following activations :cite:`nercessian2021lightweight`.
        $$
        \bar{a}_{i, 1} &= 2 \tanh(\tilde{a}_{i, 1}), \\
        \bar{a}_{i, 2} &= \frac{(2 - \left|\bar{a}_{i, 1}\right|) \tanh(\tilde{a}_{i, 2}) + \left|\bar{a}_{i, 1}\right|}{2}.
        $$
        Here, $\tilde{a}_{i, 1}$ and $\tilde{a}_{i, 2}$ are the pre-activation values.
        The unnormlized coefficients can be optionally recovered by multiplying the normalized ones with $a_{i, 0}$.
        The learnable parameters are $p = \{\mathbf{B}, \tilde{\mathbf{a}}_1, \tilde{\mathbf{a}}_2, \mathbf{a}_0 \}$, 
        where $\mathbf{B} = [\mathbf{b}_0, \mathbf{b}_1, \mathbf{b}_2] \in \mathbb{R}^{K\times 3}$ 
        is the stacked biquad coefficients for the feedforward path
        and the latter three are the pre-activation values for the feedback path.
        The last one $\mathbf{a}_0$ is optional and only used when :python:`normalized == False`.

    Args:
        num_filters (:python:`int`, *optional*):
            Number of biquads to use (default: :python:`1`).
        normalized (:python:`bool`, *optional*):
            If set to :python:`True`, the filter coefficients are assumed to be normalized by $a_{i, 0}$,
            making the number of learnable parameters $5$ per biquad instead of $6$
            (default: :python:`False`).
        backend (:python:`str`, *optional*):
            The backend to use for the filtering, which can either be the frequency-sampling method
            :python:`"fsm"` or exact time-domain filter :python:`"lfilter"` (default: :python:`"fsm"`).
        fsm_fir_len (:python:`int`, *optional*):
            The length of FIR approximation when :python:`backend == "fsm"` (default: :python:`8192`).
    """

    def __init__(self, num_filters=1, normalized=False, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.normalized = normalized
        self.biquad = IIRFilter(order=2, **backend_kwargs)

    def forward(self, input_signals, Bs, A1_pre, A2_pre, A0=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            Bs (:python:`FloatTensor`, :math:`B \times K \times 3`):
                A batch of biquad coefficients, $b_{i, 0}, b_{i, 1}, b_{i, 2}$, stacked in the last dimension.
            A1_pre (:python:`FloatTensor`, :math:`B \times K\:\!`):
                A batch of pre-activation coefficients $\tilde{a}_{i, 1}$.
            A2_pre (:python:`FloatTensor`, :math:`B \times K\:\!`):
                A batch of pre-activation coefficients $\tilde{a}_{i, 2}$.
            A0 (:python:`FloatTensor`, :math:`B \times K`, *optional*):
                A batch of $a_{i, 0}$ coefficients, only used when :python:`normalized == False` (default: :python:`None`).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        A1_act = 2 * torch.tanh(A1_pre)
        A1_act_abs = A1_act.abs()
        A2_act = ((2 - A1_act_abs) * torch.tanh(A2_pre) + A1_act_abs) / 2
        ones = torch.ones_like(A1_pre)
        As = torch.stack([ones, A1_act, A2_act], dim=-1)

        if self.normalized:
            As = As * A0.unsqueeze(-1)
        B0 = Bs[:, :, :1]
        Bs = torch.cat([B0 + torch.ones_like(B0), Bs[:, :, 1:]], -1)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signals, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"Bs": (self.num_filters, 3)}
        size["A1_pre"] = self.num_filters
        size["A2_pre"] = self.num_filters
        if self.normalized:
            size["A0"] = self.num_filters
        return size


class PoleZeroFilter(nn.Module):
    r"""
    A serial stack of biquads with pole/zero parameters.

        $$
        H(z) = g \prod_{k=1}^K \frac{(z-q_{k})(z-q_{k}^*)}{(z-p_{k})(z-p_{k}^*)}
        $$

        The poles are restricted to the unit circle and reparameterized as follows,
        $$
        p_k = \tilde{p}_k \cdot \frac{\tanh( | \tilde{p}_k | )}{ | \tilde{p}_k | + \epsilon}.
        $$

        $p = \{ g, \tilde{\mathbf{p}}, \mathbf{z} \}$ are the learnable parameters,
        where both complex poles and zeros are repesented as a real-valued tensors with last dimension of size 2.

    Args:
        num_filters (:python:`int`, *optional*):
            Number of biquads to use (default: :python:`1`).
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters  ###
        self.biquad = IIRFilter(order=2, **backend_kwargs)  ###

    def forward(self, input_signals, log_gain, poles, zeros):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_gain (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of log-gains.
            poles (:python:`FloatTensor`, :math:`B \times K \times 2`):
                A batch of complex poles.
            zeros (:python:`FloatTensor`, :math:`B \times K \times 2`):
                A batch of complex zeros.


        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        gain = torch.exp(log_gain)

        poles = torch.view_as_complex(poles)
        poles_radii = torch.abs(poles)
        poles = poles * torch.tanh(poles_radii) / (poles_radii + 1e-5)

        zeros = torch.view_as_complex(zeros)
        zeros_radii = torch.abs(zeros)

        ones = torch.ones_like(poles_radii)

        b0 = ones
        b1 = -2 * zeros.real
        b2 = zeros_radii.square()

        a0 = ones
        a1 = -2 * poles.real
        a2 = poles_radii.square()

        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)

        output_signal = self.biquad(input_signals, Bs, As)
        output_signal = gain.unsqueeze(-1) * output_signal

        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """

        return {
            "log_gain": 1,
            "poles": (self.num_filters, 2),
            "zeros": (self.num_filters, 2),
        }


class StateVariableFilter(nn.Module):
    r"""
    A series of biquads with the state variable filter (SVF) parameters :cite:`vafilter, kuznetsov2020differentiable`.

        The biquad coefficients reparameterized and computed from the SVF parameters $R_i, G_i, c^{\mathrm{HP}}_i, c^{\mathrm{BP}}_i, c^{\mathrm{LP}}_i$ as follows,
        $$
        b_{i, 0} &= G_i^2 c^{\mathrm{LP}}_i+G_i c^{\mathrm{BP}}_i+c^{\mathrm{HP}}_i, \\
        b_{i, 1} &= 2G_i^2 c^{\mathrm{LP}}_i - 2c^{\mathrm{HP}}_i, \\
        b_{i, 2} &= G_i^2 c^{\mathrm{LP}}_i-G_i c^{\mathrm{BP}}_i+c^{\mathrm{HP}}_i, \\
        a_{i, 0} &= G_i^2 + 2R_iG_i + 1, \\
        a_{i, 1} &= 2G_i^2-2, \\
        a_{i, 2} &= G_i^2 - 2R_iG_i + 1.
        $$

        Note that we are not using the exact time-domain implementation of the SVF, 
        but rather its parameterization that allows better optimization than the direct prediction of biquad coefficients 
        (empirically observed in :cite:`kuznetsov2020differentiable, nercessian2021lightweight, lee2022differentiable`).
        To ensure the stability of each biquad, we ensure that $G_i$ and $R_i$ are positive.


    Args:
        num_filters (:python:`int`, *optional*):
            Number of SVFs to use (default: :python:`1`).
        **backend_kwargs: 
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.biquad = IIRFilter(order=2, **backend_kwargs)

    def forward(self, input_signals, twoR, G, c_hp, c_bp, c_lp):
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
        G = torch.tan(HALF_PI * torch.sigmoid(G))
        twoR = TWOR_SCALE * F.softplus(twoR) + 1e-2
        Bs, As = StateVariableFilter.get_biquad_coefficients(twoR, G, c_hp, c_bp, c_lp)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signals, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {
            "twoR": self.num_filters,
            "G": self.num_filters,
            "c_hp": self.num_filters,
            "c_bp": self.num_filters,
            "c_lp": self.num_filters,
        }

    @staticmethod
    def get_biquad_coefficients(twoR, G, c_hp, c_bp, c_lp):
        G_square = G.square()

        b0 = c_hp + c_bp * G + c_lp * G_square
        b1 = -c_hp * 2 + c_lp * 2 * G_square
        b2 = c_hp - c_bp * G + c_lp * G_square

        a0 = 1 + G_square + twoR * G
        a1 = 2 * G_square - 2
        a2 = 1 + G_square - twoR * G

        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)

        return Bs, As


class BaseParametricFilter(nn.Module):
    def __init__(self, **backend_kwargs):
        super().__init__()
        self.biquad = IIRFilter(order=2, **backend_kwargs)

    def forward(self, input_signals, w0, q_inv):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of the inverse of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0, alpha = self.filter_parameter_activations(w0, q_inv)
        cos_w0, alpha = self.compute_common_filter_parameters(w0, alpha)
        Bs, As = self.get_biquad_coefficients(cos_w0, alpha)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signals, Bs, As)
        return output_signal

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        raise NotImplementedError

    @staticmethod
    def filter_parameter_activations(w0, q_inv):
        w0 = PI * torch.sigmoid(w0)
        q_inv = torch.exp(q_inv)
        return w0, q_inv

    @staticmethod
    def compute_common_filter_parameters(w0, q_inv):
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 * q_inv * ALPHA_SCALE
        return cos_w0, alpha

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"w0": 1, "q_inv": 1}


class LowPassFilter(BaseParametricFilter):
    r"""
    Compute a simple second-order low-pass filter.

        $$
        \mathbf{b} &= \left[ \frac{1 - \cos(\omega_0)}{2}, 1 - \cos(\omega_0), \frac{1 - \cos(\omega_0)}{2} \right], \\
        \mathbf{a} &= \left[ 1 + \alpha, -2 \cos(\omega_0), 1 - \alpha \right].
        $$

        These coefficients are calculated with the following pre-activations:
        $\omega_0 = \pi \cdot \sigma(\tilde{w}_0)$ and 
        $\alpha = \sin(\omega_0) / 2q$ where the inverse of the quality factor is simply parameterized as $1 / q = \exp(\tilde{q})$.
        This processor has two learnable parameters: $p = \{\tilde{w}_0, \tilde{q}\}$.

    Args:
        **backend_kwargs: 
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__(**backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        cos_w0_m_1 = cos_w0 - 1
        b0 = cos_w0_m_1 / 2
        b1 = cos_w0_m_1
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class HighPassFilter(BaseParametricFilter):
    r"""
    Compute a simple second-order high-pass filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[ \frac{1 + \cos(\omega_0)}{2}, 1 + \cos(\omega_0), \frac{1 + \cos(\omega_0)}{2} \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.


    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__(**backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        r"""
        Get biquad coefficients for high-pass filter.
        """
        cos_w0_p_1 = 1 + cos_w0
        b0 = cos_w0_p_1 / 2
        b1 = -cos_w0_p_1
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class BandPassFilter(BaseParametricFilter):
    r"""
    Compute a simple second-order band-pass filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[\alpha, 0, -\alpha \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.

    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__(**backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        b0 = alpha
        b1 = torch.zeros_like(b0)
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class BandRejectFilter(BaseParametricFilter):
    r"""
    Compute a simple second-order band-reject filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[ 1, -2 \cos \omega_0, 1 \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.

    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__(**backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        b0 = torch.ones_like(cos_w0)
        b1 = -2 * cos_w0
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class AllPassFilter(BaseParametricFilter):
    r"""
    Compute a simple second-order all-pass filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[a_2, a_1, a_0 \right],
        $$

    and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.

    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__(**backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([a2, a1, a0], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class BaseParametricEqualizerFilter(nn.Module):
    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.biquad = IIRFilter(order=2, **backend_kwargs)

    def forward(self, input_signals, w0, q_inv, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of the inverse of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0, alpha, A = self.filter_parameter_activations(w0, q_inv, log_gain)
        cos_w0, alpha = self.compute_common_filter_parameters(w0, alpha)

        Bs, As = self.get_biquad_coefficients(cos_w0, alpha, A)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signals, Bs, As)
        return output_signal

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha, A):
        raise NotImplementedError

    @staticmethod
    def filter_parameter_activations(w0, q_inv, log_gain):
        w0 = PI * torch.sigmoid(w0)
        q_inv = torch.exp(q_inv)
        A = torch.exp(log_gain)
        return w0, q_inv, A

    @staticmethod
    def compute_common_filter_parameters(w0, q_inv):
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 * q_inv * ALPHA_SCALE
        return cos_w0, alpha

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """

        return {
            "w0": self.num_filters,
            "q_inv": self.num_filters,
            "log_gain": self.num_filters,
        }


class PeakingFilter(BaseParametricEqualizerFilter):
    r"""
    A second-order peaking filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} &= \left[1 + \alpha A, -2 \cos \omega_0, 1 - \alpha A \right], \\
        \mathbf{a} &= \left[1 + \frac{\alpha}{A}, -2 \cos \omega_0, 1 - \frac{\alpha}{A} \right],
        $$

        where $A = \exp(\tilde{A})$.
        We follow the same activations as the :class:`~grafx.processors.filter.LowPassFilter`,
        and the learnable parameters are $p = \{\tilde{w}_0, \tilde{q}, \tilde{A}\}$.


    Args:
        num_filters (:python:`int`, *optional*):
            Number of filters to use (default: :python:`1`).
        **backend_kwargs: 
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__(num_filters=num_filters, **backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha, A):
        alpha_A = alpha * A
        alpha_div_A = alpha / A
        b0 = 1 + alpha_A
        b1 = -2 * cos_w0
        b2 = 1 - alpha_A
        a0 = 1 + alpha_div_A
        a1 = b1
        a2 = 1 - alpha_div_A
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class LowShelf(BaseParametricEqualizerFilter):
    r"""
    A second-order low-shelf filter.

        The biquad coefficients are given as
        $$
        b_0 &= A((A + 1) - (A - 1) \cos \omega_0 + 2 \smash{\sqrt{A}} \alpha), \\
        b_1 &= 2A((A - 1) - (A + 1) \cos \omega_0), \\
        b_2 &= A((A + 1) - (A - 1) \cos \omega_0 - 2 \smash{\sqrt{A}} \alpha), \\
        a_0 &= (A + 1) + (A - 1) \cos \omega_0 + 2 \smash{\sqrt{A}} \alpha, \\
        a_1 &= -2((A - 1) - (A + 1) \cos \omega_0), \\
        a_2 &= (A + 1) + (A - 1) \cos \omega_0 - 2 \smash{\sqrt{A}} \alpha,
        $$

        The remainings are the same as the :class:`~grafx.processors.filter.PeakingFilter`.


    Args:
        num_filters (:python:`int`, *optional*):
            Number of filters to use (default: :python:`1`).
        **backend_kwargs: 
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__(num_filters=num_filters, **backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha, A):
        A_p_1 = A + 1
        A_m_1 = A - 1
        A_p_1_cos_w0 = A_p_1 * cos_w0
        A_m_1_cos_w0 = A_m_1 * cos_w0
        A_sqrt = A.sqrt()
        two_A_sqrt_alpha = 2 * A_sqrt * alpha

        b0 = A * (A_p_1 - A_m_1_cos_w0 + two_A_sqrt_alpha)
        b1 = 2 * A * (A_m_1 - A_p_1_cos_w0)
        b2 = A * (A_p_1 - A_m_1_cos_w0 - two_A_sqrt_alpha)
        a0 = A_p_1 + A_m_1_cos_w0 + two_A_sqrt_alpha
        a1 = -2 * (A_m_1 + A_p_1_cos_w0)
        a2 = A_p_1 + A_m_1_cos_w0 - two_A_sqrt_alpha

        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)

        return Bs, As


class HighShelf(BaseParametricEqualizerFilter):
    r"""
    A second-order high-shelf filter.

        The biquad coefficients are given as
        $$
        b_0 &= A((A + 1) + (A - 1) \cos \omega_0 + 2 \smash{\sqrt{A}} \alpha), \\
        b_1 &= -2A((A - 1) + (A + 1) \cos \omega_0), \\
        b_2 &= A((A + 1) + (A - 1) \cos \omega_0 - 2 \smash{\sqrt{A}} \alpha), \\
        a_0 &= (A + 1) - (A - 1) \cos \omega_0 + 2 \smash{\sqrt{A}} \alpha, \\
        a_1 &= 2((A - 1) + (A + 1) \cos \omega_0), \\
        a_2 &= (A + 1) - (A - 1) \cos \omega_0 - 2 \smash{\sqrt{A}} \alpha,
        $$

        The remainings are the same as the :class:`~grafx.processors.filter.PeakingFilter`.


    Args:
        num_filters (:python:`int`, *optional*):
            Number of filters to use (default: :python:`1`).
        **backend_kwargs: 
            Additional keyword arguments for the :class:`~grafx.processors.core.IIRFilter`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__(num_filters=num_filters, **backend_kwargs)

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha, A):
        A_p_1 = A + 1
        A_m_1 = A - 1
        A_p_1_cos_w0 = A_p_1 * cos_w0
        A_m_1_cos_w0 = A_m_1 * cos_w0
        A_sqrt = A.sqrt()
        two_A_sqrt_alpha = 2 * A_sqrt * alpha

        b0 = A * (A_p_1 + A_m_1_cos_w0 + two_A_sqrt_alpha)
        b1 = -2 * A * (A_m_1 + A_p_1_cos_w0)
        b2 = A * (A_p_1 + A_m_1_cos_w0 - two_A_sqrt_alpha)
        a0 = A_p_1 - A_m_1_cos_w0 + two_A_sqrt_alpha
        a1 = 2 * (A_m_1 - A_p_1_cos_w0)
        a2 = A_p_1 - A_m_1_cos_w0 - two_A_sqrt_alpha

        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)

        return Bs, As
