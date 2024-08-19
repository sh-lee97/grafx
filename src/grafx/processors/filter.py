import math

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import lfilter

from grafx.processors.core.convolution import CausalConvolution
from grafx.processors.core.iir import BiquadFilterBackend, svf_to_biquad

PI = math.pi
TWO_PI = 2 * math.pi
HALF_PI = math.pi / 2
TWOR_SCALE = 1 / math.log(2)
ALPHA_SCALE = 1 / 2


class BiquadFilter(nn.Module):
    r"""
    A serial stack of second-order filters (biquads) with the given coefficients.

        The transfer function of the $K$ stacked biquads $H(z)$ is given as :cite:`smith2007introduction`
        $$
        H(z) = \prod_{i=1}^K H_i(z) = \prod_i \frac{ b_{i, 0} + b_{i, 1} z^{-1} + b_{i, 2} z^{-2}}{a_{i, 0} + a_{i, 1} z^{-1} + a_{i, 2} z^{-2}}.
        $$

        We provide two backends for the filtering.
        The first one, :python:`"lfilter"`, is the time-domain method that computes the difference equation exactly.
        It uses :python:`torchaudio.lfilter`, which uses the direct form I implementation
        (the bar denotes the normalized coefficients by $a_{i, 0}$) :cite:`yu2024differentiable`.
        $$
        x[n] &= \bar{b}_{i, 0} s[n] + \bar{b}_{i, 1} s[n-1] + \bar{b}_{i, 2} s[n-2], \\
        y_i[n] &= x[n] + \bar{a}_{i, 1} y[n-1] + \bar{a}_{i, 2} y[n-2]
        $$

        The second one, :python:`"fsm"`, is the frequency-sampling method (FSM) that approximates the filter with a finite impulse response (FIR)
        by sampling the discrete-time Fourier transform (DTFT) of the filter $H(e^{j\omega})$ at a finite number of points $N$ uniformly 
        :cite:`rabiner70freqsamp, kuznetsov2020differentiable`.
        $$
        H_N[k]
        = \prod_{i=1}^K (H_i)_N[k]
        = \prod_{i=1}^K \frac{b_{i, 0} + b_{i, 1} z_N^{-1} + b_{i, 2} z_N^{-2}}{a_{i, 0} + a_{i, 1} z_N^{-1} + a_{i, 2} z_N^{-2}}.
        $$

        Here, $z_N = \exp(j\cdot 2\pi/N)$ so that $z_N^k$ becomes the $k$-th $N$-point discrete Fourier transform (DFT) bin. 
        Then, the FIR filter $h_N[n]$ is obtained by taking the inverse DFT of the sampled DTFT $H_N[k]$
        and the final output signal is computed by convolving the input signal with the FIR filter as $y[n] = h_N[n] * s[n]$.
        This :python:`"fsm"` backend is faster than the former :python:`"lfilter"` but only an approximation.
        This error is called time-domain aliasing; the frequency-sampled FIR is given as follows :cite:`smith2007mathematics`.
        $$
        h_N[n] = \sum_{m=0}^\infty h[n+mN].
        $$
        
        where $h[n]$ is the true infinite impulse response (IIR). Clearly, increasing the number of samples $N$ reduces the error.
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
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, Bs, A1_pre, A2_pre, A0=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
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

        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
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
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, twoR, G, c_hp, c_bp, c_lp):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
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
        output_signal = self.biquad(input_signal, Bs, As)
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


class LowPassFilter(nn.Module):
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
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__()
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of the inverse of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE

        Bs, As = LowPassFilter.get_biquad_coefficients(cos_w0, alpha)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"w0": 1, "q_inv": 1}

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


class HighPassFilter(nn.Module):
    r"""
    Compute a simple second-order high-pass filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[ \frac{1 + \cos(\omega_0)}{2}, 1 + \cos(\omega_0), \frac{1 + \cos(\omega_0)}{2} \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.


    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__()
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE

        Bs, As = HighPassFilter.get_biquad_coefficients(cos_w0, alpha)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"w0": 1, "q_inv": 1}

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
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


class BandPassFilter(nn.Module):
    r"""
    Compute a simple second-order band-pass filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[ \frac{\sin \omega_0 }{2}, 0, -\frac{\sin \omega_0 }{2} \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.

    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, constant_skirt=False, **backend_kwargs):
        super().__init__()
        self.biquad = BiquadFilterBackend(**backend_kwargs)

        if constant_skirt:
            self.get_biquad_coefficients = (
                BandPassFilter.get_biquad_coefficients_constant_skirt
            )
        else:
            self.get_biquad_coefficients = BandPassFilter.get_biquad_coefficients

    def forward(self, input_signal, w0, q_inv):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        q = 1 / q_inv
        alpha = sin_w0 * q_inv * ALPHA_SCALE

        Bs, As = self.get_biquad_coefficients(cos_w0, alpha, q)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"w0": 1, "q_inv": 1}

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha, q):
        b0 = q * alpha
        b1 = torch.zeros_like(b0)
        b2 = -b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As

    @staticmethod
    def get_biquad_coefficients_constant_skirt(cos_w0, alpha, q):
        b0 = alpha
        b1 = torch.zeros_like(b0)
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([b0, b1, b2], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class BandRejectFilter(nn.Module):
    r"""
    Compute a simple second-order band-reject filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[ 1, -2 \cos \omega_0, 1 \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.

    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__()
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE

        Bs, As = BandRejectFilter.get_biquad_coefficients(cos_w0, alpha)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """

        return {"w0": 1, "q_inv": 1}

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


class AllPassFilter(nn.Module):
    r"""
    Compute a simple second-order all-pass filter.

        The feedforward coefficients are given as
        $$
        \mathbf{b} = \left[ -\sin \omega_0, 1 - \cos \omega_0, -\sin \omega_0 \right],
        $$

        and the remainings are the same as the :class:`~grafx.processors.filter.LowPassFilter`.

    Args:
        **backend_kwargs:
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, **backend_kwargs):
        super().__init__()
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times 1`):
                A batch of quality factors (or resonance).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE

        Bs, As = AllPassFilter.get_biquad_coefficients(cos_w0, alpha)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """

        return {"w0": 1, "q_inv": 1}

    @staticmethod
    def get_biquad_coefficients(cos_w0, alpha):
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        Bs = torch.stack([a2, a1, a0], -1)
        As = torch.stack([a0, a1, a2], -1)
        return Bs, As


class PeakingFilter(nn.Module):
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
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times K`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times K`):
                A batch of quality factors (or resonance).
            log_gain (:python:`FloatTensor`, :math:`B \times K`):
                A batch of log-gains.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE
        A = torch.exp(log_gain)

        Bs, As = PeakingFilter.get_biquad_coefficients(cos_w0, alpha, A)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

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


class LowShelf(nn.Module):
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
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times K`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times K`):
                A batch of quality factors (or resonance).
            log_gain (:python:`FloatTensor`, :math:`B \times K`):
                A batch of log-gains.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE
        A = torch.exp(log_gain)

        Bs, As = LowShelf.get_biquad_coefficients(cos_w0, alpha, A)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

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


class HighShelf(nn.Module):
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
            Additional keyword arguments for the :class:`~grafx.processors.core.BiquadFilterBackend`.
    """

    def __init__(self, num_filters=1, **backend_kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.biquad = BiquadFilterBackend(**backend_kwargs)

    def forward(self, input_signal, w0, q_inv, log_gain):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            w0 (:python:`FloatTensor`, :math:`B \times K`):
                A batch of cutoff frequencies.
            q_inv (:python:`FloatTensor`, :math:`B \times K`):
                A batch of quality factors (or resonance).
            log_gain (:python:`FloatTensor`, :math:`B \times K`):
                A batch of log-gains.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """

        w0 = PI * torch.sigmoid(w0)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        q_inv = torch.exp(q_inv)
        alpha = sin_w0 * q_inv * ALPHA_SCALE
        A = torch.exp(log_gain)

        Bs, As = HighShelf.get_biquad_coefficients(cos_w0, alpha, A)
        Bs, As = Bs.unsqueeze(1), As.unsqueeze(1)
        output_signal = self.biquad(input_signal, Bs, As)
        return output_signal

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
