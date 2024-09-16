import warnings

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import lfilter
from functools import reduce, partial
from torchlpc import sample_wise_lpc

from grafx.processors.core.convolution import FIRConvolution
from grafx.processors.core.midside import lr_to_ms, ms_to_lr

TORCHAUDIO_VERSION = torchaudio.__version__


if TORCHAUDIO_VERSION < "2.4.0":
    warnings.warn(
        f"The current version of torchaudio ({TORCHAUDIO_VERSION}) provides lfilter that could be either slow or faulty. For example, see https://github.com/pytorch/audio/releases/tag/v2.4.0. We recommend using torchaudio>=2.4.0 or using the frequency-sampled version instead."
    )


class IIRFilter(nn.Module):
    r"""
    A serial stack of second-order filters (biquads) with the given coefficients.

        The transfer function of the $K$ stacked biquads $H(z)$ is given as :cite:`smith2007introduction`
        $$
        H(z) = \prod_{k=1}^K H_k(z) = \prod_k \frac{ b_{k, 0} + b_{k, 1} z^{-1} + b_{i, 2} z^{-2}}{a_{i, 0} + a_{i, 1} z^{-1} + a_{i, 2} z^{-2}}.
        $$

        We provide three backends for the filtering.
        The first one, :python:`"lfilter"`, is the time-domain method that computes the difference equation exactly.
        It uses :python:`torchaudio.lfilter`, which uses the direct form I implementation
        (the bar denotes the normalized coefficients by $a_{i, 0}$) :cite:`yu2024differentiable`.
        $$
        x[n] &= \bar{b}_{i, 0} s[n] + \bar{b}_{i, 1} s[n-1] + \bar{b}_{i, 2} s[n-2], \\
        y_i[n] &= x[n] - \bar{a}_{i, 1} y[n-1] - \bar{a}_{i, 2} y[n-2]
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

        The third one, :python:`"ssm"`, is based on the diagonalisation of the state-space model (SSM) of the biquad filter so it only works for the second-order filters.
        The direct form II implementation of the biquad filter can be written in state-space form :cite:`smith2007introduction` as
        $$
        x_i[n+1] &= A_i x_i[n] + B_i s[n], \\
        y_i[n] &= C_i x_i[n] + \bar{b}_{i, 0} s[n],
        $$
        $$
        A_i = \begin{bmatrix}-\bar{a}_{i, 1} & -\bar{a}_{i, 2} \\ 1 & 0 \end{bmatrix}, \quad 
        B_i &= \begin{bmatrix}1 \\ 0 \end{bmatrix}, \quad 
        C_i = \begin{bmatrix}\bar{b}_{i, 1} - \bar{b}_{i, 0} \bar{a}_{i, 1} & \bar{b}_{i, 2} - \bar{b}_{i, 0} \bar{a}_{i, 2} \end{bmatrix}.
        $$
        If the poles of the filter are unique, the transition matrix $A_i$ can be decomposed as $A_i = V_i \Lambda_i V_i^{-1}$ where $\Lambda_i$ is either a diagonal matrix with real poles on the diagonal or a scaled rotation matrix, which can be represented by one of the complex conjugate poles.
        Using this decomposition, the filter can be implemented as first-order recursive filters on the projected siganl $V_i^{-1} B_i s[n]$, where we leverage `Parallel Scan` :cite:`martin2018parallelizing` to speed up the computation on the GPU.
        Finally, the output is projected back to the original basis using $V_i$. 

        We recommend using the :python:`"ssm"` over the :python:`"lfilter"` backend in general, not only because it runs several times faster on the GPU but it's more numerically stable.

    Args:
        num_filters (:python:`int`, *optional*):
            Number of biquads to use (default: :python:`1`).
        normalized (:python:`bool`, *optional*):
            If set to :python:`True`, the filter coefficients are assumed to be normalized by $a_{i, 0}$,
            making the number of learnable parameters $5$ per biquad instead of $6$
            (default: :python:`False`).
        backend (:python:`str`, *optional*):
            The backend to use for the filtering, which can either be the frequency-sampling method :python:`"fsm"` 
            or exact time-domain filters, :python:`"lfilter"` or :python:`"ssm"` (default: :python:`"fsm"`).
        fsm_fir_len (:python:`int`, *optional*):
            The length of FIR approximation when :python:`backend == "fsm"` (default: :python:`8192`).
    """

    def __init__(
        self,
        order=2,
        backend="fsm",
        flashfftconv=True,
        fsm_fir_len=4000,
        fsm_max_input_len=2**17,
        fsm_regularization=False,
    ):
        super().__init__()
        self.backend = backend
        self.fsm_fir_len = fsm_fir_len
        self.fsm_regularization = fsm_regularization

        if flashfftconv:
            assert fsm_fir_len % 2 == 0
            assert fsm_max_input_len % 2 == 0

        match backend:
            case "fsm":
                delays = IIRFilter.delay(torch.arange(order + 1), fsm_fir_len)
                self.register_buffer("delays", delays)
                self.conv = FIRConvolution(
                    mode="causal",
                    flashfftconv=flashfftconv,
                    max_input_len=fsm_max_input_len,
                )
                if fsm_regularization:
                    assert False
                self.process = self._process_fsm
            case "lfilter":
                self.process = self._process_lfilter
            case "ssm":
                self.process = self._process_ssm
            case _:
                raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, input_signal, Bs, As):
        r"""
        Apply the IIR filter to the input signal and the given coefficients.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C_\mathrm{in} \times L`):
                A batch of input audio signals.
            Bs (:python:`FloatTensor`, :math:`B \times C_\mathrm{filter} \times K \times 3`):
                A batch of biquad coefficients, $b_{i, 0}, b_{i, 1}, b_{i, 2}$, stacked in the last dimension.
            As (:python:`FloatTensor`, :math:`B \times C_\mathrm{filter} \times K \times 3`):
                A batch of biquad coefficients, $b_{i, 0}, b_{i, 1}, b_{i, 2}$, stacked in the last dimension.
        """
        return self.process(input_signal, Bs, As)

    def _process_fsm(self, input_signal, Bs, As):
        fsm_response = IIRFilter.iir_fsm(Bs, As, delays=self.delays)
        fsm_response = fsm_response.prod(-2)
        fsm_fir = torch.fft.irfft(fsm_response, dim=-1, n=self.fsm_fir_len)
        output_signal = self.conv(input_signal, fsm_fir)
        return output_signal

    def _process_lfilter(self, input_signal, Bs, As):
        b, c_signal, audio_len = input_signal.shape
        _, c_filter, num_biquads, _ = Bs.shape

        if c_signal == 1 and c_filter > 1:
            input_signal = input_signal.repeat(1, c_filter, 1)
            c = c_filter
        elif c_signal > 1 and c_filter == 1:
            Bs = Bs.repeat(1, c_signal, 1, 1)
            As = As.repeat(1, c_signal, 1, 1)
            c = c_signal
        else:
            assert c_filter == c_signal
            c = c_signal

        input_signal = input_signal.view(b * c, audio_len)
        Bs = Bs.view(b * c, num_biquads, 3)
        As = As.view(b * c, num_biquads, 3)

        output_signal = input_signal
        num_filters = Bs.shape[-2]
        for i in range(num_filters):
            output_signal = lfilter(
                output_signal,
                b_coeffs=Bs[:, i, :],
                a_coeffs=As[:, i, :],
                batching=True,
                clamp=False,
            )
        output_signal = output_signal.view(b, c, audio_len)
        return output_signal

    def _process_ssm(self, input_signal, Bs, As):
        batch, num_channels, audio_len = input_signal.shape
        assert Bs.shape[-1] == As.shape[-1] == 3, "The filter order must be 2."

        if num_channels == 1:
            input_signal = input_signal.repeat(1, Bs.shape[1], 1)
            num_channels = Bs.shape[1]
        elif Bs.shape[1] == 1:
            Bs = Bs.repeat(1, num_channels, 1, 1)
            As = As.repeat(1, num_channels, 1, 1)
        else:
            assert num_channels == Bs.shape[1], "The number of channels must match."

        input_signal = input_signal.reshape(batch * num_channels, audio_len)
        Bs = Bs.reshape(batch * num_channels, -1, 3)
        As = As.reshape(batch * num_channels, -1, 3)
        K = Bs.shape[1]

        # normalise the coefficients so that a_0 = 1
        Bs = Bs / As[:, :, :1]
        a12 = As[:, :, 1:] / As[:, :, :1]

        # treat the direct component separately
        b0 = Bs[:, :, :1]
        b12 = Bs[:, :, 1:] - b0 * a12

        def filter_runner(x, cur_b0, cur_b12, cur_a12):
            # find the roots of transition matrix A
            # first, check are they real or complex conjugates
            tmp = cur_a12[..., 0] ** 2 - 4 * cur_a12[..., 1]
            complex_poles_mask = tmp < 0
            real_poles_mask = tmp > 0
            double_poles_mask = ~complex_poles_mask & ~real_poles_mask

            output_signal = torch.zeros_like(x)
            if complex_poles_mask.any():
                complex_poles = 0.5 * (
                    -cur_a12[complex_poles_mask][..., 0]
                    + 1j * torch.sqrt(-tmp[complex_poles_mask])
                )
                output_signal[complex_poles_mask] = _ssm_complex_conjugate(
                    input_signal[complex_poles_mask],
                    cur_b12[complex_poles_mask],
                    complex_poles,
                )
            if real_poles_mask.any():
                root_term = torch.sqrt(tmp[real_poles_mask])
                poles = 0.5 * (
                    -cur_a12[real_poles_mask][..., :1]
                    + torch.stack((root_term, -root_term), -1)
                )
                output_signal[real_poles_mask] = _ssm_real_pole(
                    input_signal[real_poles_mask],
                    cur_b12[real_poles_mask],
                    poles,
                )

            if double_poles_mask.any():
                poles = -cur_a12[double_poles_mask][..., 0] * 0.5
                output_signal[double_poles_mask] = _ssm_double_real_pole(
                    input_signal[double_poles_mask],
                    cur_b12[double_poles_mask],
                    poles,
                )

            return cur_b0 * x + torch.cat(
                [torch.zeros_like(output_signal[:, :1]), output_signal[:, :-1]], -1
            )

        output_signal = reduce(
            lambda x, args: filter_runner(x, *args),
            zip(b0.unbind(1), b12.unbind(1), a12.unbind(1)),
            input_signal,
        )

        return output_signal.reshape(batch, num_channels, audio_len)

    @staticmethod
    def iir_fsm(Bs, As, delays, eps=1e-10):
        Bs, As = Bs.unsqueeze(-1), As.unsqueeze(-1)
        biquad_response = torch.sum(Bs * delays, -2) / torch.sum(As * delays, -2)
        return biquad_response

    @staticmethod
    def delay(delay_length, fir_length):
        ndim = delay_length.ndim
        arange = torch.arange(fir_length // 2 + 1, device=delay_length.device)
        arange = arange.view((1,) * ndim + (-1,))
        phase = delay_length.unsqueeze(-1) * arange / fir_length * 2 * np.pi
        delay = torch.exp(-1j * phase)
        return delay


def _first_order_recursive_filter(x, a):
    # x: (batch, T)
    # a: (batch,)
    return sample_wise_lpc(x, -a[:, None, None].expand(-1, x.shape[1], -1))


def _ssm_complex_conjugate(x, b12, complex_pole: torch.Tensor):
    #     |2 * Re(p) -|p|^2|
    # A = |1         0     | = V^-1 @ R @ V
    #
    #     |0  Im(p)|
    # V = |-1  Re(p)|
    #
    # The matrix R is given by
    #     |Re(p)  -Im(p)|
    # R = |Im(p)   Re(p)|
    # which is a rotation matrix and can be written as complex exponential |p|e^{j angle(p)} = p.
    # We utilise this fact to filter the signal using one complex first-order filter.
    # This kind of filter is also known as `Minimum Norm Recursive Filter`.
    u = -1j * x
    h = _first_order_recursive_filter(u, complex_pole)
    b1 = b12[..., 0]
    b2 = b12[..., 1]
    return (
        h.real
        * (b1 * complex_pole.real / complex_pole.imag + b2 / complex_pole.imag)[
            ..., None
        ]
        - b1[..., None] * h.imag
    )


def _ssm_real_pole(x, b12, poles):
    #     |p1 + p2  -p1 * p2|
    # A = |1        0       | = V @ diag(p1, p2) @ V^-1
    #
    #     |1      1    |
    # V = |p1^-1  p2^-1|
    pole_1, pole_2 = poles.unbind(-1)
    diff = pole_1 - pole_2
    u = (
        x.unsqueeze(1)
        * (torch.stack([pole_1, -pole_2], dim=-1) / diff[:, None])[..., None]
    )
    h = _first_order_recursive_filter(
        u.reshape(-1, u.shape[-1]), poles.reshape(-1)
    ).view_as(u)
    b1 = b12[..., :1]
    b2 = b12[..., 1:]
    return b1 * h.sum(1) + b2 * torch.sum(h * poles.reciprocal().unsqueeze(-1), 1)


def _ssm_double_real_pole(x, b12, pole):
    # Since we cannot perform the diagonalisation when double poles are present, we just perform two first-order recursive filters in series.
    runner = partial(_first_order_recursive_filter, a=pole)
    h = reduce(lambda u, f: f(u), [runner] * 2, x)
    return (
        h * b12[..., :1]
        + torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), -1) * b12[..., 1:]
    )
