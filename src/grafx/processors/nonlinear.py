import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhDistortion(nn.Module):
    r"""
    A simple distortion processor based on the hyperbolic tangent function.
    :cite:`peladeau2024blind`

        In the simplest setting, the processor only applies the hyperbolic tangent function to the input signal:
        $
        y[n] = \tanh(u[n]).
        $
        The processor can be set to apply pre-gain $g_{\mathrm{pre}}$ and post-gain $g_{\mathrm{post}}$
        before and after the nonlinearity, respectively.
        We can also add bias $b$ for asymmetric and increased distortion.
        The full processing is then given by
        $$
        y[n] = g_{\mathrm{post}} (\tanh(g_{\mathrm{pre}} \cdot u[n] + b) - \tanh b ).
        $$

        This processor's parameters are $p = \{\tilde{g}_{\mathrm{pre}}, \tilde{g}_{\mathrm{post}}, b\}$,
        where $\tilde{g}_{\mathrm{pre}} = \log g_{\mathrm{pre}}$ and $\tilde{g}_{\mathrm{post}} = \log g_{\mathrm{post}}$.
        Depending on the initializations, each parameter can be omitted.


    Args:
        pre_post_gain (:python:`bool`, *optional*):
            If :python:`True`, we apply the pre- and post-gain
            (default: :python:`True`).
        inverse_post_gain (:python:`bool`, *optional*):
            If :python:`True`, we set the post-gain as an inverse of the pre-gain
            (default: :python:`True`).
        remove_dc (:python:`bool`, *optional*):
            If :python:`True`, we pre-process the input signal to remove the DC component
            (default: :python:`False`).
        use_bias (:python:`bool`, *optional*):
            If :python:`True`, we apply the bias term
            (default: :python:`False`).

    """

    def __init__(
        self,
        pre_post_gain=True,
        inverse_post_gain=True,
        remove_dc=False,
        use_bias=False,
    ):
        super().__init__()
        self.pre_post_gain = pre_post_gain
        self.inverse_post_gain = inverse_post_gain
        self.remove_dc = remove_dc
        self.use_bias = use_bias

    def forward(self, input_signals, log_pre_gain=None, log_post_gain=None, bias=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_pre_gain (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of log pre-gain values, only required if :python:`pre_post_gain` is :python:`True`
                (default: :python:`None`).
            log_post_gain (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of log post-gain values, only required if
                both :python:`pre_post_gain` and :python:`inverse_post_gain` are :python:`False`
                (default: :python:`None`).
            bias (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of bias values, only required if :python:`use_bias` is :python:`True`
                (default: :python:`None`).


        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        if self.remove_dc:
            input_signals = input_signals - input_signals.mean(-1, keepdims=True)

        if self.pre_post_gain:
            pre_gain = torch.exp(log_pre_gain).unsqueeze(-1)
            input_signals = input_signals * pre_gain

        if self.use_bias:
            bias = bias.unsqueeze(-1)
            output_signals = torch.tanh(input_signals + bias) - torch.tanh(bias)
        else:
            output_signals = torch.tanh(input_signals)

        if self.pre_post_gain:
            if self.inverse_post_gain:
                post_gain = 1 / pre_gain
            else:
                post_gain = torch.exp(log_post_gain).unsqueeze(-1)
            output_signals = output_signals * post_gain
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {}
        if self.pre_post_gain:
            size["log_pre_gain"] = 1
            if not self.inverse_post_gain:
                size["log_post_gain"] = 1
        if self.use_bias:
            size["bias"] = 1
        return size


class PiecewiseTanhDistortion(nn.Module):
    r"""
    A distortion processor based on the piecewise hyperbolic tangent function :cite:`eichas2020system`.

        The nonlinearity is split into three parts.
        The middle part is the standard hyperbolic tangent function, and the other two parts are its scaled and shifted versions.
        From two segment thresholds $0 < k_p, k_n < 1$ and hardness controls $h_p, h_n > 0$, the nonlinearity is given as 
        $$
        \xi(u[n]) = 
        \begin{cases}
        a_p \cdot \tanh \left(h_p \cdot\left(u[n]-k_p\right)\right)+b_p & k_p < u[n], \\ 
        \tanh (u[n]) & -k_n \leq u[n] \leq k_p, \\ 
        a_n \cdot \tanh \left(h_n \cdot\left(u[n]+k_n\right)\right)+b_n & u[n]<-k_n
        \end{cases}
        $$

        where $a_p = (1-\tanh^2 k_p) / h_p$, $a_n = (1-\tanh^2 k_n) / h_n$, $b_p = \tanh k_p$, and $b_n = -\tanh k_n$.
        In the simplest setting, the output is given as $y[n] = \xi(u[n])$. 
        Same as :class:`~grafx.processors.nonlinear.TanhDistortion`, 
        we can optionally apply pre- and post-gain as $y[n] = g_{\mathrm{post}} \cdot \xi(g_{\mathrm{pre}} \cdot u[n])$.
        This processor has parameters of $\smash{p = \{\tilde{g}_{\mathrm{pre}}, \tilde{g}_{\mathrm{post}}, \smash{\tilde{\mathbf{k}}}, \tilde{\mathbf{h}}\}}$,
        where $\smash{\tilde{\mathbf{k}} = [\tilde{k}_p, \tilde{k}_n]}$ and $\smash{\tilde{\mathbf{h}} = [\tilde{h}_p, \tilde{h}_n]}$.
        The internal parameters are recovered with 
        $\smash{k_p = \sigma (\tilde{k}_p)}$, 
        $\smash{k_n = \sigma (\tilde{k}_n)}$, 
        $\smash{h_p = \exp \tilde{h}_p}$, and 
        $\smash{h_n = \exp \tilde{h}_n}$.

        
    """

    def __init__(self, pre_post_gain=True, inverse_post_gain=True, remove_dc=False):
        super().__init__()
        self.pre_post_gain = pre_post_gain
        self.inverse_post_gain = inverse_post_gain
        self.remove_dc = remove_dc

    def forward(
        self,
        input_signals,
        log_hardness,
        z_threshold,
        log_pre_gain=None,
        log_post_gain=None,
    ):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_hardness (:python:`FloatTensor`, :math:`B \times 2`):
                A batch of hardness controls stacked to the last dimension.
            z_threshold (:python:`FloatTensor`, :math:`B \times 2`):
                A batch of threshold values stacked to the last dimension.
            log_pre_gain (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of log pre-gain values, only required if :python:`pre_post_gain` is :python:`True`
                (default: :python:`None`).
            log_post_gain (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of log post-gain values, only required if
                both :python:`pre_post_gain` and :python:`inverse_post_gain` are :python:`False`
                (default: :python:`None`).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        if self.remove_dc:
            input_signals = input_signals - input_signals.mean(-1, keepdims=True)

        if self.pre_post_gain:
            pre_gain = torch.exp(log_pre_gain).unsqueeze(-1)
            input_signals = input_signals * pre_gain

        hardness = torch.exp(log_hardness)
        threshold = torch.sigmoid(z_threshold)

        output_signals = self.apply_distortion(input_signals, hardness, threshold)

        if self.pre_post_gain:
            if self.inverse_post_gain:
                post_gain = 1 / pre_gain
            else:
                post_gain = torch.exp(log_post_gain).unsqueeze(-1)
            output_signals = output_signals * post_gain
        return output_signals

    @staticmethod
    def apply_distortion(input_signals, hardness, threshold):
        hardness, threshold = hardness.unsqueeze(-2), threshold.unsqueeze(-2)

        kn, kp = threshold.split(1, dim=-1)
        gp, gn = hardness.split(1, dim=-1)

        ap, an = (1 - torch.tanh(kp)) / gp, (1 - torch.tanh(kn)) / gn
        bp, bn = torch.tanh(kp), -torch.tanh(kn)

        output_signals = torch.zeros_like(input_signals)

        above_mask = input_signals > kp
        below_mask = input_signals < -kn
        middle_mask = ~above_mask & ~below_mask

        above = ap * torch.tanh(gp * (input_signals - kp)) + bp
        middle = torch.tanh(input_signals)
        below = an * torch.tanh(gn * (input_signals + kn)) + bn

        output_signals = above_mask * above + middle_mask * middle + below_mask * below
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"log_hardness": 2, "z_threshold": 2}
        if self.pre_post_gain:
            size["log_pre_gain"] = 1
            if not self.inverse_post_gain:
                size["log_post_gain"] = 1
        return size


class PowerDistortion(nn.Module):
    r"""
    A distortion processor based on polynomials :cite:`peladeau2024blind`.

        The distortion output is simply give as an elementwise (memoryless) polynomial of each sample of the input signal.
        $$
        y[n] = \sum_{k=0}^{K-1} w_k u^k[n].
        $$

        where $w_k$ is the polynomial coefficients, which we also call :python:`basis_weights` in this implementation.
        We allow optional use of hypertangent and its pre-gain after the polynomial
        :cite:`colonel2022reverse`.
        In this case, the output will be given as follows,
        $$
        y[n] = \sum_{k=0}^{K-1} w_k \tanh (g_{\mathrm{pre}} u^k[n]).
        $$

        The processor has parameters of $p = \{\mathbf{w}, \tilde{\mathbf{g}}_{\mathrm{pre}}\}$,
        where the former is a stack of the coefficients and the latter is the optional log pre-gain values.
    """

    def __init__(self, max_order=10, pre_gain=True, remove_dc=False, use_tanh=False):
        super().__init__()

        assert max_order > 1

        self.pre_gain = pre_gain
        self.max_order = max_order
        self.remove_dc = remove_dc
        self.use_tanh = use_tanh

        arange = torch.arange(self.max_order)
        arange = arange[:, None, None, None]
        self.register_buffer("arange", arange)

    def forward(self, input_signals, basis_weights, log_pre_gain=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            basis_weights (:python:`FloatTensor`, :math:`B \times K`):
                A batch of polynomial coefficients.
            log_pre_gain (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of log pre-gain values, only required if :python:`pre_gain` is :python:`True`
                (default: :python:`None`).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        if self.remove_dc:
            input_signals = input_signals - input_signals.mean(-1, keepdims=True)

        if self.pre_gain:
            pre_gain = torch.exp(log_pre_gain).unsqueeze(-1)
            input_signals = input_signals * pre_gain

        basis_weights = torch.tanh(basis_weights)
        basis_weights = basis_weights.T[:, :, None, None]

        powers = torch.pow(input_signals.unsqueeze(0), self.arange)
        if self.use_tanh:
            powers = torch.tanh(powers)
        output_signals = (powers * basis_weights).sum(0)
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"basis_weights": self.max_order}
        if self.pre_gain:
            size["log_pre_gain"] = 1
        return size


class ChebyshevDistortion(nn.Module):
    r"""
    A distortion processor based on Chebyshev polynomials :cite:`peladeau2024blind`.

        We combine the outputs of Chebyshev polynomials of the input signal.
        $$
        y[n] = \sum_{k=0}^{K-1} w_k T_k(u[n]).
        $$

        where $T_k(u[n])$ is the Chebyshev polynomial of order $k$ defined as follows,
        $$
        \begin{aligned}
        T_0(u[n])&=1, \\ 
        T_1(u[n])&=u[n], \\
        T_k(u[n])&=2 x T_{k-1}(u[n])-T_{k-2}(u[n]) 
        \end{aligned}
        $$

        Same as the :class:`~grafx.processors.nonlinear.PowerDistortion`,
        we allow optional use of hypertangent and its pre-gain after the Chebyshev 
        :cite:`colonel2022reverse`.
        $$
        y[n] = \sum_{k=0}^{K-1} w_k \tanh T_k(g_{\mathrm{pre}} u[n]).
        $$

        The processor has parameters of $p = \{\mathbf{w}, \tilde{\mathbf{g}}_{\mathrm{pre}}\}$,
        where the latter is optional.
    """

    def __init__(self, max_order=10, pre_gain=True, remove_dc=False, use_tanh=False):
        super().__init__()

        assert max_order > 1

        self.pre_gain = pre_gain
        self.max_order = max_order
        self.remove_dc = remove_dc
        self.use_tanh = use_tanh

    def forward(self, input_signals, basis_weights, log_pre_gain=None):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            basis_weights (:python:`FloatTensor`, :math:`B \times K`):
                A batch of Chebyshev polynomial coefficients.
            log_pre_gain (:python:`FloatTensor`, :math:`B \times 1`, *optional*):
                A batch of log pre-gain
                values, only required if :python:`pre_gain` is :python:`True`
                (default: :python:`None`).

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        if self.remove_dc:
            input_signals = input_signals - input_signals.mean(-1, keepdims=True)

        if self.pre_gain:
            pre_gain = torch.exp(log_pre_gain).unsqueeze(-1)
            input_signals = input_signals * pre_gain

        basis_weights = torch.tanh(basis_weights)
        output_signals = self.apply_distortion(
            input_signals, basis_weights, self.use_tanh
        )
        return output_signals

    @staticmethod
    def apply_distortion(input_signals, basis_weights, use_tanh=False):
        max_order = basis_weights.shape[-1]
        shape = input_signals.shape
        b, _, _ = shape
        device = input_signals.device

        chebyshev = torch.zeros(max_order, *shape, device=device)
        chebyshev[0] = torch.ones_like(input_signals)
        chebyshev[1] = input_signals

        for k in range(2, max_order):
            chebyshev[k] = 2 * input_signals * chebyshev[k - 1] - chebyshev[k - 2]

        if use_tanh:
            chebyshev = torch.tanh(chebyshev)

        chebyshev = chebyshev * basis_weights.T.view(max_order, b, 1, 1)
        output_signals = chebyshev.sum(0)
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"basis_weights": self.max_order}
        if self.pre_gain:
            size["log_pre_gain"] = 1
        return size
