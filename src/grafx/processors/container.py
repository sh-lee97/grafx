import torch.nn as nn

from grafx.processors.core.utils import rms_difference


class GainStagingRegularization(nn.Module):
    r"""
    An regularization module that wraps an audio processor and calculates
    the energy differences between the input and output audio.
    It can be used guide the processors to mimic *gain-staging,*
    a practice that aims to keep the signal energy
    roughly the same throughout the processing chain.


        For each pair of input $u[n]$ and output signal $y[n] = f(u[n], p)$
        where $f$ and $p$ denote the wrapped processor and the parameters,
        respeectively,
        we calculate their loudness difference
        with an energy function $\sigma$ as follows,
        $$
        d = \left| g(y[n]) - g(u[n]) \right|.
        $$

        The energy function $g$ computes log of
        mean energy across the time and channel axis.
        If the signals are stereo, then it is equivalent to calculating
        the log of mid-channel energy.

    Args:
        processor (:python:`Module`):
            Any SISO processor with :python:`forward` and :python:`parameter_size` method implemented properly.
        key (:python:`str`, *optional*):
            A dictionary key that will be used to store energy difference in the intermediate results.
            (default: :python:`"gain_reg"`)


    """

    def __init__(self, processor, key="gain_reg"):
        super().__init__()
        self.processor = processor
        self.key = key

    def forward(self, input_signals, **processor_kwargs):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals that will be passed to the processor.
            **processor_kwargs (optional):
                Keyword arguments (i.e., mostly parameters)
                that will be passed to the processor.

        Returns:
            :python:`Tuple[FloatTensor, dict]`: A batch of output signals of shape :math:`B \times C \times L`
            and dictionary of intermediate/auxiliary results added with the regularization loss.
        """
        out = self.processor(input_signals, **processor_kwargs)
        if isinstance(out, tuple):
            output_signals, intermediates = out
        else:
            output_signals, intermediates = out, {}
        gain_reg = rms_difference(input_signals, output_signals)
        assert not self.key in intermediates
        intermediates[self.key] = gain_reg
        return output_signals, intermediates

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: The wrapped processor's :python:`parameter_size()`.
        """
        return self.processor.parameter_size()


class DryWet(nn.Module):
    r"""
    An utility module that mixes the *dry* input with the wrapped processor's *wet* output.

        For each pair of input $u[n]$ and output signal $y[n] = f(u[n], p)$
        where $f$ and $p$ denote the wrapped processor and the parameters,
        respeectively,
        we mix the input and output with a dry/wet mix $0 < w < 1$ as follows,
        $$
        y[n] = (1 - w)u[n]  + w y[n].
        $$

        Here, the dry/wet is further parameterized as $w = \sigma(z_w)$
        where $z_w$ is an unbounded logit and $\sigma$ is logistic sigmoid.
        Hence, this processor's learnable parameter is $p \cup \{z_w\}$.


    Args:
        processor (:python:`Module`):
            Any SISO processor with :python:`forward` and :python:`parameter_size` method implemented properly.
        external_param (:python:`bool`, *optional*):
            If set to :python:`True`, we do not add our dry/wet weight shape to the
            :python:`parameter_size` method.
            This is useful when every processor uses :python:`DryWet`
            and it is more convinient to have a single dry/wet tensor
            for entire nodes instead of keeping a tensor for each type
            (default: :python:`True`).

    """

    def __init__(self, processor, external_param=True):
        super().__init__()
        self.processor = processor
        self.external_param = external_param

    def forward(self, input_signals, drywet_weight, **processor_kwargs):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals that will be passed to the processor.
            **processor_kwargs (optional):
                Keyword arguments (i.e., mostly parameters)
                that will be passed to the processor.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        out = self.processor(input_signals, **processor_kwargs)
        if isinstance(out, tuple):
            output_signals, intermediates = out
        else:
            output_signals = out
        drywet_weight = drywet_weight.view(-1, 1, 1)
        output_signals = (
            drywet_weight * output_signals + (1 - drywet_weight) * input_signals
        )
        if isinstance(out, tuple):
            return output_signals, intermediates
        else:
            return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: The wrapped processor's :python:`parameter_size()`,
            optionally added with the dry/wet weight when :python:`external_param` is set to :python:`False`.
        """
        param_size = self.processor.parameter_size()
        if not self.external_param:
            param_size["weight"] = (1,)
        return param_size


class SerialChain(nn.Module):
    r"""
    An utility module that serially connects the provided processors in a sequential order.

        For processors $f_1, \cdots, f_K$, its output is defined
        $$
        y[n] = (f_K \circ \cdots \circ f_1)(s[n]; p_1, \cdots, p_K) = f_K(\cdots f_1(s[n]; p_1); p_K).
        $$
    """

    def __init__(self, processors):
        super().__init__()
        self.processors = nn.ModuleDict(processors)

    def forward(self, input_signals, **processors_kwargs):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signal (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            **processors_kwargs (*optional*):
                Keyword arguments (i.e., mostly parameters) that will be passed to the processor.

        Returns:
            :python:`Tuple[FloatTensor, Dict[str, Any]]`:
                A batch of output signals of shape :math:`B \times C \times L`.
        """

        output_signals = input_signals
        intermediates = {}

        for k, processor in self.processors.items():
            out = processor(output_signals, **processors_kwargs[k])
            if isinstance(out, tuple):
                output_signals, intermediates[k] = out
            else:
                output_signals = out
        return output_signals, intermediates

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A nested dictionary that contains each parameter tensor's shape.
        """
        return {k: v.parameter_size() for k, v in self.processors.items()}


class ParallelMix(nn.Module):
    r"""
    Differentiable architecture search (DARTS) :cite:`liu2018darts`

        For processors $f_1, \cdots, f_K$, its output is defined as a weighted sum of the processors' outputs,
        $$
        y[n] = \underbrace{w_0 s[n]}_{\mathrm{optional}} + \sum_{k=1}^K w_k f_k(s[n]; p_k)
        $$

        where each weight is defined as a function of learnable parameters $\alpha_0, \cdots, \alpha_K$ as follows,
        $$
        w_k = \frac{\exp(\alpha_k)}{\sum_{i=0}^K \exp(\alpha_i)}
        $$

        $p = \{\boldsymbol{\alpha}\} \cup p_1 \cup \cdots \cup p_K\}$ is the set of all learnable parameters.
    """

    def __init__(self, processors):
        super().__init__()
        self.processors = nn.ModuleDict(processors)

    def forward(self, input_signals, **processors_kwargs):
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

        out = self.processor(input_signals, **processor_kwargs)
        if isinstance(out, tuple):
            output_signals, intermediates = out
        else:
            output_signals, intermediates = out, {}
        drywet_weight = drywet_weight.view(-1, 1, 1)
        output_signals = (
            drywet_weight * output_signals + (1 - drywet_weight) * input_signals
        )
        return output_signals, intermediates

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A nested dictionary that contains each parameter tensor's shape.
        """
        return {k: v.parameter_size() for k, v in self.processors.items()}


if __name__ == "__main__":
    serialchain = SerialChain(
        processors={"gain": GainStagingRegularization(), "drywet": DryWet()}
    )
