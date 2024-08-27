import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from grafx.processors.core.utils import rms_difference


class DryWet(nn.Module):
    r"""
    An utility module that mixes the input (dry) with the wrapped processor's output (wet).

        For each pair of input $u[n]$ and output signal $y[n] = f(u[n], p)$
        where $f$ and $p$ denote the wrapped processor and the parameters,
        respectively,
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
            param_size["drywet_weight"] = (1,)
        return param_size


class SerialChain(nn.Module):
    r"""
    A utility module that serially connects the provided processors.

        For processors $f_1, \cdots, f_K$ with their respective parameters $p_1, \cdots, p_K$,
        the serial chain $f = f_K \circ \cdots \circ f_1$
        applies each processor in order, where the output of the previous processor is fed to the next one.
        $$
        y[n] = (f_K \circ \cdots \circ f_1)(s[n]; p_1, \cdots, p_K).
        $$

        The set of all learnable parameters is given as $p = \{p_1, \cdots, p_K\}$.

        Note that, from the audio processing perspective, exactly the same result can be achieved
        by connecting the processors $f_1, \cdots, f_K$ as individual nodes in a graph.
        Yet, this module can be useful when we use the same chain of processors repeatedly
        so that encapsulating them in a single node is more convenient.

    Args:
        processors (:python:`Dict[str, Module]`):
            A dictionary of processors with their names as keys.
            The order of the processors will be the same as the dictionary order.
            We assume that each processor has :python:`forward()` and :python:`parameter_size()`
            method implemented properly.

    """

    def __init__(self, processors):
        super().__init__()
        self.processors = nn.ModuleDict(processors)

    def forward(self, input_signals, **processors_kwargs):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
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
            :python:`Dict[str, Dict[str, Union[dict, Tuple[int, ...]]]]`:
                A nested dictionary of depth at least 2 that contains each processor name as key and its :python:`parameter_size()` as value.
        """
        return {k: v.parameter_size() for k, v in self.processors.items()}


class ParallelMix(nn.Module):
    r"""
    A container that mixes the multiple processor outputs.

        We create a single processor with $K$ processors $f_1, \cdots, f_K$, mixing their outputs with weights $w_1, \cdots, w_K$.
        $$
        y[n] = \sum_{k=1}^K w_k f_k(s[n]; p_k).
        $$

        By default, we take the pre-activation weights $\tilde{w}_1, \cdots, \tilde{w}_K$ as input.
        Then, for each $\tilde{w}_k$, we apply
        $w_k = \log (1 + \exp(\tilde{w}_k)) / K \log{2}$,
        making it non-negative and have value of $1/K$ if the pre-activation input is near zero.
        Also, we can force the weights to have a sum of 1 by applying softmax,
        $w_k = \exp(\tilde{w}_k)/\sum_{i=1}^K \exp(\tilde{w}_i)$.
        This resembles the Differentiable architecture search (DARTS) :cite:`liu2018darts`,
        if our aim is to select the best one among the $K$ processors.
        The set of all learnable parameters is given as $p = \{\tilde{\mathbf{w}}, p_1, \cdots, p_K\}$.
    """

    def __init__(self, processors, activation="softmax"):
        super().__init__()
        self.processors = nn.ModuleDict(processors)
        match activation:
            case "softmax":
                self.get_weight = self._get_softmax_weight
            case "softplus":
                self.get_weight = self._get_softplus_weight
                self.mult = 1 / (math.log(2) * len(self.processors))
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, input_signals, parallel_weights, **processors_kwargs):
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

        weights = self.get_weight(parallel_weights)
        output_signals, intermediates = [], {}
        for i, (k, processor) in enumerate(self.processors.items()):
            out = processor(input_signals, **processors_kwargs[k])
            if isinstance(out, tuple):
                out, intermediates[k] = out
            out = out * weights[..., i, None, None]
            output_signals.append(out)
        output_signals = sum(output_signals)
        return output_signals, intermediates

    def _get_softmax_weight(self, weights):
        return torch.softmax(weights, dim=-1)

    def _get_softplus_weight(self, weights):
        return F.softplus(weights) * self.mult

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Dict[str, Union[dict, Tuple[int, ...]]]]`:
                A nested dictionary of depth at least 2 that contains each processor name as key and its :python:`parameter_size()` as value.
        """
        size = {k: v.parameter_size() for k, v in self.processors.items()}
        size["parallel_weights"] = len(self.processors)
        return size


if __name__ == "__main__":
    serialchain = SerialChain(
        processors={"gain": GainStagingRegularization(), "drywet": DryWet()}
    )


class GainStagingRegularization(nn.Module):
    r"""
    A regularization module that wraps an audio processor and calculates
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
