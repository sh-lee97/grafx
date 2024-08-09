import torch
import torch.nn as nn

from grafx.data.configs import UTILITY_TYPES
from grafx.render.core import (
    aggregate_tensor,
    create_signal_buffer,
    expand_tensor_or_tensor_dict,
    flatten_batch_and_node,
    inplace_write_tensor,
    read_tensor_or_tensor_dict,
)
from grafx.render.prepare import RenderData


def render_grafx(
    processors,
    input_signals,
    per_type_parameters,
    render_data,
    common_parameters=None,
    parameters_grad=True,
    input_signal_grad=False,
):
    r"""
    Renders an audio graph using a specified processing method, handling both batched and non-batched inputs.

    Args:
        processors (:python:`Mapping`):
            Dictionary of audio processors, either in :python:`dict` or :python:`nn.ModuleDict`, where keys are the processor types and values are their processor objects.
        input_signals (:python:`FloatTensor`, :math:`|V_0| \times C \times L` *or* :math:`B\times |V_0| \times C \times L`):
            Tensor of $|V_0|$ input signals, either in three-dimensional or four-dimensional (i.e., with a batch axis) format.
        per_type_parameters (:python:`Mapping`):
            Dictionary of parameters in :python:`dict`, :python:`nn.ParameterDict` or :python:`nn.ModuleDict`. Each key is a processor type and each value is a single tensor or dictionary of tensors.
        render_data (:class:`~grafx.render.prepare.RenderData`):
            Metadata for rendering the graph.
        common_parameters (*optional*):
            A tensor or a dictionary of tensors for the parameter types that are common to all nodes.
            Hence, we expect each tensor's first dimension size to be $|V|$
            (default: :python:`None`).
        parameters_grad (:python:`bool`, *optional*):
            Allow calculation of the parameter gradients; we can omit certain steps to save memory when set to :python:`False`
            (default: :python:`True`).
        input_signal_grad (:python:`bool`, *optional*):
            Allow calculation of the input signal gradients; we can omit certain steps to save memory when set to :python:`False`
            (default: :python:`False`).

    Returns:
        :python:`Tuple[FloatTensor, list, FloatTensor]`: 
        Output audio signals of shape :math:`|V_N| \times C \times L` or :math:`B\times |V_N| \times C \times L`, 
        a list of intermediate tensors, 
        and the signal buffer used for rendering, i.e., all the intermediate outputs, 
        of shape :math:`|V| \times C \times L` or :math:`B\times |V| \times C \times L`.
    """

    method = render_data.method
    ndim = input_signals.ndim
    match ndim:
        case 3:
            node_dim = 0
            postprocess = None

        case 4:
            batch_size, _, channels, audio_len = input_signals.shape
            node_dim = 1
            postprocess = flatten_batch_and_node

            per_type_parameters = expand_tensor_or_tensor_dict(
                per_type_parameters, expand=batch_size, dim=0
            )

            if common_parameters is not None:
                common_parameters = expand_tensor_or_tensor_dict(
                    common_parameters, expand=batch_size, dim=0
                )

        case _:
            raise Exception(
                f"input_signal has shape of {input_signals.shape} ({ndim} ndims), which is not 3 or 4 dims."
            )

    any_grad = parameters_grad or input_signal_grad

    if input_signal_grad:
        signal_buffer = create_signal_buffer(
            method,
            render_data.num_nodes,
            input_signals,
        )
    else:
        with torch.no_grad():
            signal_buffer = create_signal_buffer(
                method,
                render_data.num_nodes,
                input_signals,
            )

    intermediates_list = []

    for i in range(1, render_data.max_order + 1):
        render_i = render_data.iter_list[i]

        input_signals = []
        for read, aggregate in zip(render_i.source_reads, render_i.aggregations):
            input_signal = read_tensor_or_tensor_dict(
                signal_buffer,
                read,
                return_copy=any_grad,
                dim=node_dim,
            )

            input_signal = aggregate_tensor(
                input_signal,
                aggregate,
                dim=node_dim,
            )

            if ndim == 4:
                input_signal = flatten_batch_and_node(input_signal)

            input_signals.append(input_signal)

        node_type = render_i.node_type
        if node_type in processors:
            parameters = read_tensor_or_tensor_dict(
                per_type_parameters[node_type],
                render_i.parameter_read,
                dim=node_dim,
                postprocess=postprocess,
            )

            if common_parameters is not None:
                common_parameters_i = read_tensor_or_tensor_dict(
                    common_parameters,
                    render_i.dest_write,
                    dim=node_dim,
                    postprocess=postprocess,
                )

            else:
                common_parameters_i = {}

            output = processors[node_type](
                *input_signals, **parameters, **common_parameters_i
            )
            if isinstance(output, tuple):
                output_signals, intermediates = output
                intermediates_list.append(intermediates)
            else:
                output_signals = output 

        elif node_type in UTILITY_TYPES:
            output_signals = input_signals
        else:
            raise Exception(f"Wrong node type given: {node_type}")

        if isinstance(output_signals, list):
            num_outputs = len(output_signals)
            if len(output_signals) == 1:
                output_signals = output_signals[0]
            else:
                output_signals = torch.stack(output_signals, -3).view(
                    -1, channels, audio_len
                )

        if ndim == 4:
            output_signals = output_signals.view(batch_size, -1, channels, audio_len)

        inplace_write_tensor(
            method,
            signal_buffer,
            output_signals,
            render_i.dest_write,
            dim=node_dim,
        )

    return output_signals, intermediates_list, signal_buffer
