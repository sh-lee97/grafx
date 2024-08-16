import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def create_signal_buffer(
    method,
    num_buffers,
    input_signals,
):

    device = input_signals.device
    ndim = input_signals.ndim

    if method == "one-by-one":
        nones = [None for _ in range(num_buffers - len(input_signals))]
        signal_buffer = [x[None, :, :] for x in list(input_signals)] + nones

    else:
        match ndim:
            case 3:
                num_sources, channels, audio_len = input_signals.shape
                signal_shape = (num_buffers, channels, audio_len)
                signal_buffer = torch.empty(*signal_shape, device=device)
                signal_buffer[:num_sources] = input_signals

            case 4:
                num_batch, num_sources, channels, audio_len = input_signals.shape
                signal_shape = (num_batch, num_buffers, channels, audio_len)
                signal_buffer = torch.empty(*signal_shape, device=device)
                signal_buffer[:, :num_sources] = input_signals

    return signal_buffer


def read_single_tensor(x, access, dim=0, return_copy=False, postprocess=None):
    match access.method:
        case "slice":
            x = x.narrow(dim, access.idx[0], access.idx[1] - access.idx[0])
            if return_copy:
                x = x.clone()
        case "index":
            x = x.index_select(dim, access.idx)
        case _:
            raise Exception(
                f"The provided read method is not available: {access_method}."
            )
    if postprocess is not None:
        x = postprocess(x)
    return x


def read_tensor_or_tensor_dict(x, access, dim=0, return_copy=False, postprocess=None):
    if isinstance(x, torch.Tensor):
        return read_single_tensor(
            x, access, dim=dim, return_copy=return_copy, postprocess=postprocess
        )
    elif (
        isinstance(x, dict)
        or isinstance(x, nn.ParameterDict)
        or isinstance(x, nn.ModuleDict)
    ):
        return {
            k: read_tensor_or_tensor_dict(
                v, access, dim=dim, return_copy=return_copy, postprocess=postprocess
            )
            for k, v in x.items()
        }
    elif isinstance(x, list):
        return x[access.idx[0]]


def read_tensor_dict(x, access, dim=0, return_copy=False):
    return {
        k: read_tensor(v, access, dim=dim, return_copy=return_copy)
        for k, v in x.items()
    }


def inplace_write_tensor(method, x, y, access, dim=0):
    if method == "one-by-one":
        x[access.idx[0]] = y
    else:
        match access.method:
            case "slice":
                if dim == 0:
                    x[access.idx[0] : access.idx[1]] = y
                elif dim == 1:
                    x[:, access.idx[0] : access.idx[1]] = y
            case "index":
                if dim == 0:
                    x[access.idx] = y
                elif dim == 1:
                    x[:, access.idx] = y
            case _:
                raise Exception(
                    f"The provided inplace write method is not available: {access_method}."
                )


def aggregate_tensor(x: torch.Tensor, aggregation, dim=0):
    match aggregation.method:
        case "sum":
            return torch.sum(x, dim, keepdim=True)
        case "scatter":
            return scatter(x, aggregation.idx, dim=dim)
        case "none":
            return x
        case _:
            raise Exception(
                f"The provided aggregation method is not available: {aggregate_method}."
            )


def expand_single_tensor(x, expand=2, dim=0):
    x = x.unsqueeze(dim)
    sizes = list(x.shape)
    sizes[dim] *= expand
    x_expand = x.expand(*sizes).contiguous()
    return x_expand


def expand_tensor_or_tensor_dict(x, expand=2, dim=0):
    x_expand = {}
    if isinstance(x, torch.Tensor):
        return expand_single_tensor(x, expand=expand, dim=dim)
    elif (
        isinstance(x, dict)
        or isinstance(x, nn.ParameterDict)
        or isinstance(x, nn.ModuleDict)
    ):
        return {
            k: expand_tensor_or_tensor_dict(v, expand=expand, dim=dim)
            for k, v in x.items()
        }


def flatten_batch_and_node(x):
    shape = x.shape
    return x.reshape(-1, *shape[2:])
