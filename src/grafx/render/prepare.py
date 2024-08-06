from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.utils import sort_edge_index

TENSOR_IDX_TYPE = Union[Tuple[int], torch.LongTensor]


@dataclass
class _TensorAccessData:
    method: str
    idx: TENSOR_IDX_TYPE

    def __str__(self):
        return f"{self.method} with {self.idx.__str__()}"


@dataclass
class _AggregationData:
    method: str
    idx: Optional[TENSOR_IDX_TYPE] = None

    def __str__(self):
        match self.method:
            case "none" | "sum":
                return self.method
            case "scatter":
                return f"scatter with {self.idx.__str__()}"


@dataclass
class _SingleRenderData:
    node_type: str
    source_reads: List[_TensorAccessData]
    aggregations: List[_AggregationData]
    parameter_read: _TensorAccessData
    dest_write: _TensorAccessData

    def __str__(self):
        strings = []
        strings.append(f"- Node type: {self.node_type}")

        if len(self.source_reads) == 1:
            strings.append(f"- Source read: {self.source_reads[0].__str__()}")
        else:
            strings.append(f"- Source reads:")
            for source_read in self.source_reads:
                strings.append(f"  * {source_read.__str__()}")

        if len(self.aggregations) == 1:
            strings.append(f"- Aggregation: {self.aggregations[0].__str__()}")
        else:
            strings.append(f"- Aggregations:")
            for aggregation in self.aggregations:
                strings.append(f"  * {aggregation.__str__()}")

        strings.append(f"- Parameter read: {self.parameter_read.__str__()}")

        strings.append(f"- Dest write: {self.dest_write.__str__()}")
        return "\n".join(strings)


@dataclass
class RenderData:
    """
    Holds all data necessary for rendering an audio processing graph.

    Args:
        method (:python:`str`): The overall rendering method.
        num_nodes (:python:`int`): Total number of nodes in the graph.
        max_order (:python:`int`): The maximum order of processing required.
        siso_only (:python:`bool`): Indicates if the graph is strictly single-input and single-output.
        iter_list (:python:`List[_SingleRenderData]`): List of rendering data for each processing stage.
    """

    method: str
    num_nodes: int
    max_order: int
    siso_only: bool
    iter_list: List[_SingleRenderData]

    def __str__(self):
        strings = []
        strings.append(
            f"Rendering of {self.num_nodes} nodes with siso_only: {self.siso_only}."
        )
        for i, it in enumerate(self.iter_list):
            strings.append(f"Render #{i}\n" + it.__str__())
        return "\n\n".join(strings)


def prepare_render(G_t):
    """
    Computes the metadata, i.e., sequence of operations including the tensor reads, aggregations, processings, and writes,
    required for the graph rendering.

    Args:
        G_t (:class:`~grafx.data.tensor.GRAFXTensor`): 
            The graph to compute the metadata for rendering.

    Returns:
        :class:`~grafx.render.prepare.RenderData`:
            The metadata required for rendering the graph.
    """

    configs = G_t.config
    method = G_t.rendering_order_method
    siso_only = configs.siso_only
    type_sequence = G_t.type_sequence

    per_type_indices = create_per_type_indices(G_t.node_types)

    if siso_only:
        edge_indices = sort_edge_index(G_t.edge_indices, sort_by_row=False)
    else:
        edge_indices, edge_types = sort_edge_index(
            G_t.edge_indices, edge_attr=G_t.edge_types, sort_by_row=False
        )
        edge_types = edge_types.tolist()

        num_outlets = torch.tensor([configs.num_outlets[t] for t in configs.node_types])
        num_outlets = num_outlets[G_t.node_types].tolist()
        buffer_offsets = torch.cumsum(torch.tensor([0] + num_outlets[:-1]), 0).tolist()

    max_order = torch.max(G_t.rendering_orders)
    num_nodes = G_t.num_nodes

    edge_indices = edge_indices.T

    iter_list = []
    for i in range(max_order + 1):
        node_mask = G_t.rendering_orders == i
        node_idxs = torch.where(node_mask)[0]
        node_list = node_idxs.tolist()
        node_type = type_sequence[i]

        if siso_only:
            source_idx = []
            scatter_idx = []
            edges = get_incoming_edges(edge_indices, node_idxs).tolist()
            for source, dest in edges:
                scatter_idx.append(node_list.index(dest))
                source_idx.append(source)
            source_reads = [check_and_convert_arange(source_idx)]
            aggregations = [check_aggregate_method(scatter_idx, node_list)]

        else:
            num_inlets = configs.num_inlets[node_type]
            scatter_idxs = [[] for _ in range(num_inlets)]
            source_idxs = [[] for _ in range(num_inlets)]
            edges = get_incoming_edges(edge_indices, node_idxs).tolist()

            for source, dest in edges:
                outlet, inlet = edge_types[i]
                scatter_idxs[inlet].append(node_list.index(dest))
                source_idxs[inlet].append(buffer_offsets[source] + outlet)

            source_reads = [check_and_convert_arange(idx) for idx in source_idxs]
            aggregations = [
                check_aggregate_method(idx, node_list) for idx in scatter_idxs
            ]

        parameter_idx = per_type_indices[node_mask]
        parameter_read = check_and_convert_arange(parameter_idx)

        if siso_only:
            buffer_idx = node_list.copy()
        else:
            num_outlets = configs.num_outlets[node_type]
            buffer_idx = []
            for idx in node_list:
                offset = buffer_offsets[idx]
                buffer_idx += list(range(offset, offset + num_outlets))
            buffer_idx = torch.tensor(buffer_idx)

        dest_write = check_and_convert_arange(buffer_idx)

        single_iter_data = _SingleRenderData(
            node_type=node_type,
            aggregations=aggregations,
            source_reads=source_reads,
            parameter_read=parameter_read,
            dest_write=dest_write,
        )
        iter_list.append(single_iter_data)

    render_data = RenderData(
        method=method,
        num_nodes=num_nodes,
        max_order=max_order,
        siso_only=siso_only,
        iter_list=iter_list,
    )
    return render_data


def check_aggregate_method(scatter_idx, node_list):
    if len(scatter_idx) == 0:
        return _AggregationData(method="none")
    else:
        if not isinstance(scatter_idx, torch.Tensor):
            scatter_idx = torch.tensor(scatter_idx)
        if len(scatter_idx) == 1 and scatter_idx[0] == 0:
            return _AggregationData(method="none")
        elif (scatter_idx == 0).all():
            return _AggregationData(method="sum")
        elif (
            len(scatter_idx) != len(node_list)
            or scatter_idx[0] != 0
            or (scatter_idx.diff() != 1).any()
        ):
            return _AggregationData(method="scatter", idx=scatter_idx)
        else:
            return _AggregationData(method="none")


def check_and_convert_arange(idx):
    if len(idx) == 0:
        return _TensorAccessData(method="none", idx=idx)
    else:
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx)
        if (idx.diff() == 1).all():
            idx = (idx[0].item(), 1 + idx[-1].item())
            return _TensorAccessData(method="slice", idx=idx)
        else:
            return _TensorAccessData(method="index", idx=idx)


def get_incoming_edges(edge_indices, node_idxs):
    dests = edge_indices[:, 1]
    edge_masks = torch.any(dests[:, None] == node_idxs[None, :], -1)
    return edge_indices[edge_masks]


def create_per_type_indices(node_types):
    per_type_indices = torch.zeros_like(node_types)
    type_set = set(node_types.tolist())
    for T in type_set:
        node_mask = node_types == T
        num_nodes_T = torch.sum(node_mask.long())
        per_type_indices[node_mask] = torch.arange(num_nodes_T)
    return per_type_indices
