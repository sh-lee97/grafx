from typing import Union

import networkx as nx

from grafx.data.conversion import convert_to_tensor
from grafx.data.graph import GRAFX
from grafx.data.tensor import GRAFXTensor
from grafx.render.order.tensor import (
    compute_render_order_tensor,
    node_id_from_render_order,
    return_render_ordered_tensor,
)


def compute_render_order(G_any, method="beam", **kwargs):
    """
    Computes a rendering order for the graph (in either type).

    Args:
        G_any (:class:`~grafx.data.graph.GRAFX` or :class:`~grafx.data.tensor.GRAFXTensor`): 
            The graph to compute the rendering order.
            As the main computation is done with tensors, if a graph is provided, it will be internally converted to a 
            :class:`~grafx.data.tensor.GRAFXTensor` object.
        method (:python:`str`, *optional*): 
            The method to use for computing the rendering order.
            Allows :python:`"greedy"`, :python:`"beam"`, :python:`"fixed"`, and :python:`"one-by-one"`.
            For the comparison of these methods, see :ref:`this section <type-scheduling>`
            (default: :python:`"beam"`).
        **kwargs: 
            Additional arguments for some methods.
            If :python:`method == "beam"`, :python:`width` and :python:`depth` can be optionally passed as arguments 
            (default: :python:`1` and :python:`64`, respectively).
            If :python:`method == "fixed"`, a sequence of node types must be passed with key :python:`fixed_order`.

    Returns:
        :python:`Tuple[List[str], LongTensor]`: 
            A ndoe type sequence and a tensor that specifies the rendering order for each node.
    """

    if isinstance(G_any, GRAFX):
        G_t = convert_to_tensor(G_any)
        return compute_render_order_tensor(G_t, method, **kwargs)
    elif isinstance(G_any, GRAFXTensor):
        return compute_render_order_tensor(G_any, method, **kwargs)
    else:
        raise Exception(f"Invalid graph type: {type(G_any)}")

def reorder_for_fast_render(G_any, method="beam", **kwargs):
    r"""
    Computes a rendering order for the graph (in either type) and reorders the graph for faster rendering.
    The former is done with :func:`~grafx.render.order.graph.compute_render_order`.

    Args:
        G_any (:class:`~grafx.data.graph.GRAFX` or :class:`~grafx.data.tensor.GRAFXTensor`): 
            The graph to compute the rendering order and reorder.
        method (:python:`str`, *optional*): 
            The ordering method.
            (default: :python:`"beam"`).
        **kwargs: 
            Additional arguments for some methods.

    Returns:
        :class:`~grafx.data.graph.GRAFX` or :class:`~grafx.data.tensor.GRAFXTensor`:
            A reordered graph.
    """
    if isinstance(G_any, GRAFX):
        return return_render_ordered_graph(G_any, method, **kwargs)
    elif isinstance(G_any, GRAFXTensor):
        return return_render_ordered_tensor(G_any, method, **kwargs)
    else:
        raise Exception(f"Invalid input type: {type(G_any)}")



def return_render_ordered_graph(G: GRAFX, method, **kwargs):
    type_sequence, render_order = compute_render_order(G, method, **kwargs)
    for i, j in zip(G.nodes, render_order):
        G.nodes[i]["rendering_order"] = j.item()
    node_id = node_id_from_render_order(render_order).tolist()
    mapping = {k: v for k, v in zip(range(len(node_id)), node_id)}
    G = nx.relabel_nodes(G, mapping=mapping)
    G = get_sorted_graph(G)
    G.type_sequence = [G.config.node_types[t] for t in type_sequence]
    G.rendering_order_method = method
    return G



def get_sorted_graph(G):
    H = GRAFX()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(sorted(G.edges(data=True)))
    H.graph = G.graph.copy()
    return H
