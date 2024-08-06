import networkx as nx
import torch

from grafx.data.graph import GRAFX
from grafx.data.tensor import GRAFXTensor


def convert_to_tensor(G):
    r"""
    Convert a graph to a collection of tensors.

    Args:
        G (:class:`~grafx.data.graph.GRAFX`): The graph to convert.

    Returns:
        :class:`~grafx.data.tensor.GRAFXTensor`: A tensor representation of the given graph.
    """
    node_configs = G.config
    if not G.consecutive_ids:
        G = _relabel_nodes_to_consequtive_ids(G)

    nodes_with_data, edges_with_data = sorted(G.nodes(data=True)), sorted(
        G.edges(data=True)
    )

    node_types = []
    for _, data in nodes_with_data:
        node_type = data["node_type"]
        node_type_id = node_configs.node_type_to_index[node_type]
        node_types.append(node_type_id)
    node_types = torch.tensor(node_types, dtype=torch.long)

    if G.rendering_order_method is not None:
        rendering_orders = []
        for _, data in nodes_with_data:
            rendering_order = data.get("rendering_order", -1)
            rendering_orders.append(rendering_order)
        rendering_orders = torch.tensor(rendering_orders, dtype=torch.long)
    else:
        rendering_orders = None


    source_ids, dest_ids = [], []
    for source_id, dest_id, _ in edges_with_data:
        source_ids.append(source_id)
        dest_ids.append(dest_id)
    source_ids = torch.tensor(source_ids)
    dest_ids = torch.tensor(dest_ids)
    edge_indices = torch.stack([source_ids, dest_ids])

    if node_configs.siso_only:
        edge_types = None
    else:
        edge_types = []
        for source_id, dest_id, data in edges_with_data:
            outlet, inlet = data["outlet"], data["inlet"]
            source_type = G.nodes[source_id]["node_type"]
            dest_type = G.nodes[dest_id]["node_type"]
            outlet_id = node_configs.outlet_to_index[source_type][outlet]
            inlet_id = node_configs.inlet_to_index[dest_type][inlet]
            edge_types.append([outlet_id, inlet_id])
        edge_types = torch.tensor(edge_types)

    return GRAFXTensor(
        node_types=node_types,
        edge_indices=edge_indices,
        edge_types=edge_types,
        rendering_order_method=G.rendering_order_method,
        rendering_orders=rendering_orders,
        type_sequence=G.type_sequence,
        counter=G.counter,
        batch=G.batch,
        config=G.config,
        config_hash=G.config_hash,
        invalid_op=G.invalid_op,
    )


def _relabel_nodes_to_consequtive_ids(G):
    node_ids = list(G.nodes())
    num_nodes = G.number_of_nodes()
    relabel_mapping = {node_ids[i]: i for i in range(num_nodes)}
    G = nx.relabel_nodes(G, relabel_mapping, copy=True)
    G.coutiguous_id = True
    return G
