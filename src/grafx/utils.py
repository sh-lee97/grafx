import torch
import torch.nn as nn

from grafx.data.graph import GRAFX
from grafx.data.tensor import GRAFXTensor


def get_node_ids_from_type(G: GRAFX, node_type: str):
    """
    Retrieves the node IDs for of a specific type in the graph.

    Args:
        G (:class:`~grafx.data.graph.GRAFX`):
            The target graph.
        node_type (:python:`str`):
            The node type to retrieve.

    Returns:
        :python:`List[int]`: A list of node IDs that match the given type.
    """
    node_ids = []
    for node_id, data in G.nodes(data=True):
        if data["node_type"] == node_type:
            node_ids.append(node_id)
    return node_ids


def count_nodes_per_type(G: GRAFX, types_to_count: list = None):
    """
    Counts the number of nodes for each specified type in the graph.

    Args:
        G (:class:`~grafx.data.graph.GRAFX`):
            The target graph.
        types_to_count (:python:`list`, *optional*):
            A list of node types to count. If :python:`None`,
            counts all types present in the graph (default: :python:`None`).

    Returns:
        :python:`Dict[str, int]`:
            A dictionary with node types as keys and counts as values.
    """
    if types_to_count is not None:
        num_nodes_per_type = {k: 0 for k in types_to_count}
    elif G.config is not None:
        num_nodes_per_type = {k: 0 for k in G.config.node_types}
    else:
        num_nodes_per_type = {}

    for _, data in G.nodes(data=True):
        node_type = data["node_type"]
        if types_to_count is not None:
            if node_type in types_to_count:
                num_nodes_per_type[node_type] += 1
        else:
            num_nodes_per_type[node_type] = 1 + num_nodes_per_type.get(node_type, 0)
    return num_nodes_per_type


def create_empty_parameters(processors, G, std=1e-2):
    """
    Creates and initializes parameter tensors in a nested dictionary format from a given graph and processors.
    The tensors values are sampled from a normal distribution $\mathcal{N}(0, \sigma^2)$,
    where the standard deviation $\sigma$ is given by the :python:`std` argument.

    Args:
        processors (:python:`Mappings`):
            A dictionary of processors, either :python:`dict` or :python:`nn.ModuleDict`,
            where keys are node types and values are processors.
        G (:class:`~grafx.data.graph.GRAFX`):
            The graph containing nodes whose parameters are to be initialized.
        std (:python:`float`, *optional*):
            Standard deviation for the parameter initialization (default: :python:`0.01`).

    Returns:
        :python:`nn.ModuleDict`:
            A module dictionary with initialized parameters for each node type in the graph.
    """
    parameter_dict = {}
    num_nodes_per_type = count_nodes_per_type(G, processors)
    for processor_type in processors:
        num_nodes = num_nodes_per_type[processor_type]
        parameter_shapes = processors[processor_type].parameter_size()
        parameter_dict[processor_type] = create_empty_parameters_from_shape_dict(
            parameter_shapes=parameter_shapes, num_nodes=num_nodes, std=std
        )
    return nn.ParameterDict(parameter_dict)


def create_empty_parameters_from_shape_dict(
    parameter_shapes,
    num_nodes,
    std=1e-2,
    root=True,
    device="cpu",
):

    def int_to_tuple(x):
        if isinstance(x, int):
            return (x,)
        elif isinstance(x, tuple):
            return x
        else:
            raise Exception(f"Parameter shape with type {type(x)} is not suppoerted")

    match parameter_shapes:
        # non-leaf node
        case dict():
            parameter = {
                k: create_empty_parameters_from_shape_dict(
                    v, num_nodes, std, root=False, device=device
                )
                for k, v in parameter_shapes.items()
            }
            parameter = nn.ParameterDict(parameter)
        # leaf node
        case int() | tuple():
            parameter = std * torch.randn(
                num_nodes, *int_to_tuple(parameter_shapes), device=device
            )
            if root:
                parameter = {"parameter": parameter}
                nn.ParameterDict(parameter)
            else:
                parameter = nn.Parameter(parameter)
        case _:
            raise Exception(
                f"Parameter shapes with type {type(parameter_shapes)} is not suppoerted"
            )

    return parameter


def permute_grafx_tensor(
    G_t,
    node_id,
    node_attrs=["node_types", "rendering_orders"],
    id_attrs=["edge_indices"],
):
    """
    Permutes the node and edge attributes of a given :class:`~grafx.data.tensor.GRAFXTensor` according to a given node ordering.
    Attributes that are not provided in the :python:`node_attrs` or :python:`id_attrs` are left unchanged.

    Args:
        G_t (:class:`~grafx.data.tensor.GRAFXTensor`):
            The graph tensor to permute.
        node_id (:python:`LongTensor`):
            The permutation index given by the node IDs.
        node_attrs (:python:`List[str]`, *optional*):
            List of node attributes to permute
            (default: :python:`["node_types", "rendering_orders"]`).
        id_attrs (:python:`List[str]`, *optional*): List of attributes that contain node IDs
            (default: :python:`["edge_indices"]`).

    Returns:
        :class:`~grafx.data.tensor.GRAFXTensor`: The permuted graph tensor.
    """

    node_feature_map = torch.empty_like(node_id)
    node_feature_map[node_id] = torch.arange(len(node_id))

    new_dict = {}
    for k, v in G_t.__dict__.items():
        if v is None:
            new_dict[k] = None
        else:
            if k in node_attrs:
                new_dict[k] = v[node_feature_map]
            elif k in id_attrs:
                new_dict[k] = node_id[v]
            else:
                new_dict[k] = v

    return GRAFXTensor(**new_dict)


if __name__ == "__main__":
    shape_dict = {
        "init": (3, 3),
        "mid": {"init_gain": 1, "delta_filter": {"cutoff": (40, 30)}},
        "post": {
            "panning": (1,),
            "dynamic_range": (2, 2),
        },
    }
    param = create_empty_parameters_from_shape_dict(shape_dict, 10)
    print(param)
