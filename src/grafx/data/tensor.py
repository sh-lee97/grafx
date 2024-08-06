from dataclasses import dataclass
from typing import Union

import torch

from grafx.data.configs import NodeConfigs


@dataclass
class GRAFXTensor:
    r"""
    A dataclass representing a tensor-based graph for audio processing.

    Args:
        node_types (:python:`LongTensor`): 
            Tensor of node types.
        edge_indices (:python:`LongTensor`): 
            Tensor of edge indices.
        counter (:python:`Union[int, LongTensor]`): 
            Counter for the number of nodes.
        batch (:python:`bool`): 
            Indicates if the tensor is part of a batch.
        config (:class:`~grafx.data.configs.NodeConfigs`): 
            Configuration for the nodes.
        config_hash (:python:`str`): 
            Hash of the configuration.
        invalid_op (:python:`str`): 
            Behavior when an invalid operation is performed.
        edge_types (:python:`Union[LongTensor, None]`, *optional*): 
            Tensor of edge types
            (default: :python:`None`).
        rendering_order_method (:python:`Union[str, None]`, *optional*): 
            Method for determining the rendering order
            (default: :python:`None`).
        rendering_orders (:python:`Union[LongTensor, None]`, *optional*): 
            Tensor of rendering orders 
            (default: :python:`None`).
        type_sequence (:python:`Union[LongTensor, None`], *optional*): 
            Tensor of type sequences 
            (default: :python:`None`).

    Attributes:
        num_nodes (:python:`int`): 
            The number of nodes in the graph.
        num_edges (:python:`int`): 
            The number of edges in the graph.
    """

    node_types: torch.LongTensor
    edge_indices: torch.LongTensor
    counter: int
    batch: bool
    config: NodeConfigs
    config_hash: str
    invalid_op: str

    edge_types: Union[torch.LongTensor, None] = None
    rendering_order_method: Union[str, None] = None
    rendering_orders: Union[torch.LongTensor, None] = None
    type_sequence: Union[torch.LongTensor, None] = None

    def __str__(self):
        """
        Returns a string representation of the :python:`GRAFXTensor` object, detailing its attributes.

        Returns:
            :python:`str`: A formatted string describing the attributes of the tensor.
        """

        strings = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                string = str(list(v.shape))
            else:
                string = repr(v)
            string = f"\n  {k}={string}"
            strings.append(string)
        string = ", ".join(strings)
        string = f"GRAFXTensor({string}\n)"
        return string

    @property
    def num_nodes(self):
        return len(self.node_types)

    @property
    def num_edges(self):
        return len(self.edge_indices)

    def to(self, device):
        """
        Moves all tensor attributes to the specified device.

        Args:
            device (:python:`torch.device`): The device to move the tensors to.

        Returns:
            :python:`None`
        """

        for k in self.__dict__:
            if isinstance(self.__dict__[k], torch.Tensor):
                self.__dict__[k] = self.__dict__[k].to(device)
