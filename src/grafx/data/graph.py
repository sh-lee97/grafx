import warnings
from typing import Dict, List, Union

import networkx as nx
import numpy as np
import torch

SINGLE_PARAMETER_TYPE = Union[int, float, np.ndarray, torch.Tensor]
PARAMETER_TYPE = Union[SINGLE_PARAMETER_TYPE, Dict[str, SINGLE_PARAMETER_TYPE]]


class GRAFX(nx.MultiDiGraph):
    r"""
    A base class for audio processing graph.
    It can be used for creating and modifying a graph.
    It inherits :python:`MultiDiGraph` class from :python:`networkx`.

    Args:
        config (:class:`~grafx.data.configs.NodeConfigs`, *optional*): 
            Node type configurations 
            (default: :python:`None`).
        invalid_op (:python:`str`, *optional*): 
            Behavior when an invalid operation is performed ("error", "warn", "mute") 
            (default: :python:`"error"`).

    Attributes:
        counter (:python:`Union[List[int], int]`): 
            A counter (for each graph, if it is a batched graph) for the number of nodes.
        consecutive_ids (:python:`bool`): 
            Indicates if node IDs are consecutive. This is useful when converting the graph to a tensor with the preserved order.
        batch (:python:`bool`): 
            Indicates if the graph is a single large disconnected graph created by batching multiple graphs.
        config (:class:`~grafx.data.configs.NodeConfigs`): 
            Node type configurations.
        config_hash (:python:`int`): 
            Hash value of the configuration.
        invalid_op (:python:`str`): 
            Behavior when an invalid operation is performed.
        rendering_order_method (:python:`str`): 
            Method used for determining the rendering order.
            Set to :python:`None` unless running a :func:`~grafx.render.order.graph.return_render_ordered_graph`.
        type_sequence (:python:`list`): 
            Node type sequence for the output audio rendering.
            Set to :python:`None` unless running a :func:`~grafx.render.order.graph.return_render_ordered_graph`.
    """

    def __init__(self, config=None, invalid_op="error"):
        if not invalid_op in ["error", "warn", "mute"]:
            raise Exception(f"Incorrect invalid_op is given: {invalid_op}.")

        super().__init__()

        self.graph = dict(
            counter=0,
            consecutive_ids=True,
            batch=False,
            config=config,
            config_hash=hash(config),
            invalid_op=invalid_op,
            rendering_order_method=None,
            type_sequence=None,
        )

    def __str__(self):
        num_nodes = self.number_of_nodes()
        num_edges = self.number_of_edges()
        string = f"GRAFX with {num_nodes} nodes & {num_edges} edges"
        for i, data in self.nodes(data=True):
            string += "\n"
            node_type = data["node_type"]
            string += f"  [{i}] {node_type}"
            out_edges = self.out_edges([i], data=True)
            num_out_edges = len(out_edges)
            if num_out_edges == 1:
                e = list(out_edges)[0]
                _, to, config = e
                outlet, inlet = config.values()
                if outlet != "main":
                    string += f" <{outlet}>"
                string += " -> "
                if inlet != "main":
                    string += f"<{inlet}> "
                string += f'[{to}] {self.nodes[to]["node_type"]}'
            elif num_out_edges > 1:
                string += "\n"
                string_es = []
                for e in out_edges:
                    _, to, config = e
                    outlet, inlet = config.values()
                    string_e = "    "
                    if outlet != "main":
                        string_e += f"<{outlet}>"
                    string_e += " -> "
                    if inlet != "main":
                        string_e += f"<{inlet}> "
                    string_e += f'[{to}] {self.nodes[to]["node_type"]}'
                    string_es.append(string_e)
                string += "\n".join(string_es)
        return string

    def add(
        self,
        node_type,
        parameters=None,
        name=None,
    ):
        r"""
        Adds a new node to the graph.

        Args:
            node_type (:python:`str`): The type of the node to be added.
            parameters (:python:`PARAMETER_TYPE`, *optional*): Parameters for the node (default: :python:`None`).
            name (:python:`str`, *optional*): Name of the node (default: :python:`None`).

        Returns:
            :python:`int`: The ID of the newly added node.
        """

        if self.graph["config"] is not None:
            node_types = self.graph["config"].node_types
            if not node_type in node_types:
                self.raise_warning(
                    f"Invalid node_type: {node_type}, this graph only allows {node_types}."
                )
                return

        node_id = self.graph["counter"]
        assert node_id not in self.nodes()
        self.add_node(node_id, node_type=node_type, parameters=parameters, name=name)
        self.graph["counter"] += 1
        return node_id

    def remove(self, node_id):
        r"""
        Removes a node from the graph and returns its connected edges.

        Args:
            node_id (:python:`int`): The ID of the node to be removed.

        Returns:
            :python:`Tuple[list, list]`: Incoming edges and outgoing edges of the removed node.
        """

        incoming_edges = list(self.in_edges(node_id, data=True))
        outgoing_edges = list(self.out_edges(node_id, data=True))
        self.remove_node(node_id)
        self.graph["consecutive_ids"] = False
        return incoming_edges, outgoing_edges

    def connect(
        self, source_id, dest_id, outlet="main", inlet="main"
    ):
        r"""
        Connects two nodes in the graph.

        Args:
            source_id (:python:`int`): The ID of the source node.
            dest_id (:python:`int`): The ID of the destination node.
            outlet (:python:`str`, *optional*): The outlet of the source node (default: :python:`"main"`).
            inlet (:python:`str`, *optional*): The inlet of the destination node (default: :python:`"main"`).

        Returns:
            :python:`None`
        """

        if self.has_edge(source_id, dest_id):
            potential_duplicates = self.get_edge_data(source_id, dest_id)
            for candidate in potential_duplicates.values():
                if candidate["outlet"] == outlet and candidate["inlet"] == inlet:
                    self.raise_warning(
                        f"{source_id} <{outlet}> -> {dest_id} <{inlet}>: existing edge."
                    )

        if source_id == dest_id:
            self.raise_warning(f"no self edge is allowed!")

        source_node_type = self.nodes[source_id]["node_type"]
        if self.graph["config"] is not None:
            outlets = self.graph["config"].node_type_dict[source_node_type]["outlets"]
            if not outlet in outlets:
                self.raise_warning(
                    f"Provided outlet: '{outlet}', while {source_node_type} only accepts {outlets}."
                )
                return

        dest_node_type = self.nodes[dest_id]["node_type"]
        if self.graph["config"] is not None:
            inlets = self.graph["config"].node_type_dict[dest_node_type]["inlets"]
            if not inlet in inlets:
                self.raise_warning(
                    f"Provided inlet: '{inlet}', while {dest_node_type} only accepts {inlets}."
                )
                return

        self.add_edge(source_id, dest_id, outlet=outlet, inlet=inlet)

    def add_serial_chain(self, node_list):
        r"""
        Adds a serial chain of nodes.

        Args:
            node_list (:python:`List[Union[str, dict]]`): 
                A list of nodes, each given as a type 
                or a dictionary that forms keyword arguments for the :func:`~grafx.data.graph.GRAFX.add` method.

        Returns:
            :python:`Tuple[int, int]`: The IDs of the first and last nodes in the chain.
        """

        for i, node_data in enumerate(node_list):
            if type(node_data) == str:
                node_id = self.add(node_data)
            else:
                self.add(**node_data)
            if i != 0:
                self.connect(node_id - 1, node_id)
            if i == 0:
                first_id = node_id
            if i == len(node_list) - 1:
                last_id = node_id
        return first_id, last_id

    def raise_warning(self, raisestring):
        match self.graph["invalid_op"]:
            case "error":
                raise Exception(raisestring)
            case "warn":
                warnings.warn("Following operation is invalid: " + raisestring)
            case "mute":
                return
            case _:
                assert False

    @property
    def counter(self):
        return self.graph["counter"]

    @counter.setter
    def counter(self, val):
        assert isinstance(val, int)
        self.graph["counter"] = val

    @property
    def consecutive_ids(self):
        return self.graph["consecutive_ids"]

    @consecutive_ids.setter
    def consecutive_ids(self, val):
        assert isinstance(val, bool)
        self.graph["consecutive_ids"] = val

    @property
    def batch(self):
        return self.graph["batch"]

    @batch.setter
    def batch(self, val):
        assert isinstance(val, bool)
        self.graph["batch"] = val

    @property
    def config(self):
        return self.graph["config"]

    @config.setter
    def config(self, val):
        raise Exception("config can be setted after the initialization.")

    @property
    def config_hash(self):
        return self.graph["config_hash"]

    @config_hash.setter
    def config_hash(self, val):
        raise Exception("config_hash cannot be setted directly.")

    @property
    def invalid_op(self):
        return self.graph["invalid_op"]

    @invalid_op.setter
    def invalid_op(self, val):
        assert isinstance(val, str)
        self.graph["invalid_op"] = val

    @property
    def rendering_order_method(self):
        return self.graph["rendering_order_method"]

    @rendering_order_method.setter
    def rendering_order_method(self, val):
        assert isinstance(val, str)
        self.graph["rendering_order_method"] = val

    @property
    def type_sequence(self):
        return self.graph["type_sequence"]

    @type_sequence.setter
    def type_sequence(self, val):
        assert isinstance(val, list) or isinstance(val, torch.LongTensor)
        self.graph["type_sequence"] = val

