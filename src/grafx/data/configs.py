from typing import Union

IN = {"inlets": [], "outlets": ["main"]}
OUT = {"inlets": ["main"], "outlets": []}
DEFAULT = {"inlets": ["main"], "outlets": ["main"]}
UTILITY_TYPES = ["in", "out", "mix"]
UTILITY_DICT = {"in": IN, "out": OUT, "mix": DEFAULT}


class NodeConfigs:
    """
    An object that stores configurations of node types and handles various utility tasks, 
    e.g., converting each node type :python:`str` to :python:`int`, or vice versa. 
    By default, the utility types, :python:`"in"`, :python:`"out"`, and :python:`"mix"`, 
    are automatically included.

    Args:
        config (:python:`Union[list, dict]`): The configuration data for the nodes.
            If a :python:`list`, it is assumed to contain node types.
            If a :python:`dict`, it should map node types to their specific configurations.

    Attributes:
        node_type_dict (:python:`Dict[str, dict]`): The full configuration for the given node types.
        node_types (:python:`list` of :python:`str`): List of node types.
        node_type_to_index (:python:`Dict[str, int]`): Mapping from node types to their indices.
        num_node_types (:python:`int`): The total number of node types.
        num_inlets (:python:`Dict[str, int]`): Number of inlets for each node type.
        num_outlets (:python:`Dict[str, int]`): Number of outlets for each node type.
        siso_only (:python:`bool`): Indicates if we only have single-input single-output (SISO) systems.
        max_num_inlets (:python:`int` *when* :python:`siso_only=False`): Maximum number of inlets.
        max_num_outlets (:python:`int` *when* :python:`siso_only=False`): Maximum number of outlets.
        inlet_to_index (:python:`Dict[str, Dict[str, int]]` *when* :python:`siso_only=False`): Nested :python:`dict` of type-to-inlet-to-index.
        outlet_to_index (:python:`Dict[str, Dict[str, int]]` *when* :python:`siso_only=False`): Nested :python:`dict` of type-to-outlet-to-index.
    """

    def __init__(self, config):
        if isinstance(config, list):
            node_type_list = UTILITY_TYPES + config
            self.unpack_list(node_type_list)
        elif isinstance(config, dict):
            node_type_dict = {**UTILITY_DICT, **config}
            self.unpack_dict(node_type_dict)
        else:
            raise ValueError("Invalid type for config.")

    def __str__(self):
        """
        Returns a string representation of the node configurations.
        """
        strings = []
        string = f"NodeConfigs with {self.num_node_types} node types (siso_only={self.siso_only})"
        strings.append(string)
        for node_type, config in self.node_type_dict.items():
            index = self.node_type_to_index[node_type]
            inlets = config["inlets"]
            if len(inlets) == 0:
                inlets = "None"
            else:
                inlets = ", ".join(inlets)
                inlets = f"<{inlets}>"
            outlets = config["outlets"]
            if len(outlets) == 0:
                outlets = "None"
            else:
                outlets = ", ".join(outlets)
                outlets = f"<{outlets}>"
            string = f"  ({index}) {node_type}: {inlets} -> {outlets}"
            strings.append(string)
        return "\n".join(strings)

    def get_default_config(self, node_type):
        match node_type:
            case "in":
                return IN
            case "out":
                return OUT
            case _:
                return DEFAULT

    def unpack_list(self, node_type_list):
        node_type_dict = {k: self.get_default_config(k) for k in node_type_list}
        self.unpack_dict(node_type_dict)

    def unpack_dict(self, node_type_dict):
        self.num_node_types = len(node_type_dict)
        self.node_types = list(node_type_dict.keys())
        self.node_type_to_index = {
            self.node_types[i]: i for i in range(self.num_node_types)
        }

        inlet_to_index, outlet_to_index = {}, {}
        num_inlets, num_outlets = {}, {}
        max_num_inlets, max_num_outlets = 1, 1

        for node_type, config in node_type_dict.items():
            inlets = config["inlets"]
            num_inlets[node_type] = len(inlets)
            inlet_to_index[node_type] = {}
            for i, inlet in enumerate(inlets):
                inlet_to_index[node_type][inlet] = i
            max_num_inlets = max(max_num_inlets, len(inlets))

            outlets = config["outlets"]
            num_outlets[node_type] = len(outlets)
            outlet_to_index[node_type] = {}
            for i, outlet in enumerate(outlets):
                outlet_to_index[node_type][outlet] = i
            max_num_outlets = max(max_num_outlets, len(outlets))

        self.num_inlets = num_inlets
        self.num_outlets = num_outlets
        self.siso_only = (max_num_inlets == 1) and (max_num_outlets == 1)

        if not self.siso_only:
            self.max_num_inlets = max_num_inlets
            self.max_num_outlets = max_num_outlets
            self.inlet_to_index = inlet_to_index
            self.outlet_to_index = outlet_to_index

        self.node_type_dict = node_type_dict

    def __getitem__(self, node_type):
        """
        Allows dictionary-like access to node configurations.
        """
        return self.node_type_dict[node_type]
