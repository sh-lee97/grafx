import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FALLBACK_CMAP = plt.get_cmap("jet")
DEFAULT_COLORS = [
    "#E6F9AF",
    "#F2E3BC",
    "#FFCC99",
    "#BAC8D3",
    "#E1D5E7",
    "#EAE8FF",
    "#EEEEEE",
    "#B3BFB8",
    "#FFE3E0",
    "#ECE2D0",
    "#FFCBDD",
    "#F4F9E9",
    "#FFFF88",
    "#A1E5B7",
    "#EEC584",
    "#FEFEE3",
    "#D4E09B",
    "#CCE5FF",
    "#CDEB8B",
    "#DAFFED",
    "#9BF3F0",
    "#EAE1DF",
    "#FFCCCC",
    "#D1FFD7",
    "#EFFFFA",
    "#C3BEF7",
]


class NodeColorHandler:
    r"""
    A class that handles the color mapping for each node type in the graph.

    Args:
        facecolor_map (:python:`Dict[str, COLORTYPE]` or :python:`None`, *optional*):
            A dictionary that maps each node type to a face color.
            If given, the other arguments are ignored.
            If :python:`None`, the other arguments must be provided, 
            as they are used to generate the mapping
            (default: :python:`None`).
        node_types (:python:`List[str]` or :python:`None`, *optional*):
            A list of node types that can be used in the graph
            (default: :python:`None`).
        colors (:python:`List[COLORTYPE]` or :python:`None`, *optional*):
            A list of colors (in any format that :python:`matplotlib` recognizes).
            This module will assign each node type a color based on its initial letter,
            and if this fails due to the lack of colors, it will assign a random color.
            (default: :python:`None`).

    """
    def __init__(self, facecolor_map=None, node_types=None, colors=None):
        if facecolor_map is not None:
            self.facecolor_map = facecolor_map
        else:
            colors = DEFAULT_COLORS if colors is None else colors

            rng = np.random.RandomState(0)
            self.facecolor_map = {}

            idxs = list(range(len(colors)))
            for node_type in node_types:
                if node_type in ["in", "out"]:
                    continue
                idx = ord(node_type[0]) - 97
                if len(idxs) > 0:
                    while True:
                        if idx in idxs:
                            idxs.remove(idx)
                            color = colors[idx]
                            self.facecolor_map[node_type] = color
                            break
                        else:
                            idx = (idx + 1) % len(colors)
                else:
                    color = DEFAULT_FALLBACK_CMAP(rng.uniform())
                    self.facecolor_map[node_type] = color

    def get_facecolor(self, node_type):
        match node_type:
            case "in" | "out":
                return "w"
            case _:
                return self.facecolor_map[node_type]

    def get_edgecolor(self, node_type):
        match node_type:
            case "in":
                return "b"
            case "out":
                return "r"
            case _:
                return "k"

    def get_colors(self, node_type):
        r"""
        Retrieves the face and edge color for the given node type.

        Args: 
            node_type (:python:`str`): The type of the node.

        Returns:
            :python:`Dict[str, COLORTYPE]`: A dictionary with the face and edge color.
        """
        return {
            "facecolor": self.get_facecolor(node_type),
            "edgecolor": self.get_edgecolor(node_type),
        }
