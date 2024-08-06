import matplotlib.pyplot as plt

from grafx.draw.edge import draw_edge
from grafx.draw.node import draw_node
from grafx.draw.position import compute_node_position
from grafx.draw.style import NodeColorHandler


def draw_grafx(
    G,
    vertical=False,
    compute_node_position_fn=compute_node_position,
    draw_node_fn=draw_node,
    draw_edge_fn=draw_edge,
    colors=None,
    **kwargs,
):
    r"""
    Draw an input :class:`~grafx.data.graph.GRAFX` object.
    It first computes the node positions, then draws each node and edge.
    These functions, 
    :func:`~grafx.draw.position.compute_node_position`,  
    :func:`~grafx.draw.node.draw_node`, and
    :func:`~grafx.draw.edge.draw_edge`, respectively,
    can be customized by passing custom functions or modules to the arguments.

    Args:
        G (:class:`~grafx.data.graph.GRAFX`): 
            An input graph to draw.
        vertical (:python:`bool`, *optional*):
            If :python:`True`, the nodes are organized from top to bottom.
            If :python:`False`, the nodes are organized from left to right 
            (default: :python:`False`).
        compute_node_position_fn (:python:`Callable`, *optional*):
            Any function or module that computes and stores node positions into
            :python:`G` as a key :python:`x0` and :python:`y0`.
            (default: :func:`~grafx.draw.position.compute_node_position`).
        draw_node_fn (:func:`Callable`, *optional*):
            Any function or module that draws each node of :python:`G`
            (default: :func:`~grafx.draw.node.draw_node`).
        draw_edge_fn (:python:`Callable`, *optional*):
            Any function or module that draws each edge of :python:`G`
            (default: :func:`~grafx.draw.edge.draw_edge`).
        colors (:python:`List`, :python:`Dict`, or :python:`None`, *optional*):
            Collection of face colors for each node type.
            If a :python:`List` is given, each type's color will be assigned based on its initial letter.
            If a :python:`Dict` is given, the color will be exactly assigned based on that dictionary.
            :python:`None` will use the default color scheme. 
            All of these are handled with :class:`~grafx.draw.style.NodeColorHandler`
            (default: :python:`None`).
        **kwargs (*optional*):
            All additional keyword arguments passed to
            :python:`compute_node_position_fn`, :python:`draw_node_fn`,
            and :python:`draw_edge_fn`.
            Each keyword must start with
            :python:`"position_"`, :python:`"node_"`, or :python:`"edge_"`
            so that it can passed to one of the three appropriately.
            For example, "node_size" will be passed to :python:`draw_node_fn`
            as a key :python:`"size"`.

    Returns:
        :python:`Tuple[Figure, Axes]`: 
            A :python:`matplotlib` plot that the input graph is visualized on.
    """

    node_kwargs, edge_kwargs, position_kwargs = {}, {}, {}
    for k, v in kwargs.items():
        k_split = k.split("_", maxsplit=1)
        if len(k_split) != 2:
            raise Exception(f"Wrong argument: {k}")
        k1, k2 = k_split
        match k1:
            case "node":
                node_kwargs[k2] = v
            case "edge":
                edge_kwargs[k2] = v
            case "position":
                position_kwargs[k2] = v
            case _:
                raise Exception(f"Wrong prefix: {k1}")


    if isinstance(colors, dict):
        color_config = NodeColorHandler(facecolor_map=colors)
    else:
        node_types = G.config.node_types
        color_config = NodeColorHandler(node_types=node_types, colors=colors)

    G = G.copy()

    compute_node_position_fn(G, **position_kwargs)
    if vertical:
        for node_id in list(G.nodes):
            x0, y0 = G.nodes[node_id]["x0"], G.nodes[node_id]["y0"] 
            G.nodes[node_id]["x0"], G.nodes[node_id]["y0"] = y0, x0

    fig, ax = plt.subplots()

    for node in G.nodes(data=True):
        draw_node_fn(ax, G, node, color_config, vertical, **node_kwargs)

    for edge in G.edges(data=True):
        draw_edge_fn(ax, G, edge, vertical, **edge_kwargs)

    postprocess_figure(fig, ax)
    return fig, ax


def postprocess_figure(fig, ax, xscale=0.3, yscale=0.3):
    ax.axis("off")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xlen, ylen = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fig.set_size_inches(xlen * xscale, ylen * yscale)
    ax.invert_yaxis()
