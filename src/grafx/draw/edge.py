import numpy as np

from grafx.draw.bezier import Bezier


def draw_edge(ax, G, edge, vertical, linewidth=0.6):
    """
    Draw an edge between two nodes in a graph.

    Args:
        ax (:python:`matplotlib.axes.Axes`): 
            Pre-existing axes for the plot.
        G (:class:`~grafx.data.graph.GRAFX`): 
            A full graph that will be drawn.
        edge (:python:`Tuple[int, int, dict]`): 
            A tuple representing the edge, containing the source ID, destination ID, and edge data.
        linewidth (:python:`float`, *optional*): 
            The line width of the edge
            (default: :python:`0.6`).

    Returns:
        :python:`None`
    """
    source_id, dest_id, e = edge
    outlet, inlet = e["outlet"], e["inlet"]
    p_from = G.nodes[source_id]["meta"]["out_points"][outlet]
    p_to = G.nodes[dest_id]["meta"]["in_points"][inlet]
    add_edge_curve(ax, p_from, p_to, vertical, linewidth=linewidth)


def add_edge_curve(ax, p_from, p_to, vertical=False, linewidth=0.6, eps=0.02):
    if p_from[1] == p_to[1]:
        ax.plot(
            [p_from[0], p_to[0]], [p_from[1], p_to[1]], c="k", zorder=-1, linewidth=0.7
        )
    else:
        if vertical:
            mid_y = (p_to[1] + p_from[1]) / 2
            curve_nodes = np.asfortranarray(
                [
                    [p_from[0], p_from[0], p_to[0], p_to[0]],
                    [p_from[1] - eps, mid_y, mid_y, p_to[1] + eps],
                ]
            )
        else:
            mid_x = (p_to[0] + p_from[0]) / 2
            curve_nodes = np.asfortranarray(
                [
                    [p_from[0] + eps, mid_x, mid_x, p_to[0] - eps],
                    [p_from[1], p_from[1], p_to[1], p_to[1]],
                ]
            )
        curve = Bezier.Curve(np.linspace(0, 1, 101), curve_nodes.T)
        ax.plot(curve[:, 0], curve[:, 1], color="k", zorder=-1, linewidth=0.7)
