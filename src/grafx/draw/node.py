from matplotlib.patches import Rectangle


def plot_points(ax, p0, off_x, off_y):
    # Dummy points plotter that allows matplotlib to calculate the xlim/ylim correctly
    ax.plot(p0[0], p0[1], alpha=0)
    ax.plot(p0[0] + off_x, p0[1], alpha=0)
    ax.plot(p0[0], p0[1] + off_y, alpha=0)
    ax.plot(p0[0] + off_x, p0[1] + off_y, alpha=0)


def draw_node(
    ax,
    G,
    node,
    color_config,
    vertical=False,
    inside="node_type",
    above=None,
    size=(0.5, 0.5),
    linewidth=0.6,
    inside_fontsize=5.6,
    above_fontsize=3.0,
):
    """
    Draws a node as an rectangle on the provided axes.

    Args:
        ax (:python:`matplotlib.pyplot.Axes`):
            Pre-existing axes for the plot.
        G (:class:`~grafx.data.graph.GRAFX`):
            A full graph that will be drawn.
        node (:python:`Tuple[int, dict]`):
            Node attributes of the target node to be drawn.
            It is an individual item of the list returned by :python:`GRAFX.nodes(data=True)`.
            It must contain :python:`inside` and :python:`above`
        color_config (:python:`List` or :python:`None`, *optional*):
            (default: :python:`None`).
        inside (:python:`str`, *optional*):
            Key of a node attribute that will be displayed inside the node
            :python:`G` as a key :python:`x0` and :python:`y0`.
            (default: :python:`"node_type"`).
        above (:python:`str` or :python:`None`, *optional*):
            Key of a node attribute that will be displayed above the node
            Any function or module that draws each node of :python:`G`
            (default: :python:`None`).
        size (:python:`Tuple[float]`, *optional*):
            A size of node shown as a rectangle
            (default: :python:`draw_edge`).
        linewidth (:python:`float`, *optional*):
            Thickness of the border of the rectangle
            (default: :python:`draw_edge`).
        inside_fontsize (:python:`float`, *optional*):
            Size of the text inside of the rectangle
            (default: :python:`5.6`).
        above_fontsize (:python:`float`, *optional*):
            Size of the text above of the rectangle
            (default: :python:`3.0`).

    Returns:
        :python:`None`
    """
    node_id, node = node

    p0 = (node["x0"], node["y0"])
    node_type = node["node_type"]
    config = G.config[node_type]

    plot_points(ax, p0, size[0], size[1])
    colors = color_config.get_colors(node_type)
    ax.add_patch(Rectangle(p0, size[0], size[1], linewidth=linewidth, **colors))

    header_y = p0[1] + size[0] / 2

    allowed_texts = ["node_id"] + list(node.keys())
    if not inside in allowed_texts:
        raise Exception(
            f"Provided inside: '{inside}', but only {allowed_texts} are allowed."
        )

    match inside:
        case "node_id":
            inside_text = node_id
            header_y += 0.025
        case "node_type":
            inside_text = node_type[0]
        case "chain" | "rendering_order" | "level":
            inside_text = node[inside]
            header_y += 0.025

    ax.text(
        p0[0] + size[0] / 2,
        header_y,
        inside_text,
        fontsize=inside_fontsize,
        ha="center",
        va="center",
    )

    # inlet pos
    inlets = config["inlets"]
    num_inlets = len(inlets)
    input_points = {}

    if vertical:
        dx = size[0] / (num_inlets + 1)
        for i, inlet in enumerate(inlets):
            input_points[inlet] = (p0[0] + dx * (i + 1), p0[1])
    else:
        dy = size[1] / (num_inlets + 1)
        for i, inlet in enumerate(inlets):
            input_points[inlet] = (p0[0], p0[1] + dy * (i + 1))
                
    # outlet pos
    outlets = config["outlets"]
    num_outlets = len(outlets)
    output_points = {}

    if vertical:
        dx = size[0] / (num_outlets + 1)
        for i, outlet in enumerate(outlets):
            output_points[outlet] = (p0[0] + dx * (i + 1), p0[1] + size[1])
    else:
        dy = size[1] / (num_outlets + 1)
        for i, outlet in enumerate(outlets):
            output_points[outlet] = (p0[0] + size[0], p0[1] + dy * (i + 1))

    if above is not None:
        if not above in allowed_texts:
            raise Exception(
                f"Provided above: '{above}', but only {allowed_texts} are allowed."
            )

        match above:
            case "node_id":
                above_text = node_id
            case "node_type":
                above_text = node_type[:4]
            case "chain" | "rendering_order" | "level":
                above_text = node[above]
        ax.text(
            p0[0],
            p0[1] - 0.13,
            above_text,
            color="g",
            zorder=5,
            fontsize=above_fontsize,
            ha="left",
            va="center",
        )

    node["meta"] = {
        "y": size[1],
        "in_points": input_points,
        "out_points": output_points,
    }
