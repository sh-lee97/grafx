import networkx as nx


def batch_grafx(G_list):
    r"""
    Batch a list of graphs into a single large disconnected graph.

    Args:
        G_list (:python:`List[GRAFX]`): A list of graphs to batch.

    Returns:
        :class:`~grafx.data.graph.GRAFX`: A single batched graph.
    """
    counters, counter = [], 0
    new_G_list = []
    for i, G in enumerate(G_list):
        if not G.consecutive_ids:
            raise Exception("The node ids must be consecutive.")
        if G.batch:
            raise Exception(f"Graph of index {i} is already a batched graph.")
        if i == 0:
            config_hash = G.config_hash
        else:
            if config_hash != G.config_hash:
                raise Exception("Graphs with different node configs cannot be batched.")
        if i != 0:
            relabel_mapping = {i: i + counter for i in range(G.number_of_nodes())}
            G = nx.relabel_nodes(G, relabel_mapping)

        new_G_list.append(G)
        counter += G.counter
        counters.append(counter)

    G_batch = nx.union_all(new_G_list)
    G_batch.counter = counters
    G_batch.batch = True
    return G_batch
