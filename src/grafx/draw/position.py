import networkx as nx
import numpy as np


def compute_node_position(
    G, 
    node_spacing=(0.8, 0.8),
):
    """
    Calculates and assigns $x$ and $y$ coordinates to the nodes in the graph based on their ranks and relative positions.

    Args:
        G (:class:`~grafx.data.graph.GRAFX`): 
            The graph whose node positions are to be computed.
        node_spacing (:python:`Tuple[float]`, *optional*): 
            The horizontal and vertical distance between nodes 
            (default: :python:`(0.8, 0.8)`).

    Returns:
        :python:`None`
    """

    G_sorted, rank_dict, levels_and_chains = compute_rank(G)

    # Compute the maximum relative y-coordinate for each chain
    max_relative_y0 = {k: 0 for k in rank_dict.keys()}
    for chain, ranks in rank_dict.items():
        ranks = {k: sorted(v) for k, v in ranks.items()}
        for rank, node_idxs in ranks.items():
            for node_idx in node_idxs:
                relative_y0 = node_idxs.index(node_idx)
                G.nodes[node_idx]["relative_y0"] = relative_y0
                max_relative_y0[chain] = max(max_relative_y0[chain], relative_y0)

    # Compute the y-coordinate offset for each chain and assign y-coordinates to nodes
    y0_offset, y0_min, y0_max = {}, {}, {}
    c = 0
    for level, chain, predecessors in levels_and_chains:
        if level != 0:
            y0_min_chain = min([y0_min[p] for p in predecessors])
            y0_max_chain = max([y0_max[p] for p in predecessors])
            y0_min[chain] = y0_min_chain
            y0_max[chain] = y0_max_chain
            c = (y0_min_chain + y0_max_chain) / 2
            y0_offset[chain] = c
        else:
            y0_offset[chain] = c
            y0_min[chain] = c
            y0_max[chain] = c
            c += 1 + max_relative_y0[chain]

    for idx, node in G.nodes(data=True):
        node["y0"] = y0_offset[G.nodes[idx]["chain"]] + G.nodes[idx]["relative_y0"]

    # Assign x-coordinates to nodes based on their ranks
    for node_idx in reversed(G_sorted):
        node = G.nodes[node_idx]
        rank = node["rank"]
        G.nodes[node_idx]["x0"] = rank

    # Apply node spacing to x and y coordinates
    for node_id in list(G.nodes):
        G.nodes[node_id]["x0"] *= node_spacing[0]
        G.nodes[node_id]["y0"] *= node_spacing[1]


def compute_rank(G, reverse=False):
    levels_and_chains = estimate_chain(G)
    chains = [t[1] for t in levels_and_chains]

    G_sorted = list(nx.topological_sort(G))
    if reverse:
        G_sorted.reverse()
    rank_module_dict = {k: {} for k in chains}
    for node_idx in G_sorted:
        if G.nodes[node_idx]["node_type"] == "in":
            rank = 0
        else:
            pranks = [G.nodes[n]["rank"] for n in G.predecessors(node_idx)]
            rank = max(pranks) + 1 if len(pranks) != 0 else -1
        G.nodes[node_idx]["rank"] = rank
        if "chain" in G.nodes[node_idx]:
            chain = G.nodes[node_idx]["chain"]
            if rank != -1:
                if rank not in rank_module_dict[chain]:
                    rank_module_dict[chain][rank] = [node_idx]
                else:
                    rank_module_dict[chain][rank].append(node_idx)
    for node_idx in G_sorted:
        if G.nodes[node_idx]["rank"] == -1:
            pranks = [G.nodes[n]["rank"] for n in G.successors(node_idx)]
            rank = min(pranks) - 1 if len(pranks) != 0 else -1
            G.nodes[node_idx]["rank"] = rank
            if "chain" in G.nodes[node_idx]:
                chain = G.nodes[node_idx]["chain"]
                if rank not in rank_module_dict[chain]:
                    rank_module_dict[chain][rank] = [node_idx]
                else:
                    rank_module_dict[chain][rank].append(node_idx)

    empty_keys = []
    for key in rank_module_dict.keys():
        if len(rank_module_dict[key]) == 0:
            empty_keys.append(key)
    for key in empty_keys:
        rank_module_dict.pop(key)
    return G_sorted, rank_module_dict, levels_and_chains


def estimate_chain(G):
    levels_and_chains = []
    G_sorted = list(nx.topological_sort(G))

    for node_idx in G_sorted:
        #if G.nodes[node_idx]["node_type"] == "in":
        if len(G.in_edges(node_idx)) == 0: #G.nodes[node_idx]["node_type"] == "in":
            G.nodes[node_idx]["chain"] = node_idx
            G.nodes[node_idx]["level"] = 0
            levels_and_chains.append((0, node_idx, []))
        else:
            predecessors = G.predecessors(node_idx)
            pchains, plevels = [], []
            for n in predecessors:
                if "chain" in G.nodes[n]:
                    pchains.append(G.nodes[n]["chain"])
                    plevels.append(G.nodes[n]["level"])
            pchains = list(set(pchains))
            plevels = list(set(plevels))
            if len(pchains) == 0:
                continue
            elif len(pchains) == 1:
                G.nodes[node_idx]["chain"] = pchains[0]
                G.nodes[node_idx]["level"] = plevels[0]
            else:
                new_level = 1 + max(plevels)
                new_chain = node_idx
                G.nodes[node_idx]["chain"] = new_chain
                G.nodes[node_idx]["level"] = new_level
                if not new_chain in levels_and_chains:
                    levels_and_chains.append((new_level, new_chain, pchains))

    levels_and_chains = sorted(levels_and_chains)
    return levels_and_chains
