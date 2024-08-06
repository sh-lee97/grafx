from typing import List

import torch
from torch_geometric.utils import scatter

from grafx.data.tensor import GRAFXTensor
from grafx.utils import permute_grafx_tensor

MAX_ITER = 100


def return_render_ordered_tensor(G_t: GRAFXTensor, method, **kwargs):
    type_sequence, render_order = compute_render_order_tensor(G_t, method, **kwargs)

    G_t.type_sequence = [G_t.config.node_types[t] for t in type_sequence]
    G_t.rendering_orders = render_order
    G_t.rendering_order_method = method

    node_id = node_id_from_render_order(render_order)
    G_t = permute_grafx_tensor(G_t, node_id)
    return G_t


@torch.no_grad()
def compute_render_order_tensor(G_t: GRAFXTensor, method: str = "beam", **kwargs):
    match method:
        case "greedy":
            return greedy_search(G_t, **kwargs)
        case "beam":
            return beam_search(G_t, **kwargs)
        case "fixed":
            return fixed_order_search(G_t, **kwargs)
        case "one-by-one":
            return one_by_one_search(G_t, **kwargs)
        case _:
            raise Exception(f"Invalid rendering method: {method}.")


def one_by_one_search(G_t):
    greedy_type_sequence, greedy_render_order = greedy_search(G_t)
    device = greedy_render_order.device
    render_order = -torch.ones(
        len(greedy_render_order), dtype=torch.long, device=device
    )
    type_sequence = []
    i, order = 0, 0
    while True:
        mask = greedy_render_order == order
        if order == 0:
            render_order[mask] = 0
            type_sequence += [0]
            i += 1
        else:
            num = torch.count_nonzero(mask)
            if num == 0:
                break
            node_type = greedy_type_sequence[order].item()
            render_order[mask] = torch.arange(i, i + num, device=device)
            i += num
            type_sequence += [node_type] * num
        order += 1
    return type_sequence, render_order


def fixed_order_search(G_t: GRAFXTensor, fixed_order: List[int]):
    T, E, num_nodes = G_t.node_types, G_t.edge_indices, G_t.num_nodes
    source_ids, dest_ids = E[0], E[1]

    device = T.device

    unique_node_types = sorted(list(set(T.tolist())))
    assert (0 in unique_node_types) and (1 in unique_node_types)
    unique_node_types.remove(0)
    unique_node_types.remove(1)
    unique_node_types = torch.tensor(unique_node_types, dtype=torch.long, device=device)

    type_mask = T[None, :] == unique_node_types[:, None]

    render_order = -torch.ones(num_nodes, dtype=torch.long, device=device)
    render_order[T == 0] = 0

    type_sequence = [0]

    visited = (T == 0) + (T == 1)
    visited = visited[:]

    i = 0
    order_i = 1
    for _ in range(MAX_ITER):
        # scatter
        visited_source = visited[source_ids]
        if T.is_cuda:
            visited_source = visited_source.char()
        new_visited_nodes = ~visited * scatter(
            visited_source, dest_ids, dim=-1, dim_size=num_nodes, reduce="mul"
        )
        if T.is_cuda:
            new_visited_nodes = new_visited_nodes.bool()

        while True:
            i += 1
            node_type_to_check = fixed_order[i]
            type_mask = T == node_type_to_check
            new_visited_nodes_per_type = new_visited_nodes * type_mask
            if torch.any(new_visited_nodes_per_type):
                visited = visited + new_visited_nodes_per_type
                type_sequence.append(node_type_to_check)
                render_order[new_visited_nodes_per_type] = order_i
                order_i += 1
                break

        if visited.all():
            break
        if i == MAX_ITER:
            assert False

    type_sequence.append(1)
    type_sequence = torch.tensor(type_sequence, device=device)
    render_order[T == 1] = order_i
    return type_sequence, render_order


def greedy_search(G_t: GRAFXTensor):
    return beam_search(G_t, width=1, depth=1)


def beam_search(G_t: GRAFXTensor, depth: int = 1, width: int = 64):
    T, E, num_nodes = G_t.node_types, G_t.edge_indices, G_t.num_nodes
    source_ids, dest_ids = E[0], E[1]

    device = T.device

    unique_node_types = sorted(list(set(T.tolist())))
    assert (0 in unique_node_types) and (1 in unique_node_types)
    unique_node_types.remove(0)
    unique_node_types.remove(1)
    unique_node_types = torch.tensor(unique_node_types, dtype=torch.long, device=device)

    type_mask = T[None, :] == unique_node_types[:, None]

    if width > 1 and depth > 1:
        max_num_idxs = width * len(unique_node_types) ** depth
        arange = torch.arange(max_num_idxs, device=device)
    render_order = -torch.ones(1, num_nodes, dtype=torch.long, device=device)
    render_order[:, T == 0] = 0

    type_sequence = torch.zeros(1, 1, dtype=torch.long, device=device)

    visited = (T == 0) + (T == 1)
    visited = visited[None, :]

    for i in range(1, MAX_ITER + 1):
        # scatter
        visited_temp = visited
        for d in range(depth):
            visited_temp_source = visited_temp[..., source_ids]
            if T.is_cuda:
                visited_temp_source = visited_temp_source.char()
            new_visited_nodes = ~visited_temp * scatter(
                visited_temp_source, dest_ids, dim=-1, dim_size=num_nodes, reduce="mul"
            )
            if T.is_cuda:
                new_visited_nodes = new_visited_nodes.bool()
            new_visited_nodes_per_type = type_mask * new_visited_nodes.unsqueeze(-2)
            visited_temp = visited_temp.unsqueeze(-2) + new_visited_nodes_per_type
            num_visited = torch.count_nonzero(visited_temp, -1)
            if d == 0:
                new_visited = visited_temp
                new_visited_nodes_per_type_0 = new_visited_nodes_per_type
            if (num_visited == num_nodes).any():
                break

        shape = num_visited.shape
        num_visited = num_visited.view(shape[0], shape[1], -1)

        if depth > 1:
            lookahead_dim = num_visited.shape[-1]

        num_visited = num_visited.view(-1)

        if width == 1:
            if depth == 1:
                unique_sorted_idx = torch.argmax(num_visited, keepdim=True)
            else:
                unique_sorted_idx = torch.argmax(num_visited, keepdim=True)
                unique_sorted_idx = unique_sorted_idx // lookahead_dim
        else:
            if depth == 1:
                unique_sorted_idx = torch.argsort(num_visited, descending=True)
                unique_sorted_idx = unique_sorted_idx[:width]
            else:
                sorted_idx = torch.argsort(num_visited, descending=True)
                sorted_idx = sorted_idx // lookahead_dim

                unique_sorted_idx, indices = torch.unique(
                    sorted_idx, return_inverse=True
                )  # unique top width
                occurance_order = scatter(
                    arange[: len(sorted_idx)], indices, reduce="min"
                )
                occurance_order_idx = torch.argsort(occurance_order)
                unique_sorted_idx = unique_sorted_idx[occurance_order_idx]
                unique_sorted_idx = unique_sorted_idx[:width]

        prev_idx, curr_type_idx = (
            unique_sorted_idx // shape[1],
            unique_sorted_idx % shape[1],
        )
        visited = new_visited[prev_idx, curr_type_idx]
        type_sequence = torch.cat(
            [type_sequence[prev_idx], unique_node_types[curr_type_idx][:, None]], -1
        )

        render_order = render_order[prev_idx]
        render_order[new_visited_nodes_per_type_0[prev_idx, curr_type_idx]] = i

        all_visited = visited.all(-1)
        if all_visited.any():
            break

        if i == MAX_ITER:
            assert False

    final_idx = torch.argmax(all_visited.long())
    type_sequence = torch.cat(
        [type_sequence[final_idx], torch.tensor([1], device=device)]
    )
    render_order = render_order[final_idx]
    render_order[T == 1] = i + 1
    return type_sequence, render_order


def node_id_from_render_order(
    render_order: torch.LongTensor,
):
    device = render_order.device
    node_id = -torch.ones(len(render_order), dtype=torch.long, device=device)
    i, order = 0, 0
    while True:
        mask = render_order == order
        num = torch.count_nonzero(mask)
        if num == 0:
            break
        node_id[mask] = torch.arange(i, i + num, device=device)
        order += 1
        i += num
    return node_id
