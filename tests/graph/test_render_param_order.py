"""Regression: per-type parameters are supplied in node-id (convert) order and must be
mapped to the correct nodes through the rendering reorder, even when a type spans render
stages out of node-id order. Guards the `parameter_indices` mechanism.

Before the fix, `render_grafx` consumed per-type rows in reorder order while callers built
them in node-id order, so two same-type nodes scheduled out of id order swapped parameters
(a silent, value-wrong render — not caught by tests that use random params + no assertion).
"""

import torch

from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.processors import ApproxCompressor, ZeroPhaseFIREqualizer
from grafx.render import prepare_render, render_grafx, reorder_for_fast_render


def _node_params(proc):
    return {k: torch.rand(v if isinstance(v, (tuple, list)) else (v,))
            for k, v in proc.parameter_size().items()}


def _per_type_node_id_order(processors, G, node_params, device):
    """Stack each type's named params over its nodes in node-id (convert) order — the
    natural caller convention that `parameter_indices` now makes correct."""
    rows = {}
    for nid in sorted(G.nodes):
        t = G.nodes[nid]["node_type"]
        if t in processors:
            rows.setdefault(t, []).append(node_params[nid])
    return {t: {k: torch.stack([d[k] for d in lst]).to(device) for k in lst[0]}
            for t, lst in rows.items()}


def _render(processors, G, node_params, x):
    G_t = reorder_for_fast_render(convert_to_tensor(G), method="beam")
    rd = prepare_render(G_t)
    params = _per_type_node_id_order(processors, G, node_params, x.device)
    y, _, _ = render_grafx(processors, x.unsqueeze(0), params, rd)
    return y[0]


def _build(procs, config, add_order, pcf, pcs, peq, x):
    G = GRAFX(config=config)
    ids = {nm: G.add("c" if nm in ("Cf", "Cs") else "eq" if nm == "eq" else nm)
           for nm in add_order}
    G.connect(ids["in"], ids["Cf"]); G.connect(ids["Cf"], ids["eq"])
    G.connect(ids["eq"], ids["Cs"]); G.connect(ids["Cs"], ids["out"])
    node_params = {ids["Cf"]: pcf, ids["Cs"]: pcs, ids["eq"]: peq}
    return _render(procs, G, node_params, x), (G, ids)


def test_parameter_order_invariant_to_node_id():
    """in -> C_first -> eq -> C_second -> out: the two compressors render at different
    stages. Output must not depend on which node ids they get (params in node-id order)."""
    config = NodeConfigs(["c", "eq"])
    procs = {"c": ApproxCompressor(flashfftconv=False), "eq": ZeroPhaseFIREqualizer()}
    torch.manual_seed(0)
    pcf, pcs, peq = _node_params(procs["c"]), _node_params(procs["c"]), _node_params(procs["eq"])
    x = torch.randn(2, 2 ** 13)

    y_canon, _ = _build(procs, config, ["in", "Cf", "eq", "Cs", "out"], pcf, pcs, peq, x)
    y_scram, _ = _build(procs, config, ["in", "out", "Cs", "Cf", "eq"], pcf, pcs, peq, x)
    assert torch.allclose(y_canon, y_scram, atol=1e-5), "render depends on node-id order"

    # sanity: swapping the two compressors' params changes the output, so the invariance
    # above is meaningful (params are actually assigned per-node, not degenerate).
    y_swap, _ = _build(procs, config, ["in", "Cf", "eq", "Cs", "out"], pcs, pcf, peq, x)
    assert not torch.allclose(y_canon, y_swap, atol=1e-5), "compressor params not distinguishable"
