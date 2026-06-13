"""Per-edge gain rendering: static (graph attr) + dynamic (render arg) + backprop,
and backward-compatibility (no gains => unchanged)."""

import torch
import torch.nn as nn

from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.render import prepare_render, render_grafx, reorder_for_fast_render
from grafx.utils import create_empty_parameters


class Ident(nn.Module):
    def parameter_size(self):
        return {}

    def forward(self, x):
        return x * 1.0


def _two_branch_graph(cfg, g1=None, g2=None):
    """in -> i1 -> mix ; in -> i2 -> mix ; mix -> out, with gains g1/g2 on i*->mix."""
    G = GRAFX(config=cfg)
    i_in = G.add("in")
    i1 = G.add("ident")
    i2 = G.add("ident")
    i_mix = G.add("mix")
    i_out = G.add("out")
    G.connect(i_in, i1)
    G.connect(i_in, i2)
    G.connect(i1, i_mix, gain=g1)
    G.connect(i2, i_mix, gain=g2)
    G.connect(i_mix, i_out)
    return G, (i1, i2, i_mix)


def _compile(G):
    G_t = convert_to_tensor(G)
    G_t = reorder_for_fast_render(G_t, method="beam")
    return G_t, prepare_render(G_t)


def _edge_gain_vector(G_t, pairs_to_gain):
    """Build a convert-order edge-gain vector from {(src,dst): gain}; others 1.0."""
    ei = G_t.edge_indices  # (2, E)
    E = ei.shape[1]
    gains = torch.ones(E)
    for k in range(E):
        key = (int(ei[0, k]), int(ei[1, k]))
        if key in pairs_to_gain:
            gains[k] = pairs_to_gain[key]
    return gains


def test_static_edge_gain_weighted_sum():
    cfg = NodeConfigs(["ident"])
    G, (i1, i2, _) = _two_branch_graph(cfg, g1=2.0, g2=3.0)
    _, rd = _compile(G)
    procs = {"ident": Ident()}
    params = create_empty_parameters(procs, G)
    x = torch.randn(1, 2, 4096)
    y, _, _ = render_grafx(procs, x, params, rd)  # static gains used
    assert torch.allclose(y[0], (2.0 * x + 3.0 * x)[0], atol=1e-6)  # 5x


def test_no_gain_is_unchanged():
    cfg = NodeConfigs(["ident"])
    G, _ = _two_branch_graph(cfg)  # no gains
    _, rd = _compile(G)
    procs = {"ident": Ident()}
    params = create_empty_parameters(procs, G)
    x = torch.randn(1, 2, 4096)
    y, _, _ = render_grafx(procs, x, params, rd)
    assert torch.allclose(y[0], (x + x)[0], atol=1e-6)  # 2x


def test_dynamic_edge_gains_override():
    cfg = NodeConfigs(["ident"])
    G, (i1, i2, i_mix) = _two_branch_graph(cfg, g1=2.0, g2=3.0)  # static 2/3
    G_t, rd = _compile(G)
    procs = {"ident": Ident()}
    params = create_empty_parameters(procs, G)
    x = torch.randn(1, 2, 4096)
    # override: i1->mix = 5, i2->mix = 1
    eg = _edge_gain_vector(G_t, {(i1, i_mix): 5.0, (i2, i_mix): 1.0})
    y, _, _ = render_grafx(procs, x, params, rd, edge_gains=eg)
    assert torch.allclose(y[0], (5.0 * x + 1.0 * x)[0], atol=1e-6)  # 6x, overrides 5x


class _Split(nn.Module):
    def parameter_size(self):
        return {}

    def forward(self, x):
        return [x * 1.0, x * 1.0]  # low, high (identity; gains differentiate them)


class _Merge(nn.Module):
    def parameter_size(self):
        return {}

    def forward(self, a, b):
        return a + b


def test_edge_gains_on_non_siso_per_inlet():
    """Distinct gains on the two inlets of a merge node (non-SISO path) are applied
    to the correct inlet."""
    cfg = NodeConfigs({
        "split": {"inlets": ["main"], "outlets": ["low", "high"]},
        "merge": {"inlets": ["a", "b"], "outlets": ["main"]},
        "ident": {"inlets": ["main"], "outlets": ["main"]},
    })
    G = GRAFX(config=cfg)
    i_in = G.add("in")
    i_sp = G.add("split")
    i_a = G.add("ident")
    i_b = G.add("ident")
    i_mg = G.add("merge")
    i_out = G.add("out")
    G.connect(i_in, i_sp)
    G.connect(i_sp, i_a, outlet="low")
    G.connect(i_sp, i_b, outlet="high")
    G.connect(i_a, i_mg, inlet="a", gain=2.0)
    G.connect(i_b, i_mg, inlet="b", gain=5.0)
    G.connect(i_mg, i_out)
    assert cfg.siso_only is False

    G_t = convert_to_tensor(G)
    G_t = reorder_for_fast_render(G_t, method="beam")
    rd = prepare_render(G_t)
    procs = {"split": _Split(), "merge": _Merge(), "ident": Ident()}
    params = create_empty_parameters(procs, G)
    x = torch.randn(1, 2, 4096)
    y, _, _ = render_grafx(procs, x, params, rd)
    # merge(a,b) = 2*x + 5*x = 7x (inlet routing + per-inlet gains correct)
    assert torch.allclose(y[0], (7.0 * x)[0], atol=1e-6)


def test_batched_edge_gains_rejected():
    """(B, E) gains must raise (not silently mis-render); use batch_grafx instead."""
    import pytest

    cfg = NodeConfigs(["ident"])
    G, (i1, i2, i_mix) = _two_branch_graph(cfg)
    G_t, rd = _compile(G)
    procs = {"ident": Ident()}
    params = create_empty_parameters(procs, G)
    x = torch.randn(4, 1, 2, 4096)  # 4D
    E = G_t.edge_indices.shape[1]
    with pytest.raises(NotImplementedError):
        render_grafx(procs, x, params, rd, edge_gains=torch.ones(4, E))


def test_edge_gains_backprop():
    cfg = NodeConfigs(["ident"])
    G, (i1, i2, i_mix) = _two_branch_graph(cfg)  # no static gains
    G_t, rd = _compile(G)
    procs = {"ident": Ident()}
    params = create_empty_parameters(procs, G)
    x = torch.randn(1, 2, 4096)
    eg = _edge_gain_vector(G_t, {(i1, i_mix): 2.0, (i2, i_mix): 3.0}).requires_grad_(True)
    y, _, _ = render_grafx(procs, x, params, rd, edge_gains=eg)
    y.pow(2).mean().backward()
    assert eg.grad is not None and torch.isfinite(eg.grad).all()
    # the two gained edges must receive non-zero gradient
    assert eg.grad.abs().sum() > 0
