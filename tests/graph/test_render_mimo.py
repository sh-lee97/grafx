"""Multi-input / multi-output (non-SISO) rendering correctness + backprop.

Regression tests for the non-SISO ``prepare_render`` path: each edge must route by
its OWN (outlet, inlet), and parallel edges between the same node pair must convert.
"""

import torch
import torch.nn as nn

from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.render import prepare_render, render_grafx, reorder_for_fast_render
from grafx.utils import create_empty_parameters


class Split(nn.Module):  # 1 inlet -> 2 outlets [low, high], distinct scales
    def parameter_size(self):
        return {}

    def forward(self, x):
        return [x * 0.5, x * 0.25]


class Merge(nn.Module):  # 2 inlets [a, b] -> 1 outlet; ORDER-sensitive
    def parameter_size(self):
        return {}

    def forward(self, a, b):
        return a - b


class Ident(nn.Module):
    def parameter_size(self):
        return {}

    def forward(self, x):
        return x * 1.0


def _cfg():
    return NodeConfigs({
        "split": {"inlets": ["main"], "outlets": ["low", "high"]},
        "merge": {"inlets": ["a", "b"], "outlets": ["main"]},
        "ident": {"inlets": ["main"], "outlets": ["main"]},
    })


def _compile(G):
    G_t = convert_to_tensor(G)
    G_t = reorder_for_fast_render(G_t, method="beam")
    return prepare_render(G_t)


def test_multi_outlet_and_multi_inlet_routing():
    """in -> split; split.low -> a-branch -> merge.a; split.high -> b-branch -> merge.b."""
    cfg = _cfg()
    assert cfg.siso_only is False
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
    G.connect(i_a, i_mg, inlet="a")
    G.connect(i_b, i_mg, inlet="b")
    G.connect(i_mg, i_out)

    rd = _compile(G)
    procs = {"split": Split(), "merge": Merge(), "ident": Ident()}
    x = torch.randn(1, 2, 4096)
    params = create_empty_parameters(procs, G)
    y, _, _ = render_grafx(procs, x, params, rd)

    expected = x * 0.5 - x * 0.25  # merge(low, high)
    assert y.shape == (1, 2, 4096)
    assert torch.allclose(y[0], expected[0], atol=1e-6)
    # would equal the swapped result only if inlet routing were wrong
    assert not torch.allclose(y[0], (x * 0.25 - x * 0.5)[0], atol=1e-6)


def test_parallel_edges_same_pair_convert_and_render():
    """split.low -> merge.a and split.high -> merge.b are two edges between the SAME
    (split, merge) pair — must not crash convert_to_tensor and must route correctly."""
    cfg = _cfg()
    G = GRAFX(config=cfg)
    i_in = G.add("in")
    i_sp = G.add("split")
    i_mg = G.add("merge")
    i_out = G.add("out")
    G.connect(i_in, i_sp)
    G.connect(i_sp, i_mg, outlet="low", inlet="a")
    G.connect(i_sp, i_mg, outlet="high", inlet="b")
    G.connect(i_mg, i_out)

    rd = _compile(G)
    procs = {"split": Split(), "merge": Merge(), "ident": Ident()}
    x = torch.randn(1, 2, 4096)
    params = create_empty_parameters(procs, G)
    y, _, _ = render_grafx(procs, x, params, rd)
    assert torch.allclose(y[0], (x * 0.5 - x * 0.25)[0], atol=1e-6)


def test_mimo_backprop():
    """Gradients flow through a MIMO graph to a learnable parameter."""
    cfg = _cfg()

    class ScaledSplit(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Parameter(torch.tensor(0.5))

        def parameter_size(self):
            return {}

        def forward(self, x):
            return [x * self.g, x * 0.25]

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
    G.connect(i_a, i_mg, inlet="a")
    G.connect(i_b, i_mg, inlet="b")
    G.connect(i_mg, i_out)
    rd = _compile(G)

    procs = {"split": ScaledSplit(), "merge": Merge(), "ident": Ident()}
    x = torch.randn(1, 2, 4096)
    params = create_empty_parameters(procs, G)
    y, _, _ = render_grafx(procs, x, params, rd)
    y.pow(2).mean().backward()
    assert procs["split"].g.grad is not None
    assert torch.isfinite(procs["split"].g.grad) and procs["split"].g.grad.abs() > 0
