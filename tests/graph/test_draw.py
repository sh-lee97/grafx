import matplotlib.pyplot as plt

from grafx.data import GRAFX, NodeConfigs
from grafx.draw import draw_grafx


def test_draw_grafx():
    config = NodeConfigs(["in", "eq", "out"])
    G = GRAFX(config=config)
    _, out_id = G.add_serial_chain(["in", "eq", "out"])
    _, eq_id = G.add_serial_chain(["in", "eq"])
    G.connect(eq_id, out_id)
    draw_grafx(G)