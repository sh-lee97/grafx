import torch

from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.processors import (
    ApproxCompressor,
    MidSideFilteredNoiseReverb,
    ZeroPhaseFIREqualizer,
)
from grafx.render import prepare_render, render_grafx, reorder_for_fast_render
from grafx.utils import create_empty_parameters


def test_render_pipeline():
    config = NodeConfigs(["eq", "compressor", "reverb"])

    G = GRAFX(config=config)

    number_of_sources = 3
    out_id = G.add("out")
    for _ in range(number_of_sources):
        chain = ["in", "eq", "compressor", "reverb"]
        _, end_id = G.add_serial_chain(chain)
        G.connect(end_id, out_id)

    G_t = convert_to_tensor(G)
    G_t = reorder_for_fast_render(G_t, method="beam")
    render_data = prepare_render(G_t)


    processors = {
        "eq": ZeroPhaseFIREqualizer(),
        "compressor": ApproxCompressor(flashfftconv=False),
        "reverb": MidSideFilteredNoiseReverb(flashfftconv=False),
    }

    parameters = create_empty_parameters(processors, G)
    input_signals = torch.zeros(3, 2, 2 ** 17)
    render_grafx(processors, input_signals, parameters, render_data)