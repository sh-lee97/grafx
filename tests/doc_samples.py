from grafx.data_structures.conversion import convert_to_tensor
from grafx.data_structures.graph import GRAFX
from grafx.draw import draw_grafx
from grafx.render.order.graph import return_render_ordered_graph
from grafx.render.order.tensor import return_render_ordered_tensor
from grafx.render.prepare import prepare_render
from grafx.render.graph import render_grafx
from grafx.data_structures.configs import NodeConfigs
from grafx.ddsp.processors import *
from grafx.utils import create_empty_parameters, count_nodes_per_type
import matplotlib.pyplot as plt 
from grafx.ddsp import processors

def test():
    config = NodeConfigs(["eq", "compressor", "reverb"])
    print(config)

    G = GRAFX(config=config)
    print(G)
    print(G)

    number_of_sources = 3
    out_id = G.add("out")
    #out_id = G.add("noisegate")
    print(G)
    for _ in range(number_of_sources):
        chain = ["in", "eq", "compressor", "reverb"]
        start_id, end_id = G.add_serial_chain(chain)
        G.connect(end_id, out_id)

    #print(G)

    fig, ax = draw_grafx(G)
    fig.savefig("example.svg", bbox_inches="tight", pad_inches=0)


    #G_tensor = convert_to_tensor(G)
    #print(G_tensor.rendering_orders)
    #print(G_tensor)

    #G = return_render_ordered_graph(G, method="beam")
    G_t = convert_to_tensor(G)
    print(G_t)
    G_t = return_render_ordered_tensor(G_t, method="beam")
    render_data = prepare_render(G_t)
    print(G_t)
    print(render_data)
    print(render_data.iter_list)


    processors = {
        "eq": ZeroPhaseFIREqualizer(),
        "compressor": ApproxCompressor(),
        "reverb": MidSideFilteredNoiseReverb(),
    }

    parameters = create_empty_parameters(processors, G)

def logo():
    config = NodeConfigs(list("g"))
    G = GRAFX(config=config)
    G.add("g")
    fig, ax = draw_grafx(G)
    fig.savefig("logo.svg", bbox_inches="tight", pad_inches=0.013)

if __name__ == "__main__":
    logo()
    #test()
