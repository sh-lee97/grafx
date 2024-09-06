import torch
from tqdm import tqdm
import torch.nn as nn
import pickle
from tqdm import tqdm
from grafx.data_structures.graph import GRAFX
from grafx.data_structures.configs import NodeConfigs
from grafx.data_structures.conversion import convert_to_tensor
from grafx.draw import draw_grafx
from grafx.render.order.graph import return_render_ordered_graph
from grafx.render.prepare import prepare_render
from grafx.data_structures.configs import DEFAULT
from grafx.data_structures.batch import batch_grafx


def test_graph(config):
    G = GRAFX(config=config)
    _, id_0 = G.add_serial_chain(
        ["in", "noisegate", "gain/panning", "compressor", "mix", "eq", "delay"]
    )

    _, id_1 = G.add_serial_chain(["in", "noisegate", "gain/panning"])
    _, id_2 = G.add_serial_chain(["in", "compressor"])
    _, id_3 = G.add_serial_chain(["in", "eq", "noisegate"])
    _, id_4 = G.add_serial_chain(["in", "gain/panning"])

    id_mix = G.add("mix")
    G.connect(id_1, id_mix)
    G.connect(id_2, id_mix)
    G.connect(id_3, id_mix)
    G.connect(id_4, id_mix)

    id_in, id_out = G.add_serial_chain(["eq", "reverb", "out"])
    G.connect(id_mix, id_in)
    G.connect(id_0, id_out)
    return G


def test_graph_mimo(config):
    G = GRAFX(config=config)
    _, id_0 = G.add_serial_chain(
        ["in", "noisegate", "gain/panning", "compressor", "eq", "delay"]
    )

    _, id_1 = G.add_serial_chain(["in", "noisegate", "gain/panning"])
    _, id_2 = G.add_serial_chain(["in", "compressor"])
    _, id_3 = G.add_serial_chain(["in", "eq", "noisegate_sidechain"])
    _, id_4 = G.add_serial_chain(["in", "gain/panning"])

    id_mix = G.add("mix")
    G.connect(id_1, id_mix)
    G.connect(id_4, id_3, inlet="sidechain")
    G.connect(id_2, id_mix)
    G.connect(id_3, id_mix)
    G.connect(id_4, id_mix)

    id_cross = G.add("crossover")
    G.connect(id_mix, id_cross)
    id_c_1 = G.add("compressor")
    id_c_2 = G.add("compressor")
    G.connect(id_cross, id_c_1, outlet="low")
    G.connect(id_cross, id_c_2, outlet="high")

    id_mix = G.add("mix")
    G.connect(id_c_1, id_mix)
    G.connect(id_c_2, id_mix)

    id_in, id_out = G.add_serial_chain(["eq", "reverb", "out"])
    G.connect(id_mix, id_in)
    G.connect(id_0, id_out)
    return G


def test_tensor():
    G = test_graph()
    print(G)
    G_t = convert_to_tensor(G)


def check_speed():
    parameter = dict(
        eq=torch.rand(130, 1000).cuda(),
        comp=torch.rand(220, 100).cuda(),
    )
    eq_node_splits = [(0, 100), (100, 30)]
    eq_param_splits = [(0, 200), (200, 500), (700, 300)]
    comp_node_splits = [(0, 100), (100, 120)]
    comp_param_splits = [(0, 30), (30, 20), (50, 50)]

    for _ in tqdm(range(10000)):
        for i in range(2):
            eq_node_split = eq_node_splits[i]
            p_eq_i = parameter["eq"].narrow(0, *eq_node_split)
            comp_node_split = comp_node_splits[i]
            p_comp_i = parameter["comp"].narrow(0, *comp_node_split)
            for j in range(3):
                eq_p_split = eq_param_splits[j]
                p_eq_i_j = p_eq_i.narrow(1, *eq_p_split)
                comp_p_split = comp_param_splits[j]
                p_comp_i_j = p_comp_i.narrow(1, *comp_p_split)
                p_eq_i_j.square().sum()
                p_comp_i_j.square().sum()

    for _ in tqdm(range(10000)):
        for j in range(3):
            eq_p_split = eq_param_splits[j]
            p_eq_j = parameter["eq"].narrow(1, *eq_p_split)
            comp_p_split = comp_param_splits[j]
            p_comp_j = parameter["comp"].narrow(1, *comp_p_split)

            for i in range(2):
                eq_node_split = eq_node_splits[i]
                p_eq_i_j = p_eq_j.narrow(0, *eq_node_split)
                comp_node_split = comp_node_splits[i]
                p_comp_i_j = p_comp_j.narrow(0, *comp_node_split)

                p_eq_i_j.square().sum()
                p_comp_i_j.square().sum()


def test_nest():
    eq_param_dict = nn.ParameterDict({"log_magnitude": torch.rand(10, 1000)})
    comp_param_dict = nn.ParameterDict(
        {
            "log_alpha": torch.rand(10, 1),
            "threshold": torch.rand(10, 1),
        }
    )
    param_dict = nn.ParameterDict({"eq": eq_param_dict, "compressor": comp_param_dict})


def test_legacy_render():
    ddsp_processors = [
        "eq",  # e
        "compressor",  # c
        "iir_noisegate",  # n
        "imager",  # i
        "gain/panning",  # p
        "multitap",  # m
        "reverb",  # r
    ]
    G = pickle.load(
        open(
            "/data1/grafx-temp/medley_TablaBreakbeatScience_RockSteady/reverb_orig_4.pickle",
            "rb",
        )
    )

    for _ in tqdm(range(1000)):
        G_t = convert_to_diff_pyg(G)
    for _ in tqdm(range(1000)):
        G = return_ordered_graph(
            G,
            order="orderwise",
            policy_kwargs=dict(fixed_processor_order=ddsp_processors),
        )
    # print("=====")
    # G_t = convert_to_diff_pyg(G)
    # print(G_t.T)
    # print(G_t.Order)
    # for _ in tqdm(range(250)):
    for _ in tqdm(range(1000)):
        render_data = precompute_processing_data(G_t)


def test_render_order():
    node_types = ["noisegate", "compressor", "gain/panning", "reverb", "eq", "delay"]
    node_type_dict = {k: DEFAULT for k in node_types}
    node_type_dict["crossover"] = {"inlets": ["main"], "outlets": ["low", "high"]}
    config = NodeConfigs(node_type_dict=node_type_dict)
    G = test_graph(config)

    fig, ax = draw_grafx(G)
    fig.savefig("graph.pdf", bbox_inches="tight", pad_inches=0)

    render_order = [
        "in",
        "eq",
        "noisegate",
        "gain/panning",
        "compressor",
        "mix",
        "eq",
        "reverb",
        "delay",
        "out",
    ]
    render_order = [config.node_type_to_index[t] for t in render_order]
    G = return_render_ordered_graph(G, method="fixed", fixed_order=render_order)
    fig, ax = draw_grafx(G, show_index=False, show_render_order=True)
    fig.savefig("fixed-order.pdf", bbox_inches="tight", pad_inches=0)

    G = return_render_ordered_graph(G, method="one-by-one")
    fig, ax = draw_grafx(G, show_index=False, show_render_order=True)
    fig.savefig("one-by-one.pdf", bbox_inches="tight", pad_inches=0)
    return

    for d in [1, 2, 3, 4]:
        G = return_render_ordered_graph(G, method="greedy", depth=d)
        fig, ax = draw_grafx(G, show_index=False, show_render_order=True)
        fig.savefig(f"depth-{d}.pdf", bbox_inches="tight", pad_inches=0)

    for d in [1, 2]:
        for w in [16, 64, 256]:
            G = return_render_ordered_graph(G, method="beam", depth=d, width=w)
            fig, ax = draw_grafx(G, show_index=False, show_render_order=True)
            fig.savefig(f"beam-{d}-{w}.pdf", bbox_inches="tight", pad_inches=0)


def test_conversion_rule():
    conversion_rule = GRAFXNodeConfig(config_path="grafx/configs/afx_modules/ddsp.yaml")


def test_batch():
    node_types = ["noisegate", "compressor", "gain/panning", "reverb", "eq", "delay"]
    node_type_dict = {k: DEFAULT for k in node_types}
    node_type_dict["crossover"] = {"inlets": ["main"], "outlets": ["low", "high"]}
    node_type_dict["noisegate_sidechain"] = {
        "inlets": ["main", "sidechain"],
        "outlets": ["main"],
    }
    # config = NodeConfigs(node_type_dict=node_type_dict)
    config = NodeConfigs(node_type_dict)
    G = test_graph(config)
    print(G.graph)
    print(G.counter)
    G.nodes[0]["node_type"] = "hi"
    #G = test_graph_mimo(config)
    # G = test_graph(config)
    # print(config)
    # print(config.outlet_to_index)
    # print(G)
    print(G)
    G, counters = batch_grafx([G] * 2)
    print(G)
    # fig, ax = draw_grafx(G, inside_node="rendering_order", above_node="node_id")
    # fig.savefig("graph_mino_render.pdf", bbox_inches="tight", pad_inches=0)

    #for _ in tqdm(range(100)):
    #    G = return_render_ordered_graph(G, "beam", width=256)
    #    G_tensor = convert_to_tensor(G)
    #    render_data = prepare_render(G_tensor)
    # render_data = prepare_render_graph(G)
    # print(render_data)


if __name__ == "__main__":
    # test_graph_new()
    # test_tensor()
    # check_speed()
    # test_nest()
    # test_legacy_render()
    # test_conversion_rule()
    # test_render_order()
    test_batch()
