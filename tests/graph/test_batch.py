from grafx.data import GRAFX, NodeConfigs, batch_grafx


def test_batch_grafx_simple():
    cfg = NodeConfigs(["eq"])

    G = GRAFX(config=cfg)
    out_id = G.add("out")
    start_id, end_id = G.add_serial_chain(["eq"])

    G.connect(end_id, out_id)

    G_list = [G for _ in range(2)]
    G_batch = batch_grafx(G_list)

    print(G_batch)


if __name__ == "__main__":
    test_batch_grafx_simple()
