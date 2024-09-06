import pytest

from grafx.data import GRAFX, NodeConfigs


def test_invalid_op_error():
    with pytest.raises(Exception):
        GRAFX(invalid_op="invalid")

def test_grafx_initialization_without_config():
    G = GRAFX()
    assert G.graph["counter"] == 0
    assert G.graph["consecutive_ids"] is True
    assert G.graph["batch"] is False
    assert G.graph["config"] is None
    assert G.graph["invalid_op"] == "error"

def test_add_node():
    grafx = GRAFX()
    node_id = grafx.add(node_type="test_type")
    assert node_id == 0
    assert grafx.number_of_nodes() == 1
    assert grafx.nodes[node_id]["node_type"] == "test_type"

def test_grafx_print():
    G = GRAFX()
    in_id = G.add("in")
    out_id = G.add("out")
    G.connect(in_id, out_id)
    print(G)


def test_remove_node():
    grafx = GRAFX()
    node_id = grafx.add(node_type="test_type")
    incoming_edges, outgoing_edges = grafx.remove(node_id)
    assert grafx.number_of_nodes() == 0
    assert len(incoming_edges) == 0
    assert len(outgoing_edges) == 0
    assert not grafx.graph["consecutive_ids"]

def test_connect_nodes():
    grafx = GRAFX()
    source_id = grafx.add(node_type="source")
    dest_id = grafx.add(node_type="dest")
    grafx.connect(source_id, dest_id)
    assert grafx.has_edge(source_id, dest_id)

def test_add_serial_chain():
    grafx = GRAFX()
    node_list = ["type1", "type2", "type3"]
    first_id, last_id = grafx.add_serial_chain(node_list)
    assert first_id == 0
    assert last_id == 2
    assert grafx.has_edge(0, 1)
    assert grafx.has_edge(1, 2)

def test_raise_warning_error():
    grafx = GRAFX(invalid_op="error")
    with pytest.raises(Exception):
        grafx.raise_warning("test error")

def test_raise_warning_warn():
    grafx = GRAFX(invalid_op="warn")
    with pytest.warns(UserWarning):
        grafx.raise_warning("test warning")

def test_raise_warning_mute():
    grafx = GRAFX(invalid_op="mute")
    grafx.raise_warning("test mute")  # Should not raise an exception or warning

def test_property_setters():
    grafx = GRAFX()
    grafx.counter = 5
    assert grafx.counter == 5
    grafx.consecutive_ids = False
    assert not grafx.consecutive_ids
    grafx.batch = True
    assert grafx.batch

    with pytest.raises(Exception):
        grafx.config = None  # Should raise exception

    with pytest.raises(Exception):
        grafx.config_hash = 12345  # Should raise exception

    grafx.invalid_op = "warn"
    assert grafx.invalid_op == "warn"

    grafx.rendering_order_method = "method"
    assert grafx.rendering_order_method == "method"

    grafx.type_sequence = [1, 2, 3]
    assert grafx.type_sequence == [1, 2, 3]