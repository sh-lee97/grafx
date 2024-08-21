import pytest

from grafx.data import GRAFX, NodeConfigs


def test_g_with_node_configs():
    config = NodeConfigs(["custom"])
    g = GRAFX(config=config)
    node_id = g.add(node_type="custom")
    assert node_id == 0
    assert g.number_of_nodes() == 1
    assert g.nodes[node_id]["node_type"] == "custom"

def test_invalid_node_type_with_config():
    config = NodeConfigs(["valid_type"])
    with pytest.raises(Exception):
        G = GRAFX(config=config)
        G.add(node_type="invalid_type")
    assert G.number_of_nodes() == 0

    G = GRAFX(invalid_op="warn", config=config)  
    with pytest.warns(UserWarning):
        G.add(node_type="invalid_type")
    assert G.number_of_nodes() == 0

    G = GRAFX(invalid_op="mute", config=config)
    G.add(node_type="invalid_type")
    assert G.number_of_nodes() == 0


def test_connect_nodes_with_config():
    config = NodeConfigs({
        "source": {"inlets": [], "outlets": ["main"]},
        "dest": {"inlets": ["main"], "outlets": []}
    })
    g = GRAFX(config=config)
    source_id = g.add(node_type="source")
    dest_id = g.add(node_type="dest")
    g.connect(source_id, dest_id)
    assert g.has_edge(source_id, dest_id)

def test_add_serial_chain_with_config():
    config = NodeConfigs(["type1", "type2", "type3"])
    g = GRAFX(config=config)
    node_list = ["type1", "type2", "type3"]
    first_id, last_id = g.add_serial_chain(node_list)
    assert first_id == 0
    assert last_id == 2
    assert g.has_edge(0, 1)
    assert g.has_edge(1, 2)

def test_invalid_op_error():
    with pytest.raises(Exception):
        GRAFX(invalid_op="invalid")

def test_raise_warning_error():
    g = GRAFX(invalid_op="error")
    with pytest.raises(Exception, match="test error"):
        g.raise_warning("test error")

def test_raise_warning_warn():
    g = GRAFX(invalid_op="warn")
    with pytest.warns(UserWarning, match="test warning"):
        g.raise_warning("test warning")

def test_raise_warning_mute():
    g = GRAFX(invalid_op="mute")
    g.raise_warning("test mute")  # Should not raise an exception or warning

def test_property_setters():
    g = GRAFX()
    g.counter = 5
    assert g.counter == 5
    g.consecutive_ids = False
    assert not g.consecutive_ids
    g.batch = True
    assert g.batch

    with pytest.raises(Exception):
        g.config = None  # Should raise exception

    with pytest.raises(Exception):
        g.config_hash = 12345  # Should raise exception

    g.invalid_op = "warn"
    assert g.invalid_op == "warn"

    g.rendering_order_method = "method"
    assert g.rendering_order_method == "method"

    g.type_sequence = [1, 2, 3]
    assert g.type_sequence == [1, 2, 3]