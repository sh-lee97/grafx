import pytest

from grafx.data import NodeConfigs


def test_node_configs_with_list():
    config_list = ["custom1", "custom2"]
    node_configs = NodeConfigs(config_list)

    assert node_configs.num_node_types == 5
    assert "custom1" in node_configs.node_types
    assert "custom2" in node_configs.node_types
    assert node_configs["custom1"] == {"inlets": ["main"], "outlets": ["main"]}
    assert node_configs["custom2"] == {"inlets": ["main"], "outlets": ["main"]}

def test_node_configs_with_dict():
    custom_config = {
        "custom1": {"inlets": ["input1"], "outlets": ["output1"]},
        "custom2": {"inlets": ["input2"], "outlets": ["output2"]},
    }
    node_configs = NodeConfigs(custom_config)

    assert node_configs.num_node_types == 5
    assert node_configs["custom1"] == {"inlets": ["input1"], "outlets": ["output1"]}
    assert node_configs["custom2"] == {"inlets": ["input2"], "outlets": ["output2"]}

def test_invalid_config_type():
    with pytest.raises(ValueError, match="Invalid type for config."):
        NodeConfigs(123)  # Invalid type

def test_node_configs_str():
    config_list = ["custom1", "custom2"]
    node_configs = NodeConfigs(config_list)
    print(node_configs)