import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def pytest_addoption(parser):
    parser.addoption(
        "--quant",
        action="store_true",
        default=False,
        help="Run quantitative related tests",
    )


def quant_test(func):
    func = pytest.mark.quant(func)
    return func


def pytest_configure(config):
    config.addinivalue_line("markers", "quant: mark test as related to quantitative")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--quant"):
        skip_non_quant = pytest.mark.skip(reason="Skipping non-quantitative tests")
        for item in items:
            if "quant" not in item.keywords:
                item.add_marker(skip_non_quant)
