import pytest
from utils import _test_single_processor

from grafx.processors import *


@pytest.mark.parametrize("pre_gain", [True, False])
@pytest.mark.parametrize("inverse_post_gain", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_piecewise_tanh(
    device,
    pre_gain,
    inverse_post_gain,
):
    processor = PiecewiseTanhDistortion(
        pre_gain=pre_gain,
        inverse_post_gain=inverse_post_gain,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("max_order", [2, 10, 20])
@pytest.mark.parametrize("pre_gain", [True, False])
@pytest.mark.parametrize("remove_dc", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chebyshev(
    device,
    pre_gain,
    inverse_post_gain,
):
    processor = ChebyshevDistortion(
        pre_gain=pre_gain,
        inverse_post_gain=inverse_post_gain,
    )
    _test_single_processor(processor, device=device)
