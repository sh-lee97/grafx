import pytest
from utils import _save_audio_mel, _test_single_processor

from grafx.processors import *


@pytest.mark.parametrize(
    "processor_cls",
    [TanhDistortion, PiecewiseTanhDistortion, PowerDistortion, ChebyshevDistortion],
)
def test_nonlinear_quantitative(processor_cls, batch_size=4):
    print(processor_cls.__name__)
    processor = processor_cls()
    _save_audio_mel(processor, "nonlinear", device="cuda", batch_size=batch_size)


@pytest.mark.parametrize("pre_post_gain", [True, False])
@pytest.mark.parametrize("inverse_post_gain", [True, False])
@pytest.mark.parametrize("remove_dc", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tanh(
    device,
    pre_post_gain,
    inverse_post_gain,
    remove_dc,
    use_bias,
):
    processor = TanhDistortion(
        pre_post_gain=pre_post_gain,
        inverse_post_gain=inverse_post_gain,
        remove_dc=remove_dc,
        use_bias=use_bias,
    ).to(device)
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("pre_post_gain", [True, False])
@pytest.mark.parametrize("inverse_post_gain", [True, False])
@pytest.mark.parametrize("remove_dc", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_piecewise_tanh(
    device,
    pre_post_gain,
    inverse_post_gain,
    remove_dc,
):
    processor = PiecewiseTanhDistortion(
        pre_post_gain=pre_post_gain,
        inverse_post_gain=inverse_post_gain,
        remove_dc=remove_dc,
    ).to(device)
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("max_order", [2, 10, 20])
@pytest.mark.parametrize("pre_gain", [True, False])
@pytest.mark.parametrize("remove_dc", [True, False])
@pytest.mark.parametrize("use_tanh", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_power(
    max_order,
    pre_gain,
    remove_dc,
    use_tanh,
    device,
):
    processor = PowerDistortion(
        max_order=max_order,
        pre_gain=pre_gain,
        remove_dc=remove_dc,
        use_tanh=use_tanh,
    ).to(device)
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("max_order", [2, 10, 20])
@pytest.mark.parametrize("pre_gain", [True, False])
@pytest.mark.parametrize("remove_dc", [True, False])
@pytest.mark.parametrize("use_tanh", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chebyshev(
    max_order,
    pre_gain,
    remove_dc,
    use_tanh,
    device,
):
    processor = ChebyshevDistortion(
        max_order=max_order,
        pre_gain=pre_gain,
        remove_dc=remove_dc,
        use_tanh=use_tanh,
    ).to(device)
    _test_single_processor(processor, device=device)
