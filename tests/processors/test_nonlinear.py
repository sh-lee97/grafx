import pytest
from utils import _save_audio_and_mel, _test_single_processor

import conftest
from grafx.processors import *

# region Fixture


@pytest.fixture(params=[True, False])
def remove_dc(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=[0, 0.01, 1])
def std(request):
    return request.param


# endregion Fixture


@conftest.quant_test
@pytest.mark.parametrize(
    "processor_cls",
    [TanhDistortion, PiecewiseTanhDistortion, PowerDistortion, ChebyshevDistortion],
)
def test_nonlinear_quantitative(processor_cls, std, batch_size=4):
    print(processor_cls.__name__)
    processor = processor_cls()
    _save_audio_and_mel(
        processor, "nonlinear", device="cuda", batch_size=batch_size, std=std
    )


@pytest.mark.parametrize("pre_post_gain", [True, False])
@pytest.mark.parametrize("inverse_post_gain", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
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
@pytest.mark.parametrize("use_tanh", [True, False])
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
@pytest.mark.parametrize("use_tanh", [True, False])
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
