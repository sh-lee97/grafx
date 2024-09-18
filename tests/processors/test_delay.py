import conftest
import pytest
from utils import (
    _save_audio_and_mel,
    _test_single_processor,
    create_empty_parameters_from_shape_dict,
)

from grafx.processors.delay import MultitapDelay

# region Fixture


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=[True, False])
def flashfftconv(request):
    return request.param


@pytest.fixture(params=[0, 0.01, 1])
def std(request):
    return request.param


# endregion Fixture


@conftest.quant_test
@pytest.mark.parametrize("processor_cls", [MultitapDelay])
def test_delay_quantitative(processor_cls, std, batch_size=4):
    print(processor_cls.__name__)
    processor = processor_cls(flashfftconv=True)

    _save_audio_and_mel(
        processor, "delay", device="cuda", batch_size=batch_size, std=std
    )


@pytest.mark.parametrize("segment_len", [3000, 6000])
@pytest.mark.parametrize("num_segments", [10, 20])
@pytest.mark.parametrize("num_delay_per_segment", [1, 2])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("zp_filter_per_tap", [True, False])
@pytest.mark.parametrize("straight_through", [True, False])
def test_multitap_delay(
    segment_len,
    num_segments,
    num_delay_per_segment,
    processor_channel,
    zp_filter_per_tap,
    straight_through,
    flashfftconv,
    device,
):
    if device == "cpu" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = MultitapDelay(
        segment_len=segment_len,
        num_segments=num_segments,
        num_delay_per_segment=num_delay_per_segment,
        processor_channel=processor_channel,
        zp_filter_per_tap=zp_filter_per_tap,
        straight_through=straight_through,
        flashfftconv=flashfftconv,
    ).to(device)

    _test_single_processor(processor, device=device)
