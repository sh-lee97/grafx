import pytest
import torch
from utils import _test_single_processor

from grafx.processors.delay import MultitapDelay


@pytest.mark.parametrize("segment_len", [3000, 6000])
@pytest.mark.parametrize("num_segments", [10, 20])
@pytest.mark.parametrize("num_delay_per_segment", [1, 2])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("zp_filter_per_tap", [True, False])
@pytest.mark.parametrize("straight_through", [True, False])
@pytest.mark.parametrize("flashfftconv", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
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
        return

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
