import pytest
from utils import _save_audio_mel, _test_single_processor

from grafx.processors.stereo import *


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.mark.parametrize("processor_cls", [StereoGain, SideGainImager, MonoToStereo])
def test_stereo_quantitative(processor_cls, batch_size=4):
    print(processor_cls.__name__)

    if processor_cls == StereoGain:
        num_channels = 2
    elif processor_cls == SideGainImager:
        num_channels = 2
    elif processor_cls == MonoToStereo:
        num_channels = 1
    else:
        raise ValueError(f"Unsupported processor class: {processor_cls}")

    processor = processor_cls()
    _save_audio_mel(
        processor,
        "stereo",
        device="cuda",
        num_channels=num_channels,
        batch_size=batch_size,
    )


@pytest.mark.parametrize("num_channels", [1, 2])
def test_stereo_gain_sanity(num_channels, device):
    processor = StereoGain().to(device)

    _test_single_processor(
        processor,
        num_channels=num_channels,
        device=device,
    )


def test_side_gain_imager_sanity(device):
    processor = SideGainImager().to(device)

    _test_single_processor(
        processor,
        num_channels=2,
        device=device,
    )


def test_mono_to_stereo_sanity(device):
    processor = MonoToStereo().to(device)

    _test_single_processor(processor, num_channels=1, device=device)


# @pytest.mark.parametrize("device", ["cpu", "cuda"])
# def test_stereo_to_midside(device):
#    processor = MonoToStereo().to(device)
#
#    _test_single_processor(processor, num_channels=1, device=device)
#
#
# @pytest.mark.parametrize("device", ["cpu", "cuda"])
# def test_midside_to_stereo(device):
#    processor = MonoToStereo().to(device)
#
#    _test_single_processor(processor, num_channels=1, device=device)
#
