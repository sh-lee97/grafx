import pytest
import torch
from utils import _test_single_processor

from grafx.processors.stereo import *


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_channels", [1, 2])
def test_stereo_gain(num_channels, device):
    processor = StereoGain().to(device)

    _test_single_processor(
        processor,
        num_channels=num_channels,
        device=device,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_side_gain_imager(device):
    processor = SideGainImager().to(device)

    _test_single_processor(
        processor,
        num_channels=2,
        device=device,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mono_to_stereo(device):
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
