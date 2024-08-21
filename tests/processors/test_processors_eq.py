import pytest
from utils import _test_lti_processor, _test_single_processor

from grafx.processors import *


@pytest.mark.parametrize("num_frequency_bins", [256, 1024])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("flashfftconv", [True, False])
@pytest.mark.parametrize(
    "window", ["hann", "hamming", "blackman", "bartlett", "kaiser", "boxcar", None]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zerophase_fir_equalizer_without_filterbank(
    num_frequency_bins, processor_channel, flashfftconv, window, device
):
    if device == "cpu" and flashfftconv:
        return

    processor = NewZeroPhaseFIREqualizer(
        num_frequency_bins=num_frequency_bins,
        processor_channel=processor_channel,
        use_filterbank=False,
        flashfftconv=flashfftconv,
        filterbank_kwargs={},
        window=window,
        window_kwargs={},
        eps=1e-7,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("num_frequency_bins", [256, 1024])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("flashfftconv", [True, False])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("num_filters", [50])
@pytest.mark.parametrize("scale", ["bark_traunmuller", "mel_slaney", "linear", "log"])
@pytest.mark.parametrize("f_max", [8000, 22050])
@pytest.mark.parametrize("sr", [16000])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zerophase_fir_equalizer_with_filterbank(
    num_frequency_bins,
    processor_channel,
    flashfftconv,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    device,
):
    if device == "cpu" and flashfftconv:
        return

    filterbank_kwargs = dict(
        num_filters=num_filters,
        scale=scale,
        f_max=f_max,
        sr=sr,
    )

    processor = NewZeroPhaseFIREqualizer(
        num_frequency_bins=num_frequency_bins,
        processor_channel=processor_channel,
        use_filterbank=True,
        flashfftconv=flashfftconv,
        filterbank_kwargs=filterbank_kwargs,
        window=window,
        window_kwargs={},
        eps=1e-7,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("num_frequency_bins", [1024])
@pytest.mark.parametrize("processor_channel", ["stereo"])
@pytest.mark.parametrize("flashfftconv", [False])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("num_filters", [50])
@pytest.mark.parametrize(
    "scale",
    [
        "bark_traunmuller",
        "bark_schroeder",
        "bark_wang",
        "mel_htk",
        "mel_slaney",
        "linear",
        "log",
    ],
)
@pytest.mark.parametrize("f_max", [8000])
@pytest.mark.parametrize("sr", [16000])
@pytest.mark.parametrize("device", ["cuda"])
def test_zerophase_fir_equalizer_with_filterbank_all_scales(
    num_frequency_bins,
    processor_channel,
    flashfftconv,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    device,
):
    if device == "cpu" and flashfftconv:
        return

    filterbank_kwargs = dict(
        num_filters=num_filters,
        scale=scale,
        f_max=f_max,
        sr=sr,
    )

    processor = NewZeroPhaseFIREqualizer(
        num_frequency_bins=num_frequency_bins,
        processor_channel=processor_channel,
        use_filterbank=True,
        flashfftconv=flashfftconv,
        filterbank_kwargs=filterbank_kwargs,
        window=window,
        window_kwargs={},
        eps=1e-7,
    )
    _test_single_processor(processor, device=device)
