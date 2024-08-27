import pytest
import torch
from utils import _test_single_processor

from grafx.processors.reverb import *


@pytest.mark.parametrize("ir_len", [30000, 60000])
@pytest.mark.parametrize(
    "processor_channel", ["mono", "stereo", "midside", "pseudo_midside"]
)
@pytest.mark.parametrize("n_fft", [256, 384])
@pytest.mark.parametrize("hop_length", [128, 192])
@pytest.mark.parametrize("fixed_noise", [True, False])
@pytest.mark.parametrize("gain_envelope", [True, False])
@pytest.mark.parametrize("flashfftconv", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stft_masked_noise_reverb(
    ir_len,
    processor_channel,
    n_fft,
    hop_length,
    fixed_noise,
    gain_envelope,
    flashfftconv,
    device,
):
    if device == "cpu" and flashfftconv:
        return

    processor = STFTMaskedNoiseReverb(
        ir_len=ir_len,
        processor_channel=processor_channel,
        n_fft=n_fft,
        hop_length=hop_length,
        fixed_noise=fixed_noise,
        gain_envelope=gain_envelope,
        flashfftconv=flashfftconv,
    ).to(device)

    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("ir_len", [30000, 60000])
@pytest.mark.parametrize("num_bands", [2, 10])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("scale", ["bark_traunmuller"])
@pytest.mark.parametrize("zerophase", [True])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("filtered_noise", ["pseudo-random"])
@pytest.mark.parametrize("use_fade_in", [True, False])
@pytest.mark.parametrize("flashfftconv", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_filtered_noise_shaping_reverb(
    ir_len,
    num_bands,
    processor_channel,
    scale,
    zerophase,
    order,
    filtered_noise,
    use_fade_in,
    flashfftconv,
    device,
):
    if device == "cpu" and flashfftconv:
        return

    processor = FilteredNoiseShapingReverb(
        ir_len=ir_len,
        num_bands=num_bands,
        processor_channel=processor_channel,
        scale=scale,
        zerophase=zerophase,
        order=order,
        noise_randomness=filtered_noise,
        use_fade_in=use_fade_in,
        flashfftconv=flashfftconv,
    ).to(device)

    _test_single_processor(processor, device=device)


if __name__ == "__main__":
    test_filtered_noise_shaping_reverb(
        ir_len=30000,
        num_bands=12,
        processor_channel="mono",
        scale="log",
        zerophase=True,
        order=2,
        filtered_noise="pseudo-random",
        use_fade_in=True,
        flashfftconv=False,
        device="cuda",
    )
