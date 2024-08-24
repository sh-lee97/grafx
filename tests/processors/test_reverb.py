import pytest
import torch
from utils import _test_single_processor

from grafx.processors.reverb import STFTMaskedNoiseReverb


@pytest.mark.parametrize("ir_len", [30000, 60000])
@pytest.mark.parametrize(
    "reverb_channel", ["mono", "stereo", "midside", "pseudo_midside"]
)
@pytest.mark.parametrize("n_fft", [256, 384])
@pytest.mark.parametrize("hop_length", [128, 192])
@pytest.mark.parametrize("fixed_noise", [True, False])
@pytest.mark.parametrize("gain_envelope", [True, False])
@pytest.mark.parametrize("flashfftconv", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_midside_filtered_noise_reverb(
    ir_len,
    reverb_channel,
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
        reverb_channel=reverb_channel,
        n_fft=n_fft,
        hop_length=hop_length,
        fixed_noise=fixed_noise,
        gain_envelope=gain_envelope,
        flashfftconv=flashfftconv,
    ).to(device)

    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("ir_len", [60000])
@pytest.mark.parametrize("reverb_channel", ["pseudo_midside"])
@pytest.mark.parametrize("n_fft", [384])
@pytest.mark.parametrize("hop_length", [192])
@pytest.mark.parametrize("gain_envelope", [True, False])
def test_midside_filtered_noise_reverb_parameter_size(
    ir_len, reverb_channel, n_fft, hop_length, gain_envelope
):
    processor = STFTMaskedNoiseReverb(
        ir_len=ir_len,
        reverb_channel=reverb_channel,
        n_fft=n_fft,
        hop_length=hop_length,
        gain_envelope=gain_envelope,
    )

    param_size = processor.parameter_size()
    assert "init_log_magnitude" in param_size
    assert "delta_log_magnitude" in param_size
    assert param_size["init_log_magnitude"] == (2, processor.num_bins)
    assert param_size["delta_log_magnitude"] == (2, processor.num_bins)

    if gain_envelope:
        assert "gain_env_log_magnitude" in param_size
        assert param_size["gain_env_log_magnitude"] == (2, processor.num_frames)
