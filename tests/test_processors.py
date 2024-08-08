import torch

from grafx.processors import (
    ApproxCompressor,
    ApproxNoiseGate,
    MidSideFilteredNoiseReverb,
    SideGainImager,
    StereoGain,
    StereoMultitapDelay,
    ZeroPhaseFIREqualizer,
)


def _test_single_processor(
    processor,
    batch_size=16,
    num_channels=2,
    audio_len=2 ** 17,
    device="cpu",
):
    processor = processor.to(device)
    parameter_size = processor.parameter_size()
    parameter_size = {k: ((v,) if isinstance(v, int) else v) for k, v in parameter_size.items()}
    parameters = {k: torch.randn(batch_size, *v, device=device) for k, v in parameter_size.items()} 
    input_signal = torch.randn(batch_size, num_channels, audio_len, device=device)
    output = processor(input_signal, **parameters)
    if isinstance(output, tuple):
        output_signal, intermediates = output
    else:
        output_signal = output
    assert output_signal.ndim == 3
    assert output_signal.shape[0] == batch_size
    assert output_signal.shape[2] == audio_len
    assert output_signal.device == input_signal.device
    assert output_signal.dtype == input_signal.dtype
    assert ~output_signal.isnan().any()


def test_approx_compressor():
    processor = ApproxCompressor(flashfftconv=False)
    _test_single_processor(processor, device="cpu")

def test_approx_noise_gate():
    processor = ApproxNoiseGate(flashfftconv=False)
    _test_single_processor(processor, device="cpu")

def test_mid_side_filtered_noise_reverb():
    processor = MidSideFilteredNoiseReverb(flashfftconv=False)
    _test_single_processor(processor, device="cpu")

def test_side_gain_imager():
    processor = SideGainImager()
    _test_single_processor(processor, device="cpu")

def test_stereo_gain():
    processor = StereoGain()
    _test_single_processor(processor, device="cpu")

def test_stereo_multitap_delay():
    processor = StereoMultitapDelay(flashfftconv=False)
    _test_single_processor(processor, device="cpu")

def test_zero_phase_fir_equalizer():
    processor = ZeroPhaseFIREqualizer()
    _test_single_processor(processor, device="cpu")
