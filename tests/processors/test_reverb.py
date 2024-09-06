import conftest
import pytest
from utils import _save_audio_and_mel, _test_single_processor

from grafx.processors.reverb import *

# region Fixture


@pytest.fixture(params=[30000, 60000])
def ir_len(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=[True, False])
def flashfftconv(request):
    return request.param


@pytest.fixture(params=[1])  # [-1, 0, 0.01, 1]
def std(request):
    return request.param


# endregion Fixture


@conftest.quant_test
@pytest.mark.parametrize(
    "processor_cls", [STFTMaskedNoiseReverb, FilteredNoiseShapingReverb]
)
def test_reverb_quantitative(processor_cls, std, batch_size=1):
    print(processor_cls.__name__)
    processor = processor_cls(flashfftconv=True)
    _save_audio_and_mel(
        processor, "reverb", device="cuda", batch_size=batch_size, std=std
    )


# @pytest.mark.parametrize(
#    "processor_channel", ["mono", "stereo", "midside", "pseudo_midside"]
# )
# @pytest.mark.parametrize("n_fft", [256, 384])
# @pytest.mark.parametrize("hop_length", [128, 192])
# @pytest.mark.parametrize("fixed_noise", [True, False])
# @pytest.mark.parametrize("gain_envelope", [True, False])
# def test_stft_masked_noise_reverb(
#    ir_len,
#    processor_channel,
#    n_fft,
#    hop_length,
#    fixed_noise,
#    gain_envelope,
#    flashfftconv,
#    device,
# ):
#    if device == "cpu" and flashfftconv:
#        pytest.skip("Skipping test due to known issue with this configuration.")
#
#    processor = STFTMaskedNoiseReverb(
#        ir_len=ir_len,
#        processor_channel=processor_channel,
#        n_fft=n_fft,
#        hop_length=hop_length,
#        fixed_noise=fixed_noise,
#        gain_envelope=gain_envelope,
#        flashfftconv=flashfftconv,
#    ).to(device)
#
#    _test_single_processor(processor, device=device)
#
#
# @pytest.mark.parametrize("num_bands", [2, 10])
# @pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
# @pytest.mark.parametrize("scale", ["bark_traunmuller"])
# @pytest.mark.parametrize("zerophase", [True])
# @pytest.mark.parametrize("order", [2, 4])
# @pytest.mark.parametrize("filtered_noise", ["pseudo-random"])
# @pytest.mark.parametrize("use_fade_in", [True, False])
# def test_filtered_noise_shaping_reverb(
#    ir_len,
#    num_bands,
#    processor_channel,
#    scale,
#    zerophase,
#    order,
#    filtered_noise,
#    use_fade_in,
#    flashfftconv,
#    device,
# ):
#    if device == "cpu" and flashfftconv:
#        pytest.skip("Skipping test due to known issue with this configuration.")
#
#    processor = FilteredNoiseShapingReverb(
#        ir_len=ir_len,
#        num_bands=num_bands,
#        processor_channel=processor_channel,
#        scale=scale,
#        zerophase=zerophase,
#        order=order,
#        noise_randomness=filtered_noise,
#        use_fade_in=use_fade_in,
#        flashfftconv=flashfftconv,
#    ).to(device)
#
#    _test_single_processor(processor, device=device)
#
#
# if __name__ == "__main__":
#    test_filtered_noise_shaping_reverb(
#        ir_len=30000,
#        num_bands=12,
#        processor_channel="stereo",
#        scale="log",
#        zerophase=True,
#        order=2,
#        filtered_noise="pseudo-random",
#        use_fade_in=True,
#        flashfftconv=False,
#        device="cuda",
#    )
#
