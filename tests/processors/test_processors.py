from pprint import pprint

import matplotlib.pyplot as plt
import pytest
import torch
from utils import _test_lti_processor, _test_single_processor

from grafx.processors import *
from grafx.utils import create_empty_parameters_from_shape_dict


def test_approx_compressor():
    processor = ApproxCompressor(flashfftconv=False)
    _test_single_processor(processor, device="cpu")


def test_approx_noise_gate():
    processor = ApproxNoiseGate(flashfftconv=False)
    _test_single_processor(processor, device="cpu")


def test_mid_side_filtered_noise_reverb():
    processor = STFTMaskedNoiseReverb(flashfftconv=False)
    _test_single_processor(processor, device="cpu")


def test_side_gain_imager():
    processor = SideGainImager()
    _test_single_processor(processor, device="cpu")


def test_stereo_gain():
    processor = StereoGain()
    _test_single_processor(processor, device="cpu")


def test_stereo_multitap_delay():
    processor = MultitapDelay(flashfftconv=False)
    _test_single_processor(processor, device="cpu")


def test_zero_phase_fir_equalizer():
    processor = ZeroPhaseFIREqualizer()
    _test_single_processor(processor, device="cpu")
    _test_lti_processor(processor, device="cpu")


def test_mono_to_stereo():
    processor = MonoToStereo()
    _test_single_processor(processor, num_channels=1, device="cpu")


# def test_one_pole_iir_compressor():
#    processor = OnePoleIIRCompressor(flashfftconv=False)
#    _test_single_processor(processor, device="cpu")
#
# def test_ballistics_compressor():
#    processor = BallisticsCompressor()
#    _test_single_processor(processor, device="cpu")


@pytest.mark.parametrize("energy_smoother", ["iir", "ballistics", None])
@pytest.mark.parametrize("gain_smoother", ["iir", "ballistics", None])
@pytest.mark.parametrize("knee", ["hard", "quadratic", "exponential"])
def test_compressor(energy_smoother, gain_smoother, knee):
    if energy_smoother is None and gain_smoother is None:
        return

    processor = Compressor(
        energy_smoother=energy_smoother,
        gain_smoother=gain_smoother,
        knee=knee,
    )
    pprint(processor.parameter_size())
    _test_single_processor(processor, device="cuda")


def test_serial_chain():
    processor = SerialChain(
        {
            "compressor": ApproxCompressor(flashfftconv=False),
            "reverb": STFTMaskedNoiseReverb(flashfftconv=False),
        }
    )
    print(processor.parameter_size())
    _test_single_processor(processor)


def test_graphic_eq():
    processor = GraphicEqualizer(backend="lfilter")
    _test_single_processor(processor, device="cuda")
    _test_lti_processor(processor, device="cuda")


if __name__ == "__main__":
    # test_zero_phase_fir_equalizer()
    # test_mid_side_filtered_noise_reverb()
    # test_lowpass_filter()
    # pass
    # test_biquad_filter()
    # test_compressor()
    # test_serial_chain()
    test_graphic_eq()
    pass
