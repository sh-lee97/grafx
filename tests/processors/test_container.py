import itertools

import pytest
import torch
from utils import (
    _test_lti_processor,
    _test_single_processor,
    create_empty_parameters_from_shape_dict,
    get_device_setup,
)

from grafx.processors import *

PROCESSORS = [
    BiquadFilter,
    # PoleZeroFilter,
    StateVariableFilter,
    # BaseParametricFilter,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
    BandRejectFilter,
    AllPassFilter,
    # BaseParametricFilter,
    PeakingFilter,
    LowShelf,
    HighShelf,
    Compressor,
    NoiseGate,
    NewZeroPhaseFIREqualizer,
    ParametricEqualizer,
    GraphicEqualizer,
    MultitapDelay,
    STFTMaskedNoiseReverb,
    StereoGain,
    SideGainImager,
    # MonoToStereo,
    # StereoToMidSide,
    # MidSideToStereo,
]

# DEVICE_SETUPS = ["cpu", "cuda", "cuda_flashfftconv"]
DEVICE_SETUPS = ["cuda_flashfftconv"]


def generate_combinations(processors, chain_lengths):
    combinations = []
    for length in chain_lengths:
        combinations.extend(itertools.combinations(processors, length))
    return combinations


PROCESSOR_COMBINATIONS = generate_combinations(PROCESSORS, [2, 3])
# PROCESSOR_COMBINATIONS = PROCESSOR_COMBINATIONS[:1]


def get_processor(processor_cls, flashfftconv):
    try:
        processor = processor_cls(flashfftconv=flashfftconv)
    except:
        processor = processor_cls()
    return processor


# region Fixture


@pytest.fixture(params=DEVICE_SETUPS)
def device_setup(request):
    return get_device_setup(request.param)


@pytest.fixture(params=PROCESSOR_COMBINATIONS)
def processors_combination(request):
    return request.param


# @pytest.fixture
# def processor_instance(request, device_setup):
#     device, flashfftconv = device_setup
#     processor_cls = request.param
#     return get_processor(processor_cls, flashfftconv)


# endregion Fixture


@pytest.mark.parametrize("key", ["gain_reg"])
@pytest.mark.parametrize("processor_cls", PROCESSORS)
def test_gain_staging_regularization(processor_cls, key, device_setup):
    device, flashfftconv = device_setup
    processor = get_processor(processor_cls, flashfftconv)
    wrapped_processor = GainStagingRegularization(processor, key)
    _test_single_processor(wrapped_processor, device=device)


@pytest.mark.parametrize("external_param", [True, False])
@pytest.mark.parametrize("processor_cls", PROCESSORS)
def test_drywet(processor_cls, external_param, device_setup):
    device, flashfftconv = device_setup
    processor = get_processor(processor_cls, flashfftconv)
    processor = DryWet(processor, external_param)

    if external_param:
        batch_size = 16
        drywet_weight = torch.rand(batch_size, 1).to(device)
        additional_params = {"drywet_weight": drywet_weight}
    else:
        additional_params = {}

    _test_single_processor(
        processor,
        device=device,
        additional_params=additional_params,
    )


def test_serial_chain(processors_combination, device_setup):
    device, flashfftconv = device_setup
    processors = {}
    for idx, processor_class in enumerate(processors_combination):
        processor = get_processor(processor_class, flashfftconv)
        processors[f"proc_{idx}"] = processor
        # Alternate between GainStagingRegularization and DryWet for variety
        # if idx % 2 == 0:
        #     processors[f"gain_{idx}"] = GainStagingRegularization(processor_instance)
        # else:
        #     processors[f"drywet_{idx}"] = DryWet(processor_instance)

    processor = SerialChain(processors)
    # processors_kwargs = {key: {} for key in processors.keys()}
    # add drywet_weight to the drywet processors
    # for key in processors.keys():
    #     if "drywet" in key:
    #         processors_kwargs[key]["drywet_weight"] = torch.tensor([0.5], device=device)

    _test_single_processor(processor, device=device)  # , **processors_kwargs)


@pytest.mark.parametrize("activation", ["softmax", "softplus"])
@pytest.mark.parametrize("batch_size", [16])
def test_parallel_mix(
    processors_combination,
    activation,
    batch_size,
    device_setup,
):
    device, flashfftconv = device_setup
    processors = {}
    for idx, processor_class in enumerate(processors_combination):
        processor = get_processor(processor_class, flashfftconv)
        processors[f"proc_{idx}"] = processor
    processor = ParallelMix(processors, activation=activation)

    _test_single_processor(
        processor,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    test_parallel_mix(PROCESSOR_COMBINATIONS[0], "softmax", 16, "cuda")
