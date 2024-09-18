import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn.functional as F
from utils import _save_audio_and_mel, _test_single_processor

import conftest
from grafx.processors import *

# region Fixture


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=[True, False])
def flashfftconv(request):
    return request.param


@pytest.fixture(params=[0, 0.01, 1])
def std(request):
    return request.param


# endregion Fixture


@conftest.quant_test
@pytest.mark.parametrize("processor_cls", [Compressor, NoiseGate])
def test_dynamics_quantitative(processor_cls, std, batch_size=4):
    print(processor_cls.__name__)
    processor = processor_cls(flashfftconv=True)
    _save_audio_and_mel(
        processor, "dynamics", device="cuda", batch_size=batch_size, std=std
    )


@pytest.mark.parametrize("energy_smoother", ["iir", "ballistics", None])
@pytest.mark.parametrize("gain_smoother", ["iir", "ballistics", None])
@pytest.mark.parametrize("gain_smooth_in_log", [True, False])
@pytest.mark.parametrize("knee", ["hard", "quadratic", "exponential"])
@pytest.mark.parametrize("iir_len", [16384, 32768])
def test_compressor(
    energy_smoother,
    gain_smoother,
    gain_smooth_in_log,
    knee,
    iir_len,
    flashfftconv,
    device,
):
    if device == "cpu" and flashfftconv:
        return
    if energy_smoother is None and gain_smoother is None:
        return

    processor = Compressor(
        energy_smoother=energy_smoother,
        gain_smoother=gain_smoother,
        gain_smooth_in_log=gain_smooth_in_log,
        knee=knee,
        iir_len=iir_len,
        flashfftconv=flashfftconv,
    )
    _test_single_processor(processor, device="cuda")


@pytest.mark.parametrize("energy_smoother", ["iir", "ballistics", None])
@pytest.mark.parametrize("gain_smoother", ["iir", "ballistics", None])
@pytest.mark.parametrize("gain_smooth_in_log", [True, False])
@pytest.mark.parametrize("knee", ["hard", "quadratic", "exponential"])
@pytest.mark.parametrize("iir_len", [16384, 32768])
def test_noisegate(
    energy_smoother,
    gain_smoother,
    gain_smooth_in_log,
    knee,
    iir_len,
    flashfftconv,
    device,
):
    if device == "cpu" and flashfftconv:
        return
    if energy_smoother is None and gain_smoother is None:
        return

    processor = NoiseGate(
        energy_smoother=energy_smoother,
        gain_smoother=gain_smoother,
        gain_smooth_in_log=gain_smooth_in_log,
        knee=knee,
        iir_len=iir_len,
        flashfftconv=flashfftconv,
    )
    _test_single_processor(processor, device="cuda")


def test_knee_visualization():
    # Parameters for the compressor
    T = torch.tensor([0])  # Threshold (in dB)
    R = torch.tensor([4])  # Compression Ratio

    # Input levels in dB
    x = torch.linspace(-60, 20, 400)

    cmap = plt.get_cmap("jet")

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for k_exp in torch.exp(torch.linspace(-2, 2, 20)):
        gain_exp = Compressor.gain_exp_knee(x, T, R, k_exp)
        gain_sigmoid = NoiseGate.gain_exp_knee(x, T, R, k_exp)
        ax[0].plot(x, gain_exp)
        ax[1].plot(x, gain_sigmoid)

    ax[0].plot(x, x, "--", color="gray", label="No Compression")
    ax[1].plot(x, x, "--", color="gray", label="No Compression")

    # Labels and title
    for x in ax:
        x.set_xlabel("Itorchut Level (dB)")
        x.set_ylabel("Output Level (dB)")
        x.axvline(T, color="red", linestyle="--", label="Threshold")
        x.legend()
        x.grid(True)
        x.set_ylim(-60, 20)

    fig.savefig("tests/outputs/knee_test.pdf", bbox_inches="tight")
