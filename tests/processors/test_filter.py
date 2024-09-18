import pytest
import torch
from utils import _save_audio_and_mel, _test_single_processor

import conftest
from grafx.processors import *
from grafx.processors.core.iir import IIRFilter

# region Fixture


@pytest.fixture(params=[1, 4, 16])
def num_filters(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=["fsm", "lfilter"])
def backend(request):
    return request.param


@pytest.fixture(params=[4000, 8192, 16384])
def fsm_fir_len(request):
    return request.param


@pytest.fixture(params=[True, False])
def flashfftconv(request):
    return request.param


@pytest.fixture(params=[True, False])
def fsm_regularization(request):
    return request.param


@pytest.fixture(params=[0, 0.01, 1])
def std(request):
    return request.param


# endregion Fixture


@conftest.quant_test
@pytest.mark.parametrize(
    "processor_cls",
    [
        BiquadFilter,
        StateVariableFilter,
        PoleZeroFilter,
        LowPassFilter,
        BandPassFilter,
        HighPassFilter,
        BandRejectFilter,
        AllPassFilter,
        PeakingFilter,
        LowShelf,
        HighShelf,
    ],
)
def test_filter_quantitative(processor_cls, std, batch_size=4):
    print(processor_cls.__name__)

    processor = processor_cls(backend="fsm", flashfftconv=True)
    _save_audio_and_mel(
        processor, "filter", device="cuda", batch_size=batch_size, std=std
    )


@pytest.mark.parametrize("normalized", [False])
def test_biquad_filter(
    device, num_filters, normalized, backend, fsm_fir_len, flashfftconv
):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = BiquadFilter(
        num_filters=num_filters,
        normalized=normalized,
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_svf(device, num_filters, backend, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = StateVariableFilter(
        num_filters=num_filters,
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_lpf(device, backend, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = LowPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_bpf(device, backend, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = BandPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_hpf(device, backend, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = HighPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_brf(device, backend, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = BandRejectFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_apf(device, backend, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = AllPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_peak(device, backend, num_filters, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = PeakingFilter(
        backend=backend,
        num_filters=num_filters,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_ls(device, backend, num_filters, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = LowShelf(
        backend=backend,
        num_filters=num_filters,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_hs(device, backend, num_filters, fsm_fir_len, flashfftconv):
    if device == "cpu" and backend == "fsm" and flashfftconv:
        pytest.skip("Skipping test due to known issue with this configuration.")

    processor = HighShelf(
        backend=backend,
        num_filters=num_filters,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


def test_ssm_lfilter_equivalence():
    batch_size = 7
    num_filters = 6
    num_channels = 5
    T = 10000

    x = torch.randn(batch_size, num_channels, T).double()
    Bs = torch.randn(batch_size, num_channels, num_filters, 3).double()
    a1 = torch.rand(batch_size, num_channels, num_filters).double() * 4 - 2
    a2 = ((torch.rand_like(a1) * 2 - 1) * (2 - a1.abs()) + a1.abs()) * 0.5
    As = torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    baseline_processor = IIRFilter(backend="lfilter")
    ssm_processor = IIRFilter(backend="ssm")

    target = baseline_processor(x, Bs, As)
    output = ssm_processor(x, Bs, As)

    assert torch.allclose(target, output), (target - output).abs().max()


if __name__ == "__main__":
    test_filter_quantitative(BiquadFilter, "cuda", "fsm", False)
