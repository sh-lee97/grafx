import pytest
from utils import _test_lti_processor, _test_single_processor

from grafx.processors import *


@pytest.mark.parametrize("num_filters", [1, 4, 16])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_biquad_filter(
    device,
    num_filters,
    normalized,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = BiquadFilter(
        num_filters=num_filters,
        normalized=normalized,
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("num_filters", [1, 4, 16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_svf(
    device,
    num_filters,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = StateVariableFilter(
        num_filters=num_filters,
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_lpf(
    device,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = LowPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_bpf(
    device,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = BandPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_hpf(
    device,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = HighPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_brf(
    device,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = BandRejectFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_apf(
    device,
    backend,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = AllPassFilter(
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_filters", [1, 4, 16])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_peak(
    device,
    backend,
    num_filters,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = PeakingFilter(
        backend=backend,
        num_filters=num_filters,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_filters", [1, 4, 16])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_ls(
    device,
    backend,
    num_filters,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = LowShelf(
        backend=backend,
        num_filters=num_filters,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_filters", [1, 4, 16])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4000, 8192, 16384])
@pytest.mark.parametrize("fsm_flashfftconv", [True, False])
def test_hs(
    device,
    backend,
    num_filters,
    fsm_fir_len,
    fsm_flashfftconv,
    # fsm_regularization,
):
    if device == "cpu" and backend == "fsm" and fsm_flashfftconv:
        return

    processor = HighShelf(
        backend=backend,
        num_filters=num_filters,
        fsm_fir_len=fsm_fir_len,
        fsm_flashfftconv=fsm_flashfftconv,
        fsm_regularization=False,
    )
    _test_single_processor(processor, device=device)
