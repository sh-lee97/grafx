import pytest
from utils import _save_audio_mel, _test_single_processor, get_device_setup

from grafx.processors import *

# region Fixture


@pytest.fixture(params=[256, 1024])
def num_frequency_bins(request):
    return request.param


@pytest.fixture(params=["mono", "stereo", "midside"])
def processor_channel(request):
    return request.param


@pytest.fixture(
    params=["hann", "hamming", "blackman", "bartlett", "kaiser", "boxcar", None]
)
def window(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda", "cuda_flashfftconv"])
def setup(request):
    return request.param


@pytest.fixture(scope="session")
def processor_list():
    return []


@pytest.fixture(scope="session", autouse=True)
def save_audio_mel_once(processor_list):
    yield
    if processor_list:
        print("Processors in the list:")
        print([name for name, _ in processor_list])
        for name, processor in processor_list:
            _save_audio_mel(processor, "eq", device="cuda", name=name)


def add_processor_if_unique(name, processor, processor_list):
    # Check if the name already exists in the list
    if not any(existing_name == name for existing_name, _ in processor_list):
        processor_list.append((name, processor))


# endregion Fixture


def test_zerophase_fir_equalizer_without_filterbank(
    num_frequency_bins, processor_channel, window, setup, processor_list
):
    device, flashfftconv = get_device_setup(setup)

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
    name = "ZeroPhaseFIREqualizer_Without_Filterbank"
    add_processor_if_unique(name, processor, processor_list)


@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("num_filters", [50])
@pytest.mark.parametrize("scale", ["bark_traunmuller", "mel_slaney", "linear", "log"])
@pytest.mark.parametrize("f_max", [8000, 22050])
@pytest.mark.parametrize("sr", [16000])
def test_zerophase_fir_equalizer_with_filterbank(
    num_frequency_bins,
    processor_channel,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    setup,
    processor_list,
):
    device, flashfftconv = get_device_setup(setup)

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
    name = "ZeroPhaseFIREqualizer_With_Filterbank"
    add_processor_if_unique(name, processor, processor_list)


@pytest.mark.parametrize("num_frequency_bins", [1024])
@pytest.mark.parametrize("processor_channel", ["stereo"])
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
def test_zerophase_fir_equalizer_with_filterbank_all_scales(
    num_frequency_bins,
    processor_channel,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    setup,
    processor_list,
):
    device, flashfftconv = get_device_setup(setup)

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
    name = "ZeroPhaseFIREqualizer_With_Filterbank_All_Scales"
    add_processor_if_unique(name, processor, processor_list)


def test_newzerophase_fir_equalizer_without_filterbank(
    num_frequency_bins, processor_channel, window, setup, processor_list
):
    device, flashfftconv = get_device_setup(setup)

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
    name = "NewZeroPhaseFIREqualizer_Without_Filterbank"
    add_processor_if_unique(name, processor, processor_list)


@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("num_filters", [50])
@pytest.mark.parametrize("scale", ["bark_traunmuller", "mel_slaney", "linear", "log"])
@pytest.mark.parametrize("f_max", [8000, 22050])
@pytest.mark.parametrize("sr", [16000])
def test_newzerophase_fir_equalizer_with_filterbank(
    num_frequency_bins,
    processor_channel,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    setup,
    processor_list,
):
    device, flashfftconv = get_device_setup(setup)

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
    name = "NewZeroPhaseFIREqualizer_With_Filterbank"
    add_processor_if_unique(name, processor, processor_list)


@pytest.mark.parametrize("num_frequency_bins", [1024])
@pytest.mark.parametrize("processor_channel", ["stereo"])
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
def test_newzerophase_fir_equalizer_with_filterbank_all_scales(
    num_frequency_bins,
    processor_channel,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    setup,
    processor_list,
):
    device, flashfftconv = get_device_setup(setup)

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
    name = "NewZeroPhaseFIREqualizer_With_Filterbank_All_Scales"
    add_processor_if_unique(name, processor, processor_list)


@pytest.mark.parametrize("num_filters", [10, 20])
@pytest.mark.parametrize("use_shelving_filters", [True, False])
def test_parametric_equalizer(
    num_filters, processor_channel, use_shelving_filters, setup, processor_list
):
    device, flashfftconv = get_device_setup(setup)

    processor = ParametricEqualizer(
        num_filters=num_filters,
        processor_channel=processor_channel,
        use_shelving_filters=use_shelving_filters,
        flashfftconv=flashfftconv,
    )
    _test_single_processor(processor, device=device)
    name = "ParametricEqualizer"
    add_processor_if_unique(name, processor, processor_list)


@pytest.mark.parametrize("scale", ["bark", "third_octave"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4096, 8192])
def test_graphic_equalizer(
    scale, processor_channel, backend, fsm_fir_len, setup, processor_list
):
    device, flashfftconv = get_device_setup(setup)

    processor = GraphicEqualizer(
        scale=scale,
        processor_channel=processor_channel,
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
    )
    _test_single_processor(processor, device=device)
    name = "GraphicEqualizer"
    add_processor_if_unique(name, processor, processor_list)


# def test_geq():
#     import matplotlib.pyplot as plt
#
#     from grafx.processors.core.iir import delay, iir_fsm
#
#     scale = "bark"
#     match scale:
#         case "bark":
#             FC = FC_BARK
#         case "third_octave":
#             FC = FC_THIRD_OCTAVE
#         case _:
#             raise ValueError(f"Unsupported scale: {scale}")
#     geq = GraphicEqualizerBiquad(
#         scale="bark",
#     )
#     Bs, As = geq(torch.ones(24))
#     arange = torch.arange(3)
#     fir_length = 2**15
#     delays = delay(arange, fir_length=fir_length)
#     geq_fsm = iir_fsm(Bs, As, delays=delays)
#     geq_fsm_magnitude = torch.abs(geq_fsm)
#     geq_fsm_db = torch.log(geq_fsm_magnitude + 1e-7)
#     max_db = geq_fsm_db.max()
#
#     faxis = torch.linspace(0, 22050, fir_length // 2 + 1)
#     fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#     for i in range(len(Bs)):
#         ax.plot(faxis, geq_fsm_db[i])
#     for i in range(len(Bs)):
#         ax.axvline(FC[i], color="black", linestyle="--")
#
#     ax.set_xlim(10, 22050)
#     ax.axhline(max_db * 0.4, color="red", linestyle="--")
#     ax.set_xscale("symlog", linthresh=10, linscale=0.1)
#
#     fig.savefig("geq.pdf", bbox_inches="tight")
#
#
# if __name__ == "__main__":
#     scale = "bark"
#     match scale:
#         case "bark":
#             FC = FC_BARK
#             FB = FB_BARK
#         case "third_octave":
#             FC = FC_THIRD_OCTAVE
#             FB = FB_THIRD_OCTAVE
#         case _:
#             raise ValueError(f"Unsupported scale: {scale}")
#
#     # FB_calc =
