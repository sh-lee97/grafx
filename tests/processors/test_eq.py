import pytest
from utils import _test_lti_processor, _test_single_processor, get_device_setup

from grafx.processors import *

DEVICE_SETUPS = ["cpu", "cuda", "cuda_flashfftconv"]


@pytest.mark.parametrize("num_frequency_bins", [256, 1024])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize(
    "window", ["hann", "hamming", "blackman", "bartlett", "kaiser", "boxcar", None]
)
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_zerophase_fir_equalizer_without_filterbank(
    num_frequency_bins, processor_channel, window, setup
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


@pytest.mark.parametrize("num_frequency_bins", [256, 1024])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("num_filters", [50])
@pytest.mark.parametrize("scale", ["bark_traunmuller", "mel_slaney", "linear", "log"])
@pytest.mark.parametrize("f_max", [8000, 22050])
@pytest.mark.parametrize("sr", [16000])
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_zerophase_fir_equalizer_with_filterbank(
    num_frequency_bins, processor_channel, window, num_filters, scale, f_max, sr, setup
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
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_zerophase_fir_equalizer_with_filterbank_all_scales(
    num_frequency_bins,
    processor_channel,
    window,
    num_filters,
    scale,
    f_max,
    sr,
    setup,
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


@pytest.mark.parametrize("num_frequency_bins", [256, 1024])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize(
    "window", ["hann", "hamming", "blackman", "bartlett", "kaiser", "boxcar", None]
)
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_newzerophase_fir_equalizer_without_filterbank(
    num_frequency_bins,
    processor_channel,
    window,
    setup,
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


@pytest.mark.parametrize("num_frequency_bins", [256, 1024])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("num_filters", [50])
@pytest.mark.parametrize("scale", ["bark_traunmuller", "mel_slaney", "linear", "log"])
@pytest.mark.parametrize("f_max", [8000, 22050])
@pytest.mark.parametrize("sr", [16000])
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_newzerophase_fir_equalizer_with_filterbank(
    num_frequency_bins, processor_channel, window, num_filters, scale, f_max, sr, setup
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
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_newzerophase_fir_equalizer_with_filterbank_all_scales(
    num_frequency_bins, processor_channel, window, num_filters, scale, f_max, sr, setup
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


@pytest.mark.parametrize("num_filters", [10, 20])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("use_shelving_filters", [True, False])
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_parametric_equalizer(
    num_filters,
    processor_channel,
    use_shelving_filters,
    setup,
):
    device, flashfftconv = get_device_setup(setup)

    processor = ParametricEqualizer(
        num_filters=num_filters,
        processor_channel=processor_channel,
        use_shelving_filters=use_shelving_filters,
        flashfftconv=flashfftconv,
    )
    _test_single_processor(processor, device=device)


@pytest.mark.parametrize("scale", ["bark", "third_octave"])
@pytest.mark.parametrize("processor_channel", ["mono", "stereo", "midside"])
@pytest.mark.parametrize("backend", ["fsm", "lfilter"])
@pytest.mark.parametrize("fsm_fir_len", [4096, 8192])
@pytest.mark.parametrize("setup", DEVICE_SETUPS)
def test_graphic_equalizer(scale, processor_channel, backend, fsm_fir_len, setup):
    device, flashfftconv = get_device_setup(setup)

    processor = GraphicEqualizer(
        scale=scale,
        processor_channel=processor_channel,
        backend=backend,
        fsm_fir_len=fsm_fir_len,
        flashfftconv=flashfftconv,
    )
    _test_single_processor(processor, device=device)


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
