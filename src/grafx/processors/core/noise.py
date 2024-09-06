import numpy as np
import torch
from scipy import signal
from scipy.signal import butter, sosfilt, sosfiltfilt

from grafx.processors.core.scale import from_scale, to_scale


def apply_linkwitz_riley(
    input_audio,
    num_bands=2,
    f_min=40,
    f_max=None,
    scale="bark_traunmuller",
    sr=44100,
    zerophase=True,
    order=2,
):
    r"""
    Apply Linkwitz-Riley crossover filter and split the audio into multiple bands.
    """
    s_min, s_max = to_scale(f_min, scale), to_scale(f_max, scale)
    num_bands = num_bands * 2 - 1
    s_breaks = np.linspace(s_min, s_max, num_bands)[1::2]
    f_breaks = from_scale(s_breaks, scale)

    lpf_soses = [
        butter(order, freq, "lowpass", fs=sr, output="sos") for freq in f_breaks
    ]
    hpf_soses = [
        butter(order, freq, "highpass", fs=sr, output="sos") for freq in f_breaks
    ]

    filtered_signals = []
    for lpf_sos, hpf_sos in zip(lpf_soses, hpf_soses):
        if zerophase:
            lpfed = sosfiltfilt(lpf_sos, input_audio)
            hpfed = sosfiltfilt(hpf_sos, input_audio)
        else:
            lpfed = sosfilt(lpf_sos, sosfilt(lpf_sos, input_audio))
            hpfed = sosfilt(hpf_sos, sosfilt(hpf_sos, input_audio))
        input_audio = hpfed
        filtered_signals.append(lpfed)
    filtered_signals.append(hpfed)
    filtered_signals = np.stack(filtered_signals, 1)
    return filtered_signals


def get_filtered_noise(
    fir_len,
    num_channels=1,
    num_bands=12,
    f_min=31.5,
    f_max=16000,
    scale="log",
    sr=44100,
    zerophase=True,
    order=2,
):
    noise = np.random.rand(num_channels, fir_len)
    noise = 2 * noise - 1
    filtered_noise = apply_linkwitz_riley(
        noise,
        num_bands=num_bands,
        f_min=f_min,
        f_max=f_max,
        scale=scale,
        sr=sr,
        zerophase=zerophase,
        order=order,
    )
    filtered_noise = torch.from_numpy(filtered_noise).float()
    return filtered_noise


def octave_band_filterbank(num_taps: int, sample_rate: float):
    # create octave-spaced bandpass filters
    bands = [
        31.5,
        63,
        125,
        250,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
    ]
    num_bands = len(bands) + 2
    filts = []  # storage for computed filter coefficients

    # lowest band is a lowpass
    filt = signal.firwin(
        num_taps,
        12,
        fs=sample_rate,
    )
    filt = torch.from_numpy(filt.astype("float32"))
    filt = torch.flip(filt, dims=[0])
    filts.append(filt)

    for fc in bands:
        f_min = fc / np.sqrt(2)
        f_max = fc * np.sqrt(2)
        f_max = np.clip(f_max, a_min=0, a_max=(sample_rate / 2) * 0.999)
        filt = signal.firwin(
            num_taps,
            [f_min, f_max],
            fs=sample_rate,
            pass_zero=False,
        )
        filt = torch.from_numpy(filt.astype("float32"))
        filt = torch.flip(filt, dims=[0])
        filts.append(filt)

    # highest is a highpass
    filt = signal.firwin(num_taps, 18000, fs=sample_rate, pass_zero=False)
    filt = torch.from_numpy(filt.astype("float32"))
    filt = torch.flip(filt, dims=[0])
    filts.append(filt)

    filts = torch.stack(filts, dim=0)  # stack coefficients into single filter
    filts = filts.unsqueeze(1)  # shape: num_bands x 1 x num_taps

    return filts
