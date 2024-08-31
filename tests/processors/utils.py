import os
from pprint import pprint

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from einops import repeat

from grafx.processors import *
from grafx.utils import create_empty_parameters_from_shape_dict

opj = os.path.join


def _test_lti_processor(
    processor,
    batch_size=1,
    audio_len=2**17,
    device="cpu",
):
    processor = processor.to(device)
    parameters = create_empty_parameters_from_shape_dict(
        processor.parameter_size(),
        num_nodes=batch_size,
        device=device,
        std=1,
    )
    input_signal = torch.zeros(batch_size, 1, audio_len, device=device)
    input_signal[0, 0, 0] = 1
    output = processor(input_signal, **parameters)

    if isinstance(output, tuple):
        output_signal, _ = output
    else:
        output_signal = output
    assert ~output_signal.isnan().any()

    output_signal = output_signal[0][0].detach().cpu().float()

    fig, ax = plt.subplots()
    magnitude = 20 * torch.log10(torch.abs(torch.fft.rfft(output_signal)))
    ax.plot(magnitude)
    name = type(processor)
    fig.savefig(f"tests/outputs/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _test_single_processor(
    processor,
    batch_size=16,
    num_channels=2,
    audio_len=2**17,
    device="cpu",
    additional_params={},  #
):
    processor = processor.to(device)
    parameter_size = processor.parameter_size()
    parameter_size = {
        k: ((v,) if isinstance(v, int) else v) for k, v in parameter_size.items()
    }
    parameters = create_empty_parameters_from_shape_dict(
        parameter_size, num_nodes=batch_size, device=device
    )
    parameters.update(additional_params)  #

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
    assert (output_signal.dtype == input_signal.dtype) or (
        output_signal.dtype == torch.bfloat16
    )
    assert ~output_signal.isnan().any()
    assert ~output_signal.isinf().any()


def get_device_setup(setup):
    if setup == "cpu":
        return setup, False
    elif setup == "cuda":
        return setup, False
    elif setup == "cuda_flashfftconv":
        return "cuda", True
    else:
        assert False


def load_audio_files():
    audio_files = {
        "speech": "tests/samples/speech.wav",
        "drums": "tests/samples/drums.wav",
        "bass": "tests/samples/bass.wav",
        "singing": "tests/samples/singing.wav",
        "music": "tests/samples/music.wav",
        "other": "tests/samples/other.wav",
    }
    audio_data = {}
    for key, file_path in audio_files.items():
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[..., : 2**17]
        audio_data[key] = (waveform, sample_rate)
    return audio_data


def save_output(processor_class, subgroup, output, sample_rate, test_name):
    output_dir = f"tests/outputs/{subgroup}/{processor_class}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = opj(output_dir, f"{test_name}.wav")
    torchaudio.save(output_file, output[0], sample_rate)


def logmelspec(x, sr):
    X = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=160)
    return 20 * np.log10(np.abs(X) + 1e-5)


def plot_melspec(X, ax, sr, delta=False):
    if delta:
        cmap, vmin, vmax = "bwr", -18, 18
    else:
        cmap, vmin, vmax = "jet", -60, 20
    return librosa.display.specshow(
        X,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        antialiased=False,
        shading="nearest",
    )


def _save_audio_mel(processor, subgroup, device, num_channels=2, batch_size=4, name=""):
    processor = processor.to(device)
    processor_name = processor.__class__.__name__

    random_params = create_empty_parameters_from_shape_dict(
        processor.parameter_size(),
        num_nodes=1,
        device=device,
        std=0.1,
    )

    audio_data = load_audio_files()

    for audio_name, (waveform, sample_rate) in audio_data.items():
        if name == "":
            name = f"{processor_name}_{audio_name}"
        else:
            name = f"{name}_{audio_name}"

        waveform = waveform.to(device)
        waveform = waveform.contiguous()

        print(f"num channels to {num_channels}")
        print(waveform.shape)

        if num_channels == 2:
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            elif waveform.shape[1] > 2:
                waveform = waveform[:, :2, :]

        elif num_channels == 1:
            if waveform.shape[1] == 2:
                waveform = torch.mean(waveform, dim=1, keepdim=True)
            elif waveform.shape[1] > 2:
                waveform = torch.mean(waveform[:, :2, :], dim=1, keepdim=True)

        print(waveform.shape)

        output = processor(waveform, **random_params)
        if isinstance(output, tuple):
            output_signal, intermediates = output
        else:
            output_signal = output

        output_signal = output_signal.detach().cpu()

        print(output_signal.shape)

        save_output(processor_name, subgroup, output_signal, sample_rate, name)

        # Handle stereo for plotting
        if waveform.shape[1] == 2:
            input_np = (
                torch.mean(waveform, dim=1, keepdim=True)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        else:
            input_np = waveform.detach().cpu().numpy().flatten()

        if output_signal.shape[1] == 2:
            output_np = (
                torch.mean(output_signal, dim=1, keepdim=True)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        else:
            output_np = output_signal.detach().cpu().numpy().flatten()

        input_mel_spec = logmelspec(input_np, sr=sample_rate)
        output_mel_spec = logmelspec(output_np, sr=sample_rate)
        difference_mel_spec = output_mel_spec - input_mel_spec

        # Plot mel spectrogram
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

        # input
        spec_input = plot_melspec(input_mel_spec, ax=axs[0], sr=sample_rate)
        axs[0].set_title(f"Input Mel Spectrogram - {audio_name}")
        plt.colorbar(spec_input, ax=axs[0], format="%+2.0f dB")

        # output
        spec_output = plot_melspec(output_mel_spec, ax=axs[1], sr=sample_rate)
        axs[1].set_title(f"Output Mel Spectrogram - {audio_name}")
        plt.colorbar(spec_output, ax=axs[1], format="%+2.0f dB")

        # difference
        spec_diff = plot_melspec(
            difference_mel_spec, ax=axs[2], sr=sample_rate, delta=True
        )
        axs[2].set_title(f"Difference Mel Spectrogram - {audio_name}")
        plt.colorbar(spec_diff, ax=axs[2], format="%+2.0f dB")

        output_dir = f"tests/outputs/{subgroup}/{processor_name}/mels"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(opj(output_dir, f"{name}_mel.png"))
        plt.close(fig)
