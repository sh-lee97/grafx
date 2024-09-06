import os
from pprint import pprint

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
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
    audio_path = "tests/samples"
    audio_files = {
        "drums": "drums.wav",
        "speech": "speech.wav",
        "bass": "bass.wav",
        "guitar": "guitar.wav",
        "singing": "singing.wav",
        "music": "music.wav",
    }
    audio_data = {}
    for key, file_name in audio_files.items():
        file_path = opj(audio_path, file_name)
        waveform, sample_rate = librosa.load(file_path, sr=None, mono=False)

        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)
        elif waveform.ndim == 2 and waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        waveform = waveform[..., : 2**17]
        audio_data[key] = (waveform, sample_rate)

    return audio_data


def save_output(processor_class, subgroup, output, sample_rate, test_name):
    output_dir = f"tests/outputs/{subgroup}/{processor_class}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = opj(output_dir, f"{test_name}.wav")
    sf.write(output_file, output[0].detach().cpu().numpy().T, sample_rate)


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


def process_waveform(processor, waveform, random_params, device):
    waveform = torch.tensor(waveform)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    waveform = waveform.to(device).contiguous()
    print(f"!!!! {waveform.shape}")

    pprint(random_params)
    output = processor(waveform, **random_params)

    if isinstance(output, tuple):
        output_signal = output[0]
    elif isinstance(output, torch.Tensor):
        output_signal = output
    print(f"!!!! {output_signal.shape}")

    output_signal = output_signal.squeeze(0)
    output_signal = output_signal.detach().cpu().numpy()
    return output_signal


def generate_mel_spectrogram(waveform, sample_rate):
    # waveform_np = waveform.detach().cpu().numpy().flatten()
    waveform_np = waveform.flatten()
    return logmelspec(waveform_np, sr=sample_rate)


def plot_and_save_mel_spectrograms(
    input_specs,
    output_specs,
    difference_specs,
    sample_rate,
    audio_name,
    processor_name,
    subgroup,
    name,
):
    num_channels = len(input_specs)
    fig, axs = plt.subplots(3, num_channels, figsize=(20, 12), constrained_layout=True)

    for i, (channel, spec) in enumerate(input_specs.items()):
        spec_input = plot_melspec(spec, ax=axs[0, i], sr=sample_rate)
        axs[0, i].set_title(f"Input Mel Spectrogram ({channel}) - {audio_name}")
        plt.colorbar(spec_input, ax=axs[0, i], format="%+2.0f dB")

    for i, (channel, spec) in enumerate(output_specs.items()):
        spec_output = plot_melspec(spec, ax=axs[1, i], sr=sample_rate)
        axs[1, i].set_title(f"Output Mel Spectrogram ({channel}) - {audio_name}")
        plt.colorbar(spec_output, ax=axs[1, i], format="%+2.0f dB")

    for i, (channel, spec) in enumerate(difference_specs.items()):
        spec_diff = plot_melspec(spec, ax=axs[2, i], sr=sample_rate, delta=True)
        axs[2, i].set_title(f"Difference Mel Spectrogram ({channel}) - {audio_name}")
        plt.colorbar(spec_diff, ax=axs[2, i], format="%+2.0f dB")

    output_dir = f"tests/outputs/{subgroup}/{processor_name}/mels"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{name}_mel.png"))
    plt.close(fig)


def save_output(processor_class, subgroup, output, sample_rate, test_name):
    output_dir = f"tests/outputs/{subgroup}/{processor_class}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = opj(output_dir, f"{test_name}.wav")
    sf.write(output_file, output[0].T, sample_rate)


def setup_processor(processor, device, std=1, num_nodes=1):
    processor = processor.to(device)
    processor_name = processor.__class__.__name__

    random_params = create_empty_parameters_from_shape_dict(
        processor.parameter_size(),
        num_nodes=num_nodes,
        device=device,
        std=std,
    )

    return processor, processor_name, random_params


def _save_audio_and_mel(
    processor, subgroup, device, num_channels=2, batch_size=4, std=1, name=""
):
    # Setup processor
    processor, processor_name, random_params = setup_processor(
        processor, device, std, num_nodes=1
    )
    # Load audio
    audio_data = load_audio_files()

    for audio_name, (input_signal, sample_rate) in audio_data.items():
        # Save processed audio
        if input_signal.shape[0] == 1:
            continue
        current_name = (
            f"{processor_name}_{audio_name}_std{std}"
            if name == ""
            else f"{name}_{audio_name}_std{std}"
        )
        output_signal = process_waveform(processor, input_signal, random_params, device)
        save_output(processor_name, subgroup, output_signal, sample_rate, current_name)

        # print(current_name)
        # print(input_signal.shape)
        # print(output_signal.shape)

        # Save mel
        input_specs, output_specs, difference_specs = {}, {}, {}

        if output_signal.shape[0] != 2:  # Output is mono
            # Generate mel spec for mono
            input_mel_spec = generate_mel_spectrogram(input_signal[0], sample_rate)
            output_mel_spec = generate_mel_spectrogram(output_signal[0], sample_rate)
            difference_mel_spec = output_mel_spec - input_mel_spec

            # Populate specs dictionaries
            input_specs = {"Mono": input_mel_spec}
            output_specs = {"Mono": output_mel_spec}
            difference_specs = {"Mono": difference_mel_spec}

            # Plot mono
            fig, axs = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

            # Plot input
            spec_input = plot_melspec(input_specs["Mono"], ax=axs[0], sr=sample_rate)
            axs[0].set_title(f"Input (Mono) - {audio_name}")
            plt.colorbar(spec_input, ax=axs[0], format="%+2.0f dB")

            # Plot output
            spec_output = plot_melspec(output_specs["Mono"], ax=axs[1], sr=sample_rate)
            axs[1].set_title(f"Output (Mono) - {audio_name}")
            plt.colorbar(spec_output, ax=axs[1], format="%+2.0f dB")

            # Plot difference
            spec_diff = plot_melspec(
                difference_specs["Mono"], ax=axs[2], sr=sample_rate, delta=True
            )
            axs[2].set_title(f"Difference (Mono) - {audio_name}")
            plt.colorbar(spec_diff, ax=axs[2], format="%+2.0f dB")

        else:  # Output is stereo
            if input_signal.shape[0] == 1:
                input_signal = np.repeat(input_signal, 2, axis=0)

            # Flatten stereo input/output
            left_input = input_signal[0, :].flatten()
            right_input = input_signal[1, :].flatten()
            left_output = output_signal[0, :].flatten()
            right_output = output_signal[1, :].flatten()

            # Generate mel spec for stereo
            input_specs = {
                "Left": logmelspec(left_input, sr=sample_rate),
                "Right": logmelspec(right_input, sr=sample_rate),
                "Mid": logmelspec((left_input + right_input) / 2, sr=sample_rate),
                "Side": logmelspec((left_input - right_input) / 2, sr=sample_rate),
            }

            output_specs = {
                "Left": logmelspec(left_output, sr=sample_rate),
                "Right": logmelspec(right_output, sr=sample_rate),
                "Mid": logmelspec((left_output + right_output) / 2, sr=sample_rate),
                "Side": logmelspec((left_output - right_output) / 2, sr=sample_rate),
            }

            # Calculate difference
            difference_specs = {
                "Left": output_specs["Left"] - input_specs["Left"],
                "Right": output_specs["Right"] - input_specs["Right"],
                "Mid": output_specs["Mid"] - input_specs["Mid"],
                "Side": output_specs["Side"] - input_specs["Side"],
            }

            # Plot stereo
            fig, axs = plt.subplots(3, 4, figsize=(22, 14), constrained_layout=True)

            # Add column hearder
            column_titles = ["Left", "Right", "Mid", "Side"]
            for i, title in enumerate(column_titles):
                axs[0, i].set_title(title, fontsize=14)

            # Add row header
            for x, row_header in zip(
                [0.82, 0.5, 0.18], ["Input", "Output", "Difference"]
            ):
                fig.text(
                    -0.01,
                    x,
                    row_header,
                    va="center",
                    rotation="vertical",
                    fontsize=14,
                    # clip_on=False,
                )

            plt.subplots_adjust(
                left=0.3, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4
            )

            # Plot input: Left, Right, Mid, Side
            for i, (channel, spec) in enumerate(input_specs.items()):
                spec_input = plot_melspec(spec, ax=axs[0, i], sr=sample_rate)
                # axs[0, i].set_title(f"Input ({channel})")
            plt.colorbar(spec_input, ax=axs[0, i], format="%+2.0f dB")

            # Plot output: Left, Right, Mid, Side
            for i, (channel, spec) in enumerate(output_specs.items()):
                spec_output = plot_melspec(spec, ax=axs[1, i], sr=sample_rate)
                # axs[1, i].set_title(f"Output ({channel})")
            plt.colorbar(spec_output, ax=axs[1, i], format="%+2.0f dB")

            # Plot difference: Left, Right, Mid, Side
            for i, (channel, spec) in enumerate(difference_specs.items()):
                spec_diff = plot_melspec(spec, ax=axs[2, i], sr=sample_rate, delta=True)
                # axs[2, i].set_title(f"Difference ({channel})")
            plt.colorbar(spec_diff, ax=axs[2, i], format="%+2.0f dB")

        # Save the plot
        output_dir = f"tests/outputs/{subgroup}/{processor_name}/mels"
        os.makedirs(output_dir, exist_ok=True)
        plt.suptitle(f"{audio_name.capitalize()} std:{std}", fontsize=20)
        plt.savefig(
            opj(output_dir, f"{current_name}_mel.png"),
            bbox_inches="tight",
            # pad_inches=0.1,
        )
        plt.close(fig)
