from pprint import pprint

import matplotlib.pyplot as plt
import torch

from grafx.processors import *
from grafx.utils import create_empty_parameters_from_shape_dict


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
):
    processor = processor.to(device)
    parameter_size = processor.parameter_size()
    parameter_size = {
        k: ((v,) if isinstance(v, int) else v) for k, v in parameter_size.items()
    }
    parameters = create_empty_parameters_from_shape_dict(
        parameter_size,
        num_nodes=batch_size,
        device=device,
    )
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
