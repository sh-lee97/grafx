import torch.nn as nn


class MonoToStereo(nn.Module):
    r"""
    A simple mono-to-stereo conversion.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_signals):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times 1 \times L`):
                A batch of input audio signals; must be mono.

        Returns:
            A batch of output signals in :python:`FloatTensor` of shape :math:`B \times C \times L`.
        """
        b, c, t = input_signals.shape
        assert c == 2
        output_signals = input_signals.repeat(1, 2, 1)
        return output_signals

    def parameter_size(self):
        """
        Returns:
            A dictionary that contains each parameter tensor's shape.
        """
        return {}

