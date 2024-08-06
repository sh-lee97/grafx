import torch.nn as nn

from grafx.processors.components import ZeroPhaseFIR
from grafx.processors.functional import convolve


class ZeroPhaseFIREqualizer(nn.Module):
    r"""
    A single-channel zero-phase finite impulse response (FIR) filter.

        From the input log-magnitude $H_{\mathrm{log}}$,
        we compute inverse FFT (IFFT) of the magnitude response
        and multiply it with a zero-centered Hann window $v^\mathrm{Hann}[n]$.
        Each input channel is convolved with the following FIR.
        $$
        h[n] = v^\mathrm{Hann}[n] \cdot \frac{1}{N} \sum_{k=0}^{N-1} \exp H_{\mathrm{log}}[k] \cdot  w_{N}^{kn}.
        $$

        Here, $-(N+1)/2 \leq n \leq (N+1)/2$ and $w_{N} = \exp(j\cdot 2\pi/N)$.
        This equalizer's learnable parameter is $p = \{ H_{\mathrm{log}} \}$.

    Args:
        num_magnitude_bins (:python:`int`, *optional*):
            The number of FFT magnitude bins (default: :python:`1024`).
    """

    def __init__(self, num_magnitude_bins=1024):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.fir = ZeroPhaseFIR(num_magnitude_bins)

    def forward(self, input_signals, log_magnitude):
        r"""
        Processes input audio with the processor and given parameters.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C \times L`):
                A batch of input audio signals.
            log_magnitude (:python:`FloatTensor`, :math:`B \times K \:\!`):
                A batch of log-magnitude vectors of the FIR filter.

        Returns:
            :python:`FloatTensor`: A batch of output signals of shape :math:`B \times C \times L`.
        """
        fir = self.fir(log_magnitude)[:, None, :]
        output_signals = convolve(input_signals, fir, mode="zerophase")
        return output_signals

    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        return {"log_magnitude": self.num_magnitude_bins}
