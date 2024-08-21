import warnings

import torch
import torch.nn as nn

from grafx.processors.core.scale import from_scale, to_scale


class TriangularFilterBank(nn.Module):
    r"""
    Creates a triangular filterbank for the given frequency range.
    Code adapted from `torchaudio <https://pytorch.org/audio/stable/generated/torchaudio.functional.melscale_fbanks.html#torchaudio.functional.melscale_fbanks>`_
    and `Diff-MST <https://github.com/sai-soum/Diff-MST/blob/main/mst/filter.py>`_.

        We provide both analysis and synthesis mode.
        For the synthesis mode, we expand the input energy $\mathbf{E}_\mathrm{fb} \in \mathbb{R}^{B \times F_{\mathrm{fb}}}$
        with the number of filterbanks $F_\mathrm{fb}$ to the linear FFT scale $\mathbf{E} \in \mathbb{R}^{B \times F}$.
        $$
        \mathbf{E} = \mathbf{E}_\mathrm{fb} \mathbf{W}_\mathrm{fb}
        $$

        $\smash{\mathbf{W}_\mathrm{fb} \in \mathbb{R}^{F \times F_{\mathrm{fb}}}}$ is the standard trainagular filterbank matrix.
        The analysis mode downsamples the frequency axis by multiplying the normalized filterbank matrix
        (sum of each filterbank is 1; hence an adaptive weighted average pooling).

    Args:
        num_frequency_bins (:python:`int`):
            Number of frequency bins from linear FFT.
        num_filters (:python:`int`):
            Number of the filterbank filters.
        scale (:python:`str`, *optional*):
            Frequency scale to use: :python:`"bark_traunmuller"`, :python:`"bark_schroeder"`, :python:`"bark_wang"`, :python:`"mel_htk"`, :python:`"mel_slaney"`, :python:`"linear"`, :python:`"log"` (default: :python:`"bark_traunmuller"`).
        f_min (:python:`float`, *optional*):
            Minimum frequency (default: :python:`40`).
        f_max (:python:`float`, *optional*):
            Maximum frequency (default: :python:`None`).
        low_half_triangle (:python:`bool`, *optional*):
            Attach the remaining low-freq parts (default: :python:`True`).
    """

    def __init__(
        self,
        num_frequency_bins,
        num_filters=50,
        scale="bark_traunmuller",
        f_min=40,
        f_max=None,
        sr=44100,
        low_half_triangle=True,
    ):
        super().__init__()

        if f_max is not None:
            if f_max > sr // 2:
                warnings.warn(
                    f"The value for `f_max` ({f_max}) is higher than the Nyquist frequency ({sr // 2}). The value for `f_max` will be set to the Nyquist frequency."
                )
                f_max = sr // 2

        filterbank = TriangularFilterBank.compute_matrix(
            num_frequency_bins=num_frequency_bins,
            num_filters=num_filters,
            scale=scale,
            f_min=f_min,
            f_max=f_max,
            sr=sr,
            low_half_triangle=low_half_triangle,
        )
        self.register_buffer("filterbank", filterbank.T)
        self.num_filters = num_filters

        filterbank_normalized = filterbank / filterbank.sum(0, keepdim=True)
        self.register_buffer("filterbank_normalized", filterbank_normalized)

    def forward(self, energy, mode="synthesis"):
        r"""
        Apply the filterbank to the energy tensor.

        Args:
            energy (:python:`FloatTensor`, :math:`B\times F \:\!`):
                A batch of energy tensors.
            mode (:python:`str`, *optional*):
                Mode of operation: :python:`"analysis"` or :python:`"synthesis"` (default: :python:`"synthesis"`).

        Returns:
            :python:`FloatTensor`: The energy tensor after applying the filterbank.
        """
        match mode:
            case "analysis":
                energy = torch.matmul(energy, self.filterbank_normalized)
            case "synthesis":
                energy = torch.matmul(energy, self.filterbank)
        return energy

    @staticmethod
    def compute_matrix(
        num_frequency_bins,
        num_filters,
        scale,
        f_min,
        f_max,
        sr,
        low_half_triangle,
    ):
        r"""
        Compute the triangular filterbank matrix
        $\smash{\mathbf{W}_\mathrm{fb} \in \mathbb{R}^{F \times F_{\mathrm{fb}}}}$.
        """

        assert scale in [
            "bark_traunmuller",
            "bark_schroeder",
            "bark_wang",
            "mel_htk",
            "mel_slaney",
            "linear",
            "log",
        ]

        if low_half_triangle:
            num_filters -= 1

        # freq bins
        all_freqs = torch.linspace(0, sr // 2, num_frequency_bins)

        # calculate bark freq bins
        s_min, s_max = to_scale(f_min, scale), to_scale(f_max, scale)
        s_pts = torch.linspace(s_min, s_max, num_filters + 2)
        f_pts = from_scale(s_pts, scale)

        # create filterbank
        fb = TriangularFilterBank._create_triangular_filterbank(all_freqs, f_pts)

        # remaining low-freq parts
        if low_half_triangle:
            remaining = 1 - torch.sum(fb, -1)
            fb = torch.cat([remaining[:, None], fb], -1)

        # sanity check
        if (fb.max(dim=0).values == 0.0).any():
            warnings.warn(
                f"At least one bark filterbank has all zero values. The value for `n_bins` ({num_filters}) may be set too high. Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
            )
        return fb

    @staticmethod
    def _create_triangular_filterbank(all_freqs, f_pts):
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))
        return fb
