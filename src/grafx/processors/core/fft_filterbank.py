import warnings

import torch

from grafx.processors.core.scale import from_scale, to_scale


def _create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
) -> torch.Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


def fft_triangular_filterbank(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_filters: int,
    scale="bark_traunmuller",
    attach_remaining=True,
) -> torch.Tensor:
    r"""
    Creates a triangular filterbank for the given frequency range.
    Code adapted from `torchaudio <https://pytorch.org/audio/stable/generated/torchaudio.functional.melscale_fbanks.html#torchaudio.functional.melscale_fbanks>`_
    and `Diff-MST <https://github.com/sai-soum/Diff-MST/blob/main/mst/filter.py>`_.

    Args:
        n_freqs (:python:`int`): Number of frequency bins.
        f_min (:python:`float`): Minimum frequency.
        f_max (:python:`float`): Maximum frequency.
        n_filters (:python:`int`): Number of filters.
        scale (:python:`str`, *optional*): Scale to use: :python:`"bark_traunmuller"`, :python:`"bark_schroeder"`, :python:`"bark_wang"`, :python:`"mel_htk"`, :python:`"mel_slaney"`, :python:`"linear"`, :python:`"log"` (default: :python:`"bark_traunmuller"`).
        attach_remaining (:python:`bool`, *optional*): Attach the remaining low-freq parts (default: :python:`True`).

    Returns:
        :python:`FloatTensor`: Filterbank of size :python:`(n_freqs, n_filters)`.
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

    if attach_remaining:
        n_filters -= 1

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate bark freq bins
    s_min, s_max = to_scale(f_min, scale), to_scale(f_max, scale)
    s_pts = torch.linspace(s_min, s_max, n_filters + 2)
    f_pts = from_scale(s_pts, scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    # remaining low-freq parts
    if attach_remaining:
        remaining = 1 - torch.sum(fb, -1)
        fb = torch.cat([remaining[:, None], fb], -1)

    # sanity check
    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one bark filterbank has all zero values. "
            f"The value for `n_bins` ({n_filters}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )
    return fb


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_freqs = 900
    f_min = 40
    f_max = 18000
    n_barks = 50
    sample_rate = 36000

    fig, ax = plt.subplots(7, 1)
    scales = [
        "bark_traunmuller",
        "bark_schroeder",
        "bark_wang",
        "mel_htk",
        "mel_slaney",
        "linear",
        "log",
    ]
    for i in range(len(scales)):
        scale = scales[i]
        print(scale)
        fb = fft_triangular_filterbank(n_freqs, f_min, f_max, n_barks, scale)
        print(fb.shape)
        ax[i].plot(fb, label=scale)
        ax[i].plot(fb.sum(-1), label=scale)
    fig.set_size_inches(10, 20)
    fig.savefig("filterbanks.pdf", bbox_inches="tight")
