import math
import warnings

import torch


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


def _hz_to_bark(freqs: float, bark_scale: str = "traunmuller") -> float:
    r"""Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "schroeder", "traunmuller" or "wang".'
        )

    if bark_scale == "wang":
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == "schroeder":
        return 7.0 * math.asinh(freqs / 650.0)
    # Traunmuller Bark scale
    barks = ((26.81 * freqs) / (1960.0 + freqs)) - 0.53
    # Bark value correction
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)

    return barks


def _bark_to_hz(barks: torch.Tensor, bark_scale: str = "traunmuller") -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (torch.Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (torch.Tensor): Barks converted in Hz
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "traunmuller", "schroeder" or "wang".'
        )

    if bark_scale == "wang":
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == "schroeder":
        return 650.0 * torch.sinh(barks / 7.0)
    # Bark value correction
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22

    # Traunmuller Bark scale
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))

    return freqs


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def _hz_to_log(freqs):
    return math.log(freqs)


def _log_to_hz(logs):
    return torch.exp(logs)


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
    match scale:
        case "bark_traunmuller" | "bark_schroeder" | "bark_wang":
            bark_scale = scale.split("_")[1]
            b_min = _hz_to_bark(f_min, bark_scale=bark_scale)
            b_max = _hz_to_bark(f_max, bark_scale=bark_scale)
            b_pts = torch.linspace(b_min, b_max, n_filters + 2)
            f_pts = _bark_to_hz(b_pts, bark_scale=bark_scale)

        case "mel_htk" | "mel_slaney":
            mel_scale = scale.split("_")[1]
            m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
            m_max = _hz_to_mel(f_max, mel_scale=mel_scale)
            m_pts = torch.linspace(m_min, m_max, n_filters + 2)
            f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

        case "linear":
            f_pts = torch.linspace(f_min, f_max, n_filters + 2)
            fb = _create_triangular_filterbank(all_freqs, f_pts)

        case "log":
            l_min = _hz_to_log(f_min)
            l_max = _hz_to_log(f_max)
            l_pts = torch.linspace(l_min, l_max, n_filters + 2)
            f_pts = _log_to_hz(l_pts)

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
