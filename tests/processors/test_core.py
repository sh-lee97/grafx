def _hmm():
    import matplotlib.pyplot as plt

    num_frequency_bins = 900
    f_min = 40
    f_max = 18000
    n_barks = 50
    sr = 36000

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
        fb = TriangularFilterBank.compute_matrix(
            num_frequency_bins, f_min, f_max, n_barks, scale
        )
        print(fb.shape)
        ax[i].plot(fb, label=scale)
        ax[i].plot(fb.sum(-1), label=scale)
    fig.set_size_inches(10, 20)
    fig.savefig("filterbanks.pdf", bbox_inches="tight")
