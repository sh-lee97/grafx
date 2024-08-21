.. role:: python(code)
     :language: python
     :class: highlight

Versions
===========================

..
  
.. --------------------------

TODOs
--------------------------

* New processors: higher-order filters, such as Moog ladder :cite:`vafilter`, stacked SVFs :cite:`werner2020moog`, and butterworth filters. 
  A Linkwitz-Riley crossover. 
  Differentiable artificial reverberations, including velvet noise-based ones and feedback delay networks :cite:`lee2022differentiable, dal2023differentiable`.
  Higher-order allpass filter, e.g., (frequency-dependent) Schroeder allpass. 
  Factorized compressors and noisegates. Limiters. 
  Memoryless nonlinearities :cite:`peladeau2024blind`.
  Modulation effects. Simple oscillators with modulation capabilities. 
  ADSR envelopes. 
* New containers: a multiband processor based on Linkwitz-Riley crossover 

v0.6.0
--------------------------

* Parameters: each processor can have a nested parameter dictionary of an arbitrary depth.

* Exposed core DDSP methods: 

  * 

* New processors:

  * Basic second-order IIR filters: 
    :class:`~grafx.processors.filter.BiquadFilter`, 
    :class:`~grafx.processors.filter.PoleZeroFilter`, 
    :class:`~grafx.processors.filter.StateVariableFilter`, 
    :class:`~grafx.processors.filter.LowPassFilter`, 
    :class:`~grafx.processors.filter.HighPassFilter`, 
    :class:`~grafx.processors.filter.BandPassFilter`, 
    :class:`~grafx.processors.filter.BandRejectFilter`, 
    :class:`~grafx.processors.filter.AllPassFilter`,
    :class:`~grafx.processors.filter.PeakingFilter`,
    :class:`~grafx.processors.filter.LowShelf`, and
    :class:`~grafx.processors.filter.HighShelf`.
  * New equalizers: :class:`~grafx.processors.eq.GraphicEqualizer` and :class:`~grafx.processors.eq.ParametricEqualizer`,
  * New reverb: :class:`~grafx.processors.reverb.FilteredNoiseShapingReverb` 
    (based on `dasp_pytorch <https://github.com/csteinmetz1/dasp-pytorch/>`_ and :cite:`steinmetz2021filtered, lee2024fade`).

* New containers:
  
  * :class:`~grafx.processors.container.SerialChain` and 
  * :class:`~grafx.processors.container.ParallelMix` added. 

* Updated processors: 

  * :class:`~grafx.processors.eq.ZeroPhaseFIREqualizer` (new window argument & allows triangular filterbank),
  * :class:`~grafx.processors.reverb.STFTFilteredNoiseReverb` and :class:`~grafx.processors.delay.MultitapDelay` (renamed): 
  * :class:`~grafx.processors.dynamics.Compressor` and :class:`~grafx.processors.dynamics.NoiseGate` (renamed): support efficient ballistics :cite:`yu2024differentiable` and more compression curves.


v0.5.0
--------------------------

* Graph data structures: :class:`~grafx.data.graph.GRAFX` and :class:`~grafx.data.tensor.GRAFXTensor`, and their basic utility functions.
* Audio rendering methods: :func:`~grafx.render.graph.render_grafx` and other preparation methods.
* Basic graph visualization tools: :func:`~grafx.draw.graph.draw_grafx` and other utility functions.
* Differentiable audio processors: :class:`~grafx.processors.eq.ZeroPhaseFIREqualizer`, 
  :class:`~grafx.processors.stereo.StereoGain`, 
  :class:`~grafx.processors.stereo.SideGainImager`, 
  :class:`~grafx.processors.dynamics.ApproxCompressor`, 
  :class:`~grafx.processors.dynamics.ApproxNoiseGate`, 
  :class:`~grafx.processors.reverb.MidSideFilteredNoiseReverb`, and
  :class:`~grafx.processors.delay.StereoMultitapDelay`. 
* Auxiliary processor containers: :class:`~grafx.processors.container.DryWet` and :class:`~grafx.processors.container.GainStagingRegularization`.

.. --------------------------

Pre-Release
---------------------------

A preliminary version of this library was created for work `Blind Estimation of Audio Processing Graph` :cite:`lee2023blind`.
Its aim was to create a simple baseline that can predict a graph from its output audio (or also with input audio).
At that time (Summer 2022), literature on the differentiable audio processors (and their efficient computation in GPU) was not as rich as now.
This led us to re-implement various processors in :python:`jax` 
to run both the forward and backward pass efficiently in CPU with :python:`jax.compile`.
Our hope was that, if the forward pass is written correctly, the parameter optimization with gradient descent should work as well.
Of course, this was not the case; for example, the modulation effects were not trained at all (now, we know why: :cite:`hayes2023sinusoidal, carson2023differentiable`).
Furthermore, the backpropagation through the graphs (even with ten nodes) was still too slow to be practical.
Consequently, we decided to only use the graph engine for the forward passes and the training of the graph and parameter predictors
was done with a simple "parameter loss."

After a year, we decided to revisit this idea of differentiable audio processing graphs
as many advances on the differentiable processors were made in the meantime 
:cite:`hayes2023sinusoidal, colonel2023music, carson2023differentiable, ye2023fm, hayes2023review, colone2023reverse, bargum2023differentiable, steinmetz2023high, masuda2023improving`.
This led us to the current version of :python:`GRAFX`, which is entirely based on :python:`PyTorch`
(in the current state, whether the backend is :python:`PyTorch` or :python:`jax` do not matter much, 
but we used the former for its popularity and ease of use).
This library :cite:`lee2024grafx` was developed along with the companion work `Searching For Music Mixing Graphs: A Pruning Approach` :cite:`lee2024searching`.
Its motivation was, unlike the previous work :cite:`lee2023blind`, we wanted to find graphs and their parameters 
that matches the real-world music mixture so that we do not need to rely on the previous synthetic data when training the neural networks.



