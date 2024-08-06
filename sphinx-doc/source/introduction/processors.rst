.. role:: python(code)
     :language: python
     :class: highlight

.. _differentiable-processors:

Differentiable Processors
===========================

..  

.. ---------------------------

.. What kind of processors $f$ are suitable for our graph framework?

We can import to or approximate the conventional audio processors, 
building blocks of the audio processing graphs, 
in the automatic differentiation framework 
(:python:`PyTorch` for our case).
This practice is known as *differentiable digital signal processing* :cite:`engel2020ddsp, hayes2023review`, 
or DDSP in short.
The differentiable processors have the following advantages.

#. Recall that many works 
   :cite:`uzrad2024diffmoog, caspe2022ddx7, steinmetz2020diffmixconsole, ramirez2021differentiable, colonel2023music, ye2023fm, Mitcheltree_2021, guo2023automatic, lee2023blind, lee2024searching`
   involves a parameter estimation task.
   With the differentiable processors, we can optimize these parameters (or their neural predictors) 
   by comparing the processed audio $\smash{\hat{\mathbf{Y}}} = G(\mathbf{S}, \mathbf{P})$ and the target $\mathbf{Y}$
   and back-propagating the gradients through the entire processing graph $G$
   (commonly refered as *end-to-end* optimization).

#. As they are identical to or approximate the real-world processors that the practitioners are familiar with, 
   the obtained parameters are easy to interpret and control.
   Of course, our framework can be used with any neural network that provides the gradients;
   the differentiable processors just suit more to the compositional and interpretable nature of the audio processing graph.

On this page, we share how we implemented and handled such processors within :python:`GRAFX` framework.

.. while each processor's detail is delayed to the :python:`grafx.processors` API documentation.

    Currently, most of them are *audio effects*, e.g., 
    equalizers :cite:`nercessian2020neural`, 
    dynamic range compressors :cite:`steinmetz2022style`, 
    and artificial reverberations :cite:`lee2022differentiable`. 
    We are planning to also add *generators* such as simple oscillators :cite:`engel2020ddsp`.

.. ---------------------------

Batched Processing
---------------------------

In the previous section, we defined the processor $f$, as a node $v \in V$ with type $t$, 
that outputs audio from input signals and parameters.
Here, we introduce a notation for the batched processing of the processor $f$.
Assume that there is a node subset $Q \subset V_t$ that consists of the type-$t$ processors.
Instead of processing $|Q|$ nodes independently, we can process them with a single processor $f$ that takes the batched input signals and parameters.
We write this as follows,
$$
\mathbf{Y}^{(1)}_{Q}, \cdots, \mathbf{Y}^{(N)}_{Q} 
= f\!\left(\mathbf{U}^{(1)}_{Q}, \cdots, \mathbf{U}^{(M)}_{Q}, \mathbf{P}_{Q}\right).
$$

Here, $\smash{\mathbf{U}^{(m)}_{Q} \in \mathbb{R}^{\left| Q \right|\times C\times L}}$ 
is a stack of subset nodes' input signals for the $m^\mathrm{th}$ inlet.
The parameters $\smash{\mathbf{P}_{Q}}$ are stacked in a similar manner 
with each tensor having the first dimension of size $|Q|$.
Note that the node order in parameter tensors must be the same as the one in the input signals.
The output returned by the processor 
$\smash{\mathbf{Y}^{(n)}_{Q} \in \mathbb{R}^{\left| Q \right|\times C\times L}}$ are also stacked the same way as the input.
In the remaining documentation, this arbitrary batching is always assumed, and the subscript $Q$ is dropped for brevity,
unless we need to emphasize it. 

.. Observe that we can find a sequence of such node subsets to obtain the final output $\mathbf{Y}=G(\mathbf{S}, \mathbf{P})$.
    This will be discussed on the :ref:`following post <batched-audio-processing>`.


Parameter Gradients
~~~~~~~~~~~~~~~~~~~~

A differentiable processor $f$ should be able to compute the gradients of the output audio 
with respect to its input audio and parameters.
We denote these gradients as
$\nabla_p \mathbf{Y}^{(n)}$ and $\nabla_{\mathbf{U}^{(m)}} \mathbf{Y}^{(n)}$, respectively, for all $m$, $n$, and $p \in \mathbf{P}$
($Q$ omitted).
If every processor provides these gradients, after computing the graph output $\mathbf{Y} = G(\mathbf{S}, \mathbf{P})$,
we can backpropagate through the entire graph $G$ and optimize the parameters $\mathbf{P}$ via chain rule.

  While it is not our primary focus, we can also obtain the gradients w.r.t. the input source signals $\mathbf{S}$.
  These gradients are useful in some scenarios, e.g., 
  when we want to solve an inverse problem with pre-trained diffusion models :cite:`chung2022diffusion`.
  Some recent works on audio watermarking also utilize such gradients :cite:`chen2023wavmark, roman2024proactive`.
  Finally, instead of the source $\mathbf{S}$ and parameters $\mathbf{P}$, 
  we might want to obtain gradients w.r.t. the *graph structure* $G$.
  It is not directly possible, as the graph structure is inherently discrete, but we may obtain the gradients 
  w.r.t. the relaxed continuous representation (of some structural modifications).
  Notable examples include the graph pruning via soft gating mechanism :cite:`lee2024searching, he2023structured, cheng2023survey`,
  which also can be generalized into the *differentiable artitecture search (DARTS)* framework :cite:`liu2018darts, ye2023fm`.

About this DDSP Business
~~~~~~~~~~~~~~~~~~~~~~~~~

Due to the aforementioned chain rule, every processor in the graph must provide the gradients: 
$\nabla_p \mathbf{Y}^{(n)}$ and $\nabla_{\mathbf{U}^{(m)}} \mathbf{Y}^{(n)}$.
If the one processor does not provide the former, we cannot optimize its parameters $p$.
If the latter is not available, the backpropagation stops at that node $v$.

Moreover, it is desirable to compute these gradients in GPU efficiently, 
this is not so straightforward for some processors.
To address this, various approximation methods have been proposed;
one notable example is the frequency sampling method that eliminates 
the linear recurrent loop in the infinite impulse response filters 
:cite:`nercessian2020neural, lee2022differentiable`.
Approximation of dynamic range compressors, which contain nonlinear recurrent loops,
has been also proposed :cite:`colonel2023music, colonel2022reverse, steinmetz2022style`.

.. Time-varying processors :cite:`carson2023differentiable`,

Sometimes, the analytical gradients are simply not available, e.g., ones that include black-boxes or discrete operations.
In this case, we can still resort to finite difference methods :cite:`martinez2020deep`, straight-through estimators :cite:`bengio2013estimating`, 
or use a pre-trained auxiliary neural network that mimics the processors :cite:`steinmetz2020diffmixconsole` to approximate the gradients.
In the literature, these approaches are also referred to as "differentiable;" hence making it an umbrella term encompassing all practical methods that obtain the output signals, gradients, or their approximates of the processors of interest.

Our parameter gradient $\nabla_\mathbf{P} L$ is a function $h$ of not only the graph $G$ and the current parameter $\mathbf{P}$,
but also the signals (the source $\mathbf{S}$ if exists and the target $\mathbf{Y}$) and the loss function $L$.
By slightly abusing the notation of partial derivatives, we can write the gradient as follows,
$$
\nabla_\mathbf{P} L = h(\mathbf{P}; G, \mathbf{Y}, \mathbf{S}, L)
= \frac{\partial G(\mathbf{S}; \mathbf{P})}{\partial \mathbf{P}} \frac{\partial L(\mathbf{Y}, G(\mathbf{S}; \mathbf{P}))}{\partial G(\mathbf{S}; \mathbf{P})}.
$$

This indicates that, even if the model $G$ itself is very simple, its gradients can be highly nonconvex, 
hindering the optimization.
One notable example is the unconstrained sinusoid model; 
it is still an unsolved problem to optimize the amplitude, frequency, and phase of the sinusoids 
so that their sum matches the target audio.
To mitigate this, a surrogate model $\smash{\tilde G}$ is introduced :cite:`hayes2023sinusoidal`
or a novel loss function $L$ is designed :cite:`torres2024unsupervised, schwar2023multi`,
albeit none of them completely solves the problem.

For more details, refer to the recent review :cite:`hayes2023review`, ISMIR tutorial `"Introduction to DDSP for Audio Synthesis," <https://intro2ddsp.github.io/intro.html>`_ and references therein.


.. ---------------------------

Implementation Details
---------------------------

Following the standard practice, our dfferentiable processors inherit :python:`nn.Module`.
For example, see the implementation of :class:`~grafx.processors.stereo.StereoGain`, 
which applies channel-wise gain to a batch of (mono or stereo) signals, resulting in a panning effect.

.. code-block:: python

    import torch
    import torch.nn as nn

    class StereoGain(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_signals, log_gain):
            gain = torch.exp(log_gain)[..., None]
            output_signals = input_signals * gain 
            return output_signals

        def parameter_size(self):
            return {'log_gain': 2}


Forward Pass
~~~~~~~~~~~~~~~~~~~~

Each processor accepts *both* input signal(s) and (collection of) parameters for each forward pass with a signature of 
:python:`forward(*input_signals, **processor_parameters)`.
Observe that we do not store the processor parameters (e.g., as a :python:`nn.Parameter`) internally;
only the fixed buffers or hyperparameters are stored as class attributes.
This approach avoids creating multiple processor instances of type $t$ 
and allows processing of any batched input tensors, $\mathbf{P}_{Q}$,
of an arbitrary node subset $Q \subset V_t$.
Also, exposing the parameters makes the implementation of parameter gradient descent 
and training of the neural networks as parameter predictors almost identical.
Along with the outputs, one can also return a dictionary 
(e.g., containing regularization terms) as a second return value,
which will be collected when we compute the graph output.


Parameter Shapes
~~~~~~~~~~~~~~~~~~~~~~

We also implement the :python:`parameter_size()` method, which returns the shape of each parameter tensor in a dictionary format.
Note that it returns the tensor shapes *without* the batch (or node) dimension. 
While this method is not essential, it is useful for initializing parameters for the gradient descent or creating a prediction head for the neural network.

For example, to perform a gradient descent of the given graph's parameters,
we first prepare the processors in a dictionary format, 
where each key is the processor name (one provided to the config) 
and its value is a processor instance.
This approach will also be used to render the output audio (described in the following page),

.. code-block:: python

    from grafx.processors import (
        ZeroPhaseFIREqualizer,
        ApproxCompressor,
        MidSideFilteredNoiseReverb
    )

    processors = {
        "eq": ZeroPhaseFIREqualizer(),
        "compressor": ApproxCompressor(),
        "reverb": MidSideFilteredNoiseReverb()
    } 


Then, we use a :func:`~grafx.utils.create_empty_parameters` method that creates an empty parameter dictionary from the graph tensor :python:`G_t` and its processors.
Here, the graph :python:`G_t` is the one we created in the last page, containing three equalizers, compressors, and reverbs.

.. code-block:: bash

    from grafx.utils import create_empty_parameters 
    parameters = create_empty_parameters(G_t, processors)
    
The returned :python:`parameters` will have a nested dictioary format, 
where the first key is the processor name and the second key is the parameter name.
Note that we use :python:`nn.ModuleDict` and :python:`nn.ParameterDict` to store the parameters, instead of the default :python:`dict`.
We can check the shapes of the parameters with :python:`print(parameters)` and the outputs will be as follows. 
Observe that the first dimension of each tensor is the number of nodes of that type in the graph, i.e., three in this case.

.. code-block:: bash

    ModuleDict(
        (eq): ParameterDict(  
            (log_magnitude): Parameter containing: [torch.FloatTensor of size 3x1024]
        )
        (compressor): ParameterDict(
            (log_knee): Parameter containing: [torch.FloatTensor of size 3x1]
            (log_ratio): Parameter containing: [torch.FloatTensor of size 3x1]
            (log_threshold): Parameter containing: [torch.FloatTensor of size 3x1]
            (z_alpha): Parameter containing: [torch.FloatTensor of size 3x1]
        )
        (reverb): ParameterDict(
            (delta_log_magnitude): Parameter containing: [torch.FloatTensor of size 3x2x193]
            (init_log_magnitude): Parameter containing: [torch.FloatTensor of size 3x2x193]
        )
    )

.. 
    Limitations and Restrictions
    -----------------------------
    Once the above specification is met, users can use any custom processor (not only the differentiable one, but even a neural network) within our framework. However, there are some limitations and restrictions to the processors.

    III* For the batched node processing, all input signals must be provided.
    * All inputs and outputs will 
