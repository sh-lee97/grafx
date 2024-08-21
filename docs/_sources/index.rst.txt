.. role:: python(code)
     :language: python
     :class: highlight

GRAFX 
================================================================================================

:python:`GRAFX` is an open-source library designed for handling audio processing graphs in :python:`PyTorch`.
One can create and modify a graph, convert it to tensor representations,
and process output audio efficiently in GPU with batched node processing.
The library is complemented with various differentiable audio processors, 
which enables end-to-end optimization of processor parameters or their estimators (e.g., graph neural networks) via gradient descent.
The code can be found in `this repository <https://github.com/sh-lee97/grafx>`_.


Installation
---------------------------

.. code-block:: bash
    
    pip install grafx

Some processors use convolutions; for their efficient processing, install :python:`FlashFFTConv` from the following 
`github repository <https://github.com/HazyResearch/flash-fft-conv>`_.


Contents 
--------------------------
.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/graph
   introduction/processors
   introduction/render

.. toctree::
   :maxdepth: 1
   :caption: Graph API

   graph_api/data
   graph_api/render
   graph_api/draw
   graph_api/utils

.. toctree::
   :maxdepth: 1
   :caption: Processor API

   processor_api/core
   processor_api/filter
   processor_api/eq
   processor_api/stereo
   processor_api/dynamics
   processor_api/reverb
   processor_api/delay
   processor_api/container

.. toctree::
   :maxdepth: 1
   :caption: References

   references/history
   references/reference



Citation
----------------------------

.. code-block:: tex

   @inproceedings{lee2024grafx, 
   title={{GRAFX}: an open-source library for audio processing graphs in {P}y{T}orch},
   author={Lee, Sungho and Mart{\"\i}nez-Ram{\"\i}rez, Marco A and Liao, Wei-Hsiang and Uhlich, Stefan and Fabbro, Giorgio and Lee, Kyogu and Mitsufuji, Yuki},
   booktitle={DAFx (Demo)},
   year={2024}
   }
