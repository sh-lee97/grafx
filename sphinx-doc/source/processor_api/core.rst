:tocdepth: 2

.. role:: python(code)
     :language: python
     :class: highlight


grafx.processors.core
============================

.. 
     Equalizers (EQs) that modify input audio's magnitude response.
     We currently have a single FIR EQ, but others will be added soon.


.. FIR Equalizers 
     ----------------------------

.. autoclass:: grafx.processors.core.convolution.FIRConvolution
   :members:
   :show-inheritance:

.. autoclass:: grafx.processors.core.fft_filterbank.TriangularFilterBank
   :members:
   :show-inheritance:

.. automodule:: grafx.processors.core.iir
   :members:
   :show-inheritance:

.. autoclass:: grafx.processors.core.delay.SurrogateDelay
   :members:
   :show-inheritance:

.. autoclass:: grafx.processors.core.envelope.TruncatedOnePoleIIRFilter
   :members:
   :show-inheritance:

.. autoclass:: grafx.processors.core.envelope.Ballistics
   :members:
   :show-inheritance: