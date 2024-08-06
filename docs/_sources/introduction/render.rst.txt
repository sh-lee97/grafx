.. role:: python(code)
     :language: python
     :class: highlight

.. _batched-audio-processing:

Batched Audio Processing
===========================

..  

.. ---------------------------

To compute the output audio of the graph faster in GPU, 
it is desirable to parallelize the computation as much as possible.
The most standard approach is batched processing.
We note that there are three levels of batched processing: *node-level*, *source-level*, and *graph-level*.


.. ---------------------------

Node-Level Parallelism
---------------------------

First, consider the computation of a single graph $G$ with a single source $\mathbf{S} \in \mathbb{R}^{K\times C\times L}$.
In other words, both graph and source are set to a batch size of $1$.
In this setup, a common approach is to compute each processor "one-by-one" in a topological order.
However, observe that we can process multiple processors of the same type simultaneously.

Node Subset Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifically, consider a sequence of $N+1$ node subsets
$V_0, \cdots, V_N \subset V$ satisfying the followings.

* It forms a *partition*: $\cup_n V_n = V$ and $V_n\cap V_m = \emptyset$ if $n \neq m$.
* It is *causal*: no path from $u \in V_n$ to $v \in V_m$ exists if $n \geq m$.
* Each subset $V_n$ is *homogeneous*: it has only a single type $t_n$.

Then, we can compute a batch of output signals 
$\mathbf{Y}_n^{(1)}, \cdots, \mathbf{Y}_n^{(N_n)} \in \mathbb{R}^{\left|V_n\right| \times C\times L}$ 
of each subset $V_n$ sequentially, from $n=0$ to $N$.
Consequently, we reduce the number of the gather-aggregate-process iterations from $|V|$ to $N$ 
(we have no processings for $n=0$ as $V_0$ contains input modules). 
We call this approach *node-level parallelism*.

  The sequence length $N+1$ will vary depending on structure of the graph $G$. 
  The worst case is when the graph $G$ is a serial chain; it results in $N + 1 = \left|V\right|$.
  However, in many cases, we can find a much shorter sequence, i.e., $N + 1 \ll \left|V\right|$.
  This is because, in many cases, the processing of audio (especially musical) involves independent processing of multiple sources
  with common types of processors. 
  A good example is music mixing with a *mixing console*, where each source goes through 
  a *channel strip* that comprises the same serial chain processors.

See the following Figure for an example. 
For a graph with $|V|=107$ nodes, there is a node subset sequence with $N=14$.

.. _type-scheduling:

Type Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To maximize the batched node processing, we want to find the shortest node subset sequence.
.. However, the existence of such short sequences does not gaurantee that we can find them easily (especially the optimal one).
This is a variant of the scheduling problem. 
First, we always choose a maximal subset $V_i$ when the type $t_i$ is fixed. 
This makes the subset sequence equivalent to a sequence of types, e.g., :python:`["in", "eq", "compressor", ...]`.
We also choose the first and the last subset, $V_0$ and $V_N$, 
to have all of the :python:`"in"` and :python:`"out"` nodes, respectively.

* **Fixed ---** In some cases, from the way the graph is constructed, we know a priori a fixed type sequence that is optimal (or its supersequence that contains the optimal). 
  For example, the below graph was first constructed with a fixed chain of processors, and then got pruned to the current graph :cite:`lee2024searching`.
  In such a case, we know that the optimal order is a subsequence of the original processor chain.

  .. figure:: ../imgs/fixed.svg
   :scale: 12 %

* **Greedy method ---** If we cannot use the :python:`"fixed"` method, we have to find a short sequence by our own. However, since the search tree for the shortest sequence exponentially grows,
  the brute-force search is too expensive for most graphs. 
  Instead, we can try the :python:`"greedy"` method that chooses a type with the largest number of computable nodes.

  .. figure:: ../imgs/greedy.svg
   :scale: 12 %

* **Beam search ---** The greedy method usually finds a longer sequence and slows down the processing.
  We can alleviate this with the beam search, i.e., keeping multiple best $W>1$ schedules as candidates instead of one.  
  By default, we use $W=32$.

  .. figure:: ../imgs/beam.svg
    :scale: 12 %

* **One-by-one ---** Finally, we can ignore the batched processing and compute each node one by one. 

  .. figure:: ../imgs/one-by-one.svg
    :scale: 12 %


Which method should we use for our graphs? The general rule of thumb is to use the :python:`"beam"` method 
unless you already know the optimal sequence (use :python:`"fixed"` in this case).
Sometimes, graphs are not parallelizable at all (e.g., being a simple serial chain); 
in such cases, the :python:`"one-by-one"` can be the best choice, 
as it bypasses some additional overheads of the batched processing (albeit small).
The type sequence and the render order can be computed with the following code.

.. code-block:: python

   from grafx.render import compute_render_order
   type_sequence, render_order = compute_render_order(G_t, method="beam")

.. 
  Note that we can visualize the render order with :python:`draw_grafx` 
  and passing the :python:`inside_node="rendering_order"`.
  .. code-block:: python
      from grafx import compute_render_data
      fig, ax = draw_grafx(G, inside_node="rendering_order")
      fig.savefig("beam.pdf")

Note that we can further optimize the batched node processing by reordering the nodes 
so that the memory access becomes contiguous (e.g., read with :python:`torch.narrow`) when possible.
To achieve this, we can use the following instead of the above code.

.. code-block:: python

   from grafx.render import reorder_for_fast_render
   G_t = reorder_for_fast_render(G_t, method="beam")

The Remaining Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once we computed the render order (and optionally reordered the nodes),
we can compute metadata that describes the sequence of all computations required,
including the reads, aggreations, processes, and writes.

.. code-block:: python

    render_data = prepare_render(G_t)

:python:`print(render_data)` will describe the rendering sequence as follows.

.. code-block:: python

    Render #0
      - Node type: in
      - Source read: none with []
      - Aggregation: none
      - Parameter read: slice with (0, 3)
      - Dest write: slice with (0, 3)

    Render #1
      - Node type: eq
      - Source read: slice with (0, 3)
      - Aggregation: none
      - Parameter read: slice with (0, 3)
      - Dest write: slice with (3, 6)

    Render #2
      - Node type: compressor
      - Source read: slice with (3, 6)
      - Aggregation: none
      - Parameter read: slice with (0, 3)
      - Dest write: slice with (6, 9)

    Render #3
      - Node type: reverb
      - Source read: slice with (6, 9)
      - Aggregation: none
      - Parameter read: slice with (0, 3)
      - Dest write: slice with (9, 12)

    Render #4
      - Node type: out
      - Source read: slice with (9, 12)
      - Aggregation: sum
      - Parameter read: slice with (0, 1)
      - Dest write: slice with (12, 13)

Note that, all the above pre-processings can be done in CPU with seperate threads (i.e., by the dataloader workers)
so that the GPU is not blocked by these pre-processings.
Finally, we can compute the output audio with the following code.
The :python:`processors` and :python:`parameters` are the dictionaries that we introduced in the previous section, respectively.

.. code-block:: python

    import torch
    from grafx.render import render_grafx
    source = torch.randn(4, 2, 2**17)
    output, intermediates = render_grafx(source, processors, parameters, render_data) 

Where :python:`output` will contain a :python:`FloatTensor` of shape :python:`(1, 2, 2**17)`

.. 
  One more note --- the :python:`"mix"` is a special type that, in fact, does nothing; it just outputs the sum of its inputs.
  However, it becomes useful when we want to aggregate such summations in parallel
  (see the :ref:`batched processing <batched-audio-processing>` for more details).


.. ---------------------------

Other Parallelisms
---------------------------
With the batched node processing, the remaining parallelisms are straightforward.

Source-Level 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a given graph $G$, 
we can process a batch of sources 
$\mathbf{S}_\mathrm{batch} = [\mathbf{S}_1, \cdots, \mathbf{S}_B] \in \mathbb{R}^{B\times K\times C\times L}$ with the same code; the implementation is almost identical 
(with some tensor reshapes & repeats added).

.. code-block:: python

    souurce = torch.randn(16, 4, 2, 10000)
    output, intermediates = render_grafx(source, processors, parameters, render_data) 

Graph-Level 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also batch multiple graphs $G_1, \cdots, G_B$ by treating them as a single large disconnected graph
$G_\mathrm{batch} = (\cup_{i=1}^B V_i, \cup_{i=1}^B E_i)$.

.. code-block:: python

    from grafx.data import batch_grafx
    G_list = [G for _ in range(4)]
    G_batch = batch_grafx(G)

Then, its corresponding source will be a node-axis concatenation of the individual sources:
$\mathbf{S}_\mathrm{batch} = \mathbf{S}_1 \oplus \cdots \oplus \mathbf{S}_B\in \mathbb{R}^{ {K}_\mathrm{batch} \times C\times L}$
where $\smash{{K}_\mathrm{batch} = \sum_{i=1}^B K_i}$.
The output computation will be the same as above.
Note that, conceptually, the source-level parallelism is a special case of the graph-level parallelism; we obtain the former when we set $G_1 = \cdots = G_B = G$ to the latter.

.. However, they are different in the implementation. 

.. 
  * The former creates $B$ copies of the same graph and the source has $3$ dimensions: 
    $\mathbf{S}_\mathrm{batch} \in \mathbb{R}^{(B\times K)\times C\times L}$.
  * The latter keeps a single graph and the source has $4$ dimensions: 
    $\mathbf{S}_\mathrm{batch} \in \mathbb{R}^{B\times K\times C\times L}$.


