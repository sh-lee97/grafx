.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _audio-processing-graphs:

Audio Processing Graphs
===========================

..  

.. ---------------------------

Domain practitioners (e.g., musicians and audio engineers) sculpt and transform their sounds by combining multiple processors,
forming an *audio processing graph* :cite:`lee2023blind`.

  Audio processing graphs are everywhere --- guitar pedalboards, modular synthesizers, and mixing consoles, to name a few.
  The popular JUCE framework has `AudioProcessorGraph <https://docs.juce.com/master/classAudioProcessorGraph.html>`_ class.
  `Web Audio API <https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_Web_Audio_API#audio_graphs>`_ is also based on graphs.
  This ubiquitous structure is not just a coincidence; *compositionality* and *modularity* of processors are essential features 
  that allow practitioners to create and control complex processing with ease.

Nevertheless, recent deep-learning methods for musical signal processing have overlooked this practice, 
assuming fixed processing chains or relying on end-to-end neural network-based methods.
:python:`GRAFX` aims to close this gap --- it provides a collection of functionalities for creating, manipulating, and rendering audio processing graphs.
Especially, it provides the following distinctive features.

.. However, it is not just a reinvention of the existing libraries. 

#. Our graphs can be converted into :python:`PyTorch` tensor representations :cite:`paszke2019pytorch`,
   which can be used for graph representation learning, i.e., as input of graph neural networks (GNNs) :cite:`fey2019fast`.
#. We provide a collection of various :ref:`differentiable audio processors <differentiable-processors>`, including equalizers, dynamic range compressors, reverb, and many more.
#. We provide an :ref:`efficient and flexible method <batched-audio-processing>` for calculating output audio from graphs in GPU.
   Combined with the differentiable processors, 
   we can optimize the graph structure and parameters (or their neural predictors)
   via gradient descent end-to-end :cite:`lee2024searching, lee2023blind, joseph2021reverse`.

This short series of introductions will share the core concepts of :python:`GRAFX`. 
While it generally follows our paper :cite:`lee2024grafx`, 
we extend the discussion to the latest updates and provide practical examples.


.. ---------------------------

Definitions and Notations
---------------------------

Following the standard notation,
we write an audio processing graph as $G=(V, E)$ where $V$ and $E$ are the node and edge set, respectively. 

Nodes 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each node $v_i \in V$ ($i$ denotes a node index) can represent a *processor* $f_i$.
Each node has a type attribute $t_i$, e.g., :python:`"reverb"`. 
It takes $M$ signal(s) $u_i[n]$ and a collection of parameters $p_i$ as input 
and produces $N$ output(s) $y_i[n]$ ($n$ denotes a time index).
$$
\underbrace{y^{(1)}_i[n], \cdots, y^{(N)}_i[n]}_{y_i[n]} = f(\underbrace{u^{(1)}_i[n], \cdots, u^{(M)}_i[n]}_{u_i[n]}, p).
$$

.. Following the terminology in MAX, we refer

.. 
  Nodes also include auxiliary input :python:`"in"` and output :python:`"out"` modules.
  Each input module takes no input signals within the graph; it just outputs a source signal provided by the user.
  Each output module takes no input signals within the graph; it just outputs a source signal provided by the user.


Edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each edge $e_{ij}\in E$ naturally represents a *cable* connecting two nodes, 
Each edge also requires a type attribute $t_{ij}=(k, l)$, 
a tuple of input and output channel indices, 
unless every processor in a graph is a single-input single-output system (SISO), i.e., $M=N=1$.
Each input signal is decided by the connected edges, which can be written as follows,
$$
u_{i}^{(l)}[n] = \sum_{(j, k) \in \mathcal{N}^+(i, l)} y_{j}^{(k)}[n]
$$

where $\mathcal{N}^+(i, l)$ is a collection of nodes and channel indices 
that send their output signals to $l^\mathrm{th}$ input of node $i$. 
In short, each node's outputs are computed by finding its inputs,
aggregating those, and processing the sums with the parameters.

Graph Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We allow any *directed acyclic multigraph* as an audio processing graph.
*Directed* means that the edges have directions, connecting the output of a source to the input of a destination.
*Acyclic* means that are no cycles (or *feedback loop*) in the graph. 
Finally, *multigraph* means that multiple edges can exist between the same pair of nodes unless their types are identical. 
The following figure shows an example of a valid graph used in 
(different letters denote different node types)
:cite:`lee2024searching`.

.. figure:: ../imgs/cambridge_RememberDecember_CUNextTime.svg
   :scale: 11 %

Crucially, the acyclic property allows us to compute the output audio by repeating the 
*gather-aggregate-process* over the entire nodes in topological order, from the inputs to the output
(this is impossible if there is a cycle or *feedback loop*).
Usually, we have $K$ inputs and a single output.
In such a case, the output signal can be written as 
$$
y[n] = G(s_1[n], \cdots, s_K[n]; \mathbf{P})
$$

where $s_k[n]$ is a source signal that corresponds to the $k^\mathrm{th}$ :python:`"in"` node 
and $\mathbf{P}$ is a collection of all parameters in the graph.
We can also write the signals as tensors,
$\mathbf{S} \in \mathbb{R}^{K\times C\times L}$ and $\mathbf{Y} \in \mathbb{R}^{1\times C\times L}$,
where
$K$, $C$, and $L$ are the number of sources, channels, and length, respectively. 
This simplifies the above equation to $\mathbf{Y} = G(\mathbf{S}; \mathbf{P})$.


.. ---------------------------

Creating Graphs
---------------------------

For creating and manipulating the audio processing graphs,
we provide a mutable data structure :class:`~grafx.data.graph.GRAFX`
(same as the library name). 
It inherits :python:`MultiDiGraph` class from :python:`networkx` :cite:`hagberg2008exploring` 
and provides additional functionalities, e.g., adding a serial chain of processors.

Node Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before creating a graph, we need to pre-define allowed processor types 
(and their additional configurations, if needed).
This procedure is necessary when we use our tensor representation or compute the output audio of the graph.
We provide a :class:`~grafx.data.configs.NodeConfigs` class for this purpose.
Suppose that we have three processor types, :python:`"eq"`, :python:`"compressor"`, and :python:`"reverb"`.

.. code-block:: python

    from grafx.data import NodeConfigs
    config = NodeConfigs(["eq", "compressor", "reverb"])

:python:`print(config)` will give the following output.

.. code-block:: bash

   NodeConfigs with 6 node types (siso_only=True)               
     (0) in: None -> <main>                                     
     (1) out: <main> -> None                                    
     (2) mix: <main> -> <main>                                  
     (3) eq: <main> -> <main>                                   
     (4) compressor: <main> -> <main>                           
     (5) reverb: <main> -> <main>   

With the outputs, we note the following.

* Along with the processor types, 
  auxiliary :python:`"in"`, :python:`"out"`, :python:`"mix"` are also included by default.
* The ``eq: <main> -> <main>`` denotes that the :python:`"eq"` processor has a single input and output.
  The ``None`` denotes there is no input or output.

  .. When the processors are all SISO systems, we set :python:`config.siso_only` to :python:`True`.

* When the processors are provided as a :python:`list`, they are all assumed to be SISO systems.
  To set up MIMO systems, provide
  a :python:`dict` of types as keys and their inlet/outlet configurations as values.

An Empty Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we can create an empty graph the :class:`~grafx.data.config.NodeConfigs`.

.. code-block:: python

   from grafx.data import GRAFX
   G = GRAFX(config=config)

Here, :python:`print(G)` will give the following output. 

.. code-block:: bash

   GRAFX with 0 nodes & 0 edges

Basic Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, we add an :python:`"out"` node to the graph with the :python:`add` method.

.. code-block:: python

   out_id = G.add("out") 

The :python:`add` returns the added node's integer node ID, which can be used to access its attributes or connect to other nodes.
Now :python:`print(G)` will give the following.

.. code-block:: bash

   GRAFX with 1 nodes & 0 edges                  
     [0] out                                                                                                                 

If we try to add a node type that is not provided to :class:`~grafx.data.config.NodeConfig`, it will raise an error. 
For example, if we try :python:`G.add("noisegate")`,

.. code-block:: bash

   Exception: Invalid node_type: noisegate, it only allows ['in', 'out', 'mix', 'eq', 'compressor', 'reverb'.

We can also add a serial chain of processors to the graph with :func:`~grafx.data.graph.GRAFX.add_serial_chain` method, 
which returns the ID of the start and end nodes. 
Other nodes can be accessed by the id in between.
Also, we can connect the nodes with :func:`~grafx.data.graph.GRAFX.connect` method, providing the source and destination node IDs.


.. code-block:: python

   number_of_sources = 3
   for _ in range(number_of_sources):
       chain = ["in", "eq", "compressor", "reverb"] 
       start_id, end_id = G.add_serial_chain(chain) 
       G.connect(end_id, out_id) 

The above code will add three sources, each connected to an equalizer, compressor, and reverb in series.
Them, their outputs, i.e., the reverb outputs, are connected to the output node.
Now, our graph is as follows.

.. code-block:: bash

    GRAFX with 13 nodes & 12 edges
      [0] out
      [1] in -> [2] eq
      [2] eq -> [3] compressor
      [3] compressor -> [4] reverb
      [4] reverb -> [0] out
      [5] in -> [6] eq
      [6] eq -> [7] compressor
      [7] compressor -> [8] reverb
      [8] reverb -> [0] out
      [9] in -> [10] eq
      [10] eq -> [11] compressor
      [11] compressor -> [12] reverb
      [12] reverb -> [0] out

.. 
   You can also visualize the graph with :python:`draw_grafx`.

   .. code-block:: python

      from grafx.draw import draw_grafx
      fig, ax = draw_grafx(G, node_above="node_id", vertical=False)
      fig.savefig("graph.pdf")


   .. figure:: ../imgs/example.svg
      :scale: 11 %

.. ---------------------------

Tensor Representations
---------------------------

Once the graphs are created, they can be used to compute output audio or fed into a GNN. 
In such cases, representing each graph as a collection of tensors is more convenient and efficient. 
Therefore, we provide a :class:`~grafx.data.tensor.GRAFXTensor` class, 
which is compatible with :python:`Data` class from :python:`torch\_geometric` :cite:`fey2019fast`.
The processor types are assigned with integer values,
The following are the tensors we use to describe each graph.

#. A node type vector $\mathbf{T}_V \in \mathbb{N}^{\left|V\right|}$.
   The mapping between the node type and its integer value is determined by the predefined :python:`NodeConfig` object.
#. An edge index tensor $\mathbf{E} \in \mathbb{N}^{2\times\left|E\right|}$, 
#. (*Optional*) An edge type tensor $\mathbf{T}_E \in \mathbb{N}^{2\times \left|E\right|}$ 
   where $\left|\cdot \right|$ denotes the size of a given set.
   Only used when the :class:`~grafx.data.configs.NodeConfigs` contains MIMO processors.
#. (*Optional*) A collection of all parameters in a dictionary $\mathbf{P}$ 
   (or any reasonable :python:`Mapping` such as :python:`nn.ParameterDict`) 
   whose key is a node type $t$ and value contains the parameters of that type.

.. 
   We ensure that all the tensors, 
   $\mathbf{T}_V$, $\mathbf{T}_E$, $\mathbf{E}$, $\mathbf{P}$, and $\mathbf{S}$, 
   share the same node order.
   For example, 
   a $k^\mathrm{th}$ source $s_k[n]$ must correspond to the first $k^\mathrm{th}$ :python:`"in"` (or :python:`0` by default)
   in the node type list $\mathbf{T}_V$.
   Likewise, an $l^\mathrm{th}$ type-$t$ parameter $\mathbf{P}[t]_l \in \mathbb{R}^{N_t}$ 
   must correspond to the $l^\mathrm{th}$ type $t$ in the type list $\mathbf{T}_V$.

We can obtain the tensors by converting the :class:`~grafx.data.graph.GRAFX` graph with :func:`~grafx.data.conversion.convert_to_tensor`.

.. code-block:: python

   from grafx.data import convert_to_tensor
   G_t = convert_to_tensor(G)

Here, :python:`print(G_t)` gives the followings,

.. code-block:: bash

   GRAFXTensor(
     node_types=[13], 
     edge_indices=[2, 12], 
     edge_types=None,              
     rendering_order_method=None, 
     rendering_orders=None,           
     type_sequence=None,          
     counter=13, 
     batch=False, 
     config=<grafx.data.configs.NodeConfigs object at 0x7ff43769c760>, 
     config_hash=8792929901686,    
     invalid_op='error'
   )                                  

It contains :python:`node_types`, :python:`edge_indices`, and others that are useful for several purposes, 
e.g., computing output audio.

.. 
   It will corresponds to :python:`node_types` attribute in :python:`GRAFXTensor`.

Processor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We further elaborate on the parameter dictionary $\mathbf{P}$ as it is the most nontrivial part of our tensor representation.
It has the dictionary type as each processor type can have a different set and number of parameters.
All parameters of type $t$ are collected to a dictionary value $\mathbf{P}[t]$, 
which can be either a single tensor or a dictionary of tensors.
When the latter case holds, $\mathbf{P}$ becomes a nested dictionary.
Every parameter tensor for type $t$ must have a first dimension that has the number of nodes of that type $|V_t|$.
Also, each tensor follows the node ordering of the node type vector $\mathbf{T}_V$.
For example, parameters of a $n^\mathrm{th}$ type-$t$ node corresponds to $n^\mathrm{th}$ element of each tensor in $\mathbf{P}[t]$, i.e., 
$\mathbf{P}[t][n]$ or $\mathbf{P}[t][k][n]$ for all $k$ where $k$ is a key of the $\mathbf{P}[t]$.
This way, we can easily access the parameters of specific nodes of that type.

Note that, throughout the above example, we did not provide any parameter while adding the nodes.
This is because one of our main applications is to estimate the parameters from some references
by direct parameter optimization or training a neural network as a predictor.
In this scenario, we only have the connectivity information $(\mathbf{T}_V, \mathbf{E}, \mathbf{T}_E)$, 
i.e., *what is connected to what*.
A previous work :cite:`lee2023blind` named it a *prototype graph* (and denoted with $G_0$).
Many existing works assume that this prototype graph is given (fixed in many cases) 
and predict its parameters $\mathbf{P}$ 
:cite:`uzrad2024diffmoog, caspe2022ddx7, steinmetz2020diffmixconsole, ramirez2021differentiable, colonel2023music`,
Of course, there are a few exceptions where the connectivity is also estimated 
(albeit, in most cases, the underlying connectivity $G_0$ is a simple serial chain) 
:cite:`ye2023fm, Mitcheltree_2021, guo2023automatic, lee2023blind, lee2024searching`.
Refer to Appendix A of :cite:`lee2024searching` for a comparative review on this matter.
Throughout this series of posts, we will not distinguish the prototype from the *full* graph 
$(\mathbf{T}_V, \mathbf{E}, \mathbf{T}_E, \mathbf{P})$ as it is mostly clear from the context.
More details on the parameters will be in the following posts on the 
:ref:`processors <differentiable-processors>` and 
:ref:`rendering <batched-audio-processing>`.

.. 
   Notes
   -----------------------------

   Here, we note several considerations.

   * 


   * The acylclic property restricts possible graph structures.
   However, this is necessary to compute output audio efficiently.
   Delay-less loops must be resolved analytically, which is highly nontrivial.
   Even when possible, automating this is another challenge.
   Furthermore, even without the delay-less loops, the forced sample-level recursion bottlenecks the speed.
   Hence, when the feedback loops are necessary, we can encapsulate them into a node and optimize it with, 
   e.g., JIT compile or other methods.
   * We 
  .. Note that we are implicitly assuming that nodes with the same type must have the same parameter shapes.
