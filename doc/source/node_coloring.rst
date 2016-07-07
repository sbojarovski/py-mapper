.. _node_coloring_page:

Node coloring
=============

By default, the nodes in the Mapper output are colored by the average filter value: for all points in a node, the average filter value is computed, and then a color map is applied to all nodes. Currently, the color map is Matplotlib's default “jet” color map, with a range from the lowest to the highest filter value of all points. Low filter values are represented by blue, high filter values by red.

.. figure:: /jet.png
   :align: center

   The “jet” color map

The Python Mapper GUI allows user code to assign arbitrary scalar values to nodes for a different coloring. Code can be entered in the field “Node coloring” in Step 5:

.. image:: /inputfields/node_coloring.png

As in the :ref:`data_processing` and :ref:`filter_processing` examples, any Python code can be entered. In particular, new modules may be imported for more complex procesing.

The following variables are predefined when the Python interpreter processes the code:

.. py:data:: f
  :noindex:

  The filter function, a 1-dimensional ``numpy.ndarray`` with ``double`` data type and length equal to the number of data points.

.. py:data:: data
  :noindex:

  The input data of the Mapper algorithm, a ``numpy.ndarray`` with ``double`` data type. If it is one-dimensional (``len(data.shape)==1``), it is a compressed array of pairwise distances. Otherwise, it contains vector data.

.. py:data:: nodes

   A list of ``mapper.mapper_output.node`` objects. If information contained in the nodes should be extracted, please refer to the source file ``mapper_output.py`` for details.

.. py:data:: node_color

   Initially ``None``. If this is assigned a ``numpy.ndarray`` of shape *(n,)* for *n* nodes, then the nodes are colored according to the scalar values in this array.

.. py:data:: point_color

   Initially ``None``. If this is assigned a ``numpy.ndarray`` of shape *(N,)* for *N* data points, then the nodes are colored according to the average value for all points in a node.

.. py:data:: name

   Initially the string ``'custom scheme'``. Replace it with the name of the coloring scheme for the figure legend.

.. py:data:: np
  :noindex:

   This gives access to the `NumPy <http://numpy.scipy.org/>`_ package.

Only one of the variables ``node_color`` and ``point_color`` may be given a value different from ``None``.

As a simple example, the line ::

   name = 'z-coordinate'; point_color = data[:,2]

colorizes vector data according to the 3rd coordinate. With this coloring, the horse example looks like this:

.. figure:: /Mapper_output_screenshot_5.png
   :align: center

   Horse colorized by the 3rd coordinate.

The color map ranges from dark blue at the tail to red at the head.
