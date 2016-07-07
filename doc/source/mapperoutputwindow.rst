The Mapper output window
========================

Here is a typical Mapper output from one of the toy examples (the horse):

.. image:: /Mapper_output_screenshot_1.png
   :align: center

General navigation
------------------

See here: :doc:`figure_navigation`

Node highlighting
-----------------

*   Click on a node to highlight it.
*   :kbd:`Shift`-Click allows to select and de-select multiple nodes.

    .. figure:: /Mapper_output_screenshot_3c.png
       :align: center

       Mapper output window with several highlighted nodes.

*   Click on the white background to reset the selection.
*   Keyboard navigation

    -   :kbd:`L` selects all nodes in the same level set(s) as the currently highlighted node(s).
    -   :kbd:`<` moves to the next lower level set.
    -   :kbd:`>` moves to the next higher level set.

*   When 2- or 3-dimensional data is displayed, the data points in the selected nodes are highlighted accordingly.

    .. figure:: /Mapper_output_screenshot_3a.png
       :align: center

       Data points in the selected nodes above are highlighted.

*   The identities of the selected data points can be saved to a file, see :ref:`Save highlighted nodes <save_highlighted_nodes>` below.

Menus
-----

“File” menu
```````````

Save figure :kbd:`Ctrl-S`
'''''''''''''''''''''''''

Save the figure in the current view. A variety of file formats is available, for both vector and bitmap graphics.

.. _save_highlighted_nodes:

Save highlighted nodes :kbd:`Ctrl-Shift-S`
''''''''''''''''''''''''''''''''''''''''''

In order to check which actual data points are represented by a set of nodes, the list of data points in the currently highlighted nodes can be saved to a text file. The text file is a simple list of indices (one number per line, separator: newline, *LF*). The first data point has index 0.


“Options” menu
``````````````

Relabel :kbd:`Alt-L`
''''''''''''''''''''

*(Advanced.)* This allows to display custom node labels. A file dialog asks for a Python script with a function that determines the labels from the node data. The Python script must define the following objects:

name
  a string with the name of the labelling scheme (for the figure legend).

label
  a function that is given a node (ie. a ``mapper.mapper_output.node`` object) and must return a string with the node label.

    In short, a ``node`` object contains the following data:

    level
      level identifier, usually an integer or a tuple of integers
    points
      a ``numpy.ndarray(N, dtype=int)``; these are the indices of the points in the data set which are contained in the node.
    attribute
      any attribute, eg. a color, normally the average filter value in the node.

Below is a small example which replaces the standard label (node size) with the average filter value.

.. code-block:: python

   name = 'filter value'

   def label(node):
       return '{:.3n}'.format(node.attribute)

“View” menu
```````````

Reset :kbd:`1`
''''''''''''''

Restore the original view.

Show labels :kbd:`Ctrl-L`
'''''''''''''''''''''''''

Show or hide the node labels. The default node labels are nodes sizes (ie. number of data points in each node).

.. figure:: /Mapper_output_screenshot_4.png
   :align: center

   Mapper output with and without labels.


Resize window :kbd:`Ctrl-R`
'''''''''''''''''''''''''''

Resize the figure window to a given exact size in pixels. Enter eg. ``1000x500`` for a canvas of size 1000×500, or just ``1000`` for width 1000 and height corresponding to the figure aspect ratio.
