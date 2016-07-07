Installation
============

Python Mapper does not contain any platform-specific code and depends only on cross-platform packages. Hence, it should run on almost all modern operating systems.

So far, installation has been tested under:

* Arch Linux
* Ubuntu (see also here: :doc:`installation_tips_ubuntu`)
* Microsoft Windows 8.1 (see also here: :doc:`installation_tips_windows`)
* OS X Mountain Lion (see also here: :doc:`installation_tips_osx`)

If you install Python Mapper on a certain platform not in the list above, please let me (`Daniel <http://danifold.net>`_) know so that I can extend the list. Especially if you needed to tweak or modify something, I am interested to know about it.

Requirements
------------

* `Python <http://www.python.org/>`_ 2.6 or higher. The GUI needs Python 2 since it depends on wxPython and PyOpenGL; Python Mapper itself can be run under Python 2 and Python 3. I also recommend to install `pip <https://pip.pypa.io/>`_ if it is not already included in your Python distribution. (Without pip, replace ``pip install`` with ``easy_install`` in the command lines below.)
* `NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org/>`_
* `Matplotlib  <http://matplotlib.sourceforge.net/>`_
* `Graphviz <http://www.graphviz.org/>`_
* Optionally cmappertools. Python Mapper will run without this module, but with limited functionality.

  -  cmappertools need the `Boost C++ libraries <http://www.boost.org/>`_.

For the GUI:

* `wxPython <http://www.wxpython.org/>`_
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_

Highly recommended:

* cmappertools

  This is Daniel Müllner's Python module, written in C++, which replaces some slower Python routines by fast, parallelized C++ algorithms. Currently, it is in the ``cmappertools`` subdirectory of the Python Mapper distribution. Cmappertools are hosted on `PyPI <https://pypi.python.org/pypi/cmappertools>`_, so they can be installed with:

  .. code-block :: bash

    $ pip install cmappertools --user

* `fastcluster <http://danifold.net/fastcluster.html>`_

  This is Daniel Müllner's C++ library for fast hierarchical clustering, again wrapped as a Python module. Again, it can be installed from `PyPI <https://pypi.python.org/pypi/fastcluster>`_ by:

  .. code-block :: bash

    $ pip install fastcluster --user

  If this does not work, please refer to the `detailed installation instructions <http://cran.r-project.org/web/packages/fastcluster/INSTALL>`_ in the fastcluster distribution. You need the Python interface; the R interface can be ignored.

Installation
------------

Simply type

.. code-block :: bash

  $ pip install mapper --user

on a command line.

If everything worked, you may stop here and start using Mapper. The steps below describe alternatives and optional steps.

Source distribution
-------------------

If the simple method above does not work, the source distribution of Python Mapper can be downloaded here:

.. admonition:: Download link for Python Mapper

   http://danifold.net/mapper/mapper.tar.gz

Since Python Mapper is not stable yet and under active development, the distribution will be updated frequently. If you want to follow updates more easily and avoid to install the same package over and over again, it is recommended to use the `Mercurial <http://mercurial.selenic.com/>`_ repository. Create a local copy of the repository with:

.. code-block :: bash

  $ hg clone http://danifold.net/hg/mapper

To update the repository, type:

.. code-block :: bash

  $ cd mapper
  $ hg pull
  $ hg up

Installation from source
------------------------

The Python Mapper archive can be extracted anywhere. There is a setup script in the ``mapper`` directory, which can be run with:

.. code-block :: bash

  $ python setup.py install --user

Alternatively, no real installation is necessary. Python just needs to know the location of the package. For this, add the directory where the files were extracted to `Python's search path <http://docs.python.org/2/install/#inst-search-path>`_. (Ie., add the directory which contains ``mapper`` as a subdirectory to the Python path.)

Users may also want to add a link to the ``mapper/bin/MapperGUI.py`` script in a directory which is searched for executables. For example, my ``.bashrc`` contains a line

.. code-block :: bash

  export PATH="${PATH+$PATH:}$HOME/.local/bin"

so I can add a link to the GUI by:

.. code-block :: bash

  $ cd ~/.local/bin
  $ ln -s (MAPPER PATH)/bin/MapperGUI.py

Troubleshooting
---------------

If the GUI refuses to start with an error message like ::

  /usr/bin/env: python2: No such file or directory

there are three ways to deal with the problem:

* Do not call ``MapperGUI`` as an executable script but ``(your Python 2 interpreter) (your path)/MapperGUI.py``, eg.:

  .. code-block :: bash

    $ python mapper/bin/MapperGUI.py

* Create a symbolic link like:

  .. code-block :: bash

    $ sudo ln -s (path to the Python 2 interpreter) /usr/local/bin/python2

* Change the first line in ``MapperGUI.py`` from ::

    #!/usr/bin/env python2

  to::

    #!/usr/bin/env (your Python 2 interpreter)

  With the last method, however, changes will be lost when Python Mapper is updated.

Compiling the documentation
---------------------------

The HTML documentation (this page!) can be compiled with `Sphinx <http://sphinx-doc.org/>`_:

.. code-block :: bash

  $ cd mapper/doc
  $ make html

If you get an error like ::

  make: sphinx-build2: No such file or directory

use:

.. code-block :: bash

  $ make html SPHINXBUILD=sphinx-build
