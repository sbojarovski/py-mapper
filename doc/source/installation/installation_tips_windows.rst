Installation instructions: Windows
==================================

Here are step-by-step instructions to install Python Mapper and all its dependencies in Windows.

#. Download and install Python 2.

   Go to https://www.python.org/downloads/windows/. Go to “Latest Python 2 Release”. Download the
   “Windows x86-64 MSI installer”. Start the installer and follow the instructions.

#. Download and install wxPython from http://wxpython.org/download.php. Make sure to match your Python
   version and architecture (32/64bit).

#. Download the Windows NumPy package from http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy. You'll need
   the package later. Again, make sure to match your Python version and architecture.

#. Download the Windows SciPy package from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy.

#. Download and install Graphviz from http://www.graphviz.org/Download.php.

#. Open a Command Prompt.

#. The directory with the Graphviz executables must be appended to the ``PATH`` environment variable.
   There are lots of tutorials for this step in the internet, eg. here:
   https://www.java.com/en/download/help/path.xml.

   (On my system, the directory is ``C:\Program Files (x86)\Graphviz2.38\bin``.)

#. Close the command prompt and open a new one to process the changes in the ``PATH`` variable.

#. Check that the search path is configured correctly by typing::

     neato -?

   If Graphviz's “neato” program responds with a help message, everything is alright.

#. Go to the ``Scripts`` folder in your Python installation, eg.::

     cd /d C:\Python27\Scripts

#. Type::

     pip install wheel

#. Install the downloaded NumPy and SciPy packages as follows::

     pip install C:\Users\MYUSERNAME\Downloads\numpy-A.B.C+mkl-cp2X-none-win_amd64.whl
     pip install C:\Users\MYUSERNAME\Downloads\scipy-A.B.C-cp2X-none-win_amd64.whl

#. Install the remaining Python packages::

     pip install matplotlib
     pip install pyopengl
     pip install fastcluster
     pip install mapper
     pip install cmappertools

#. The Mapper GUI should work now::

     C:\Python27\Scripts\MapperGUI.py

   I recommend to always execute the GUI from the command line. This way, debugging information is
   displayed on the terminal if an error occurs.

#. `Optional:` You may want to add the Python ``Scripts`` directory to the search path, in the same way
   as you did it for Graphviz. On my system, it is ``C:\Python27\Scripts``. This way, the Mapper GUI
   can be started by the command ``MapperGUI.py`` from any directory.
