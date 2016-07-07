#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

if sys.hexversion < 0x03000000: # uniform unicode handling for both Python 2.x and 3.x
    def u(x):
        return x.decode('utf-8')
else:
    def u(x):
        return x

#import distutils.debug
#distutils.debug.DEBUG = 'yes'
from setuptools import setup, Extension

with open('__init__.py', 'r') as f:
    for line in f:
        if line.find('__version__ =')==0:
            version = line.split("'")[1].split("'")[0]
            break

print('Version: ' + version)

if sys.hexversion < 0x03000000:
    scripts = ['bin/MapperGUI.py']
else:
    scripts = []
    print('Warning: the Python Mapper GUI is not installed since setup.py is being executed\n'
          'from Python 3. Please install Python Mapper with Python 2 in order to have the\n'
          'GUI. (The "mapper" package itself works with both Python 2 and 3, but the GUI\n'
          'depends on wxPython, which is available for Python 2 only.)')

setup(name='mapper',
      version=version,
      provides=['mapper'],
      packages=['mapper', 'mapper.tools'],
      package_dir={'mapper' : '.'},
      package_data={'mapper' : ['d3js/*', 'exampleshapes/*']},
      scripts=scripts,
      description=('Python Mapper: an open source tool for exploration, analysis and visualization of data.'),
      long_description=('''See the project home page http://danifold.net/mapper
for a detailed description and documentation.

This package features both a GUI and a Python package for custom scripts. The
Python package itself works with Python 2 and 3. The GUI, however, depends on
wxPython, which is available for Python 2 only. Therefore, the setup script will
install the GUI only if it is executed by Python 2.

See also https://pypi.python.org/pypi/cmappertools for the companion package with fast C++ algorithms.

The authors of Python mapper are `Daniel Müllner <http://danifold.net>`_ and `Aravindakshan Babu <mailto:anounceofpractice@hotmail.com>`_. (PyPI apparently suppresses everything but the first name in the “author” field, hence only one author is displayed below.)
'''),
      keywords=['Mapper', 'Topological data analysis', 'TDA'],
      author=u("Daniel Müllner, Aravindakshan Babu"),
      author_email="daniel@danifold.net (Daniel), anounceofpractice@hotmail.com (Aravind)",
      maintainer=u("Daniel Müllner"),
      maintainer_email="daniel@danifold.net",
      license="GPLv3 <http://www.gnu.org/licenses/gpl.html>",
      classifiers = [
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Visualization",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Intended Audience :: Science/Research",
          "Development Status :: 5 - Production/Stable"
      ],
      url = 'http://danifold.net/mapper',
)
