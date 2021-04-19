# -*- coding: utf-8 -*-

from setuptools import setup
from os import path
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

sys.path.insert(0, "src")
from version import __version__

# Load requirements
requirements = None
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

# If Python3: Add "README.md" to setup.
# Useful for PyPI. Irrelevant for users using Python2.
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

extensions = Extension("polyfitcore",
                         sources=["src/polyfit_pyc.pyx", "src/polyfit.c"],
                         include_dirs=[numpy.get_include(),'/opt/local/include'],
                         library_dirs=['/opt/local/lib'],
                         libraries=['m', 'gsl', 'gslcblas'],
                         extra_compile_args=["-O2","-g"],
                         define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

desc='Polynomial chain fitter and morphology type based classifier.'

setup(name='polyfitter',
      version=__version__,
      description=desc,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Attila Bodi',
      author_email='astrobatty@gmail.com',
      url='https://github.com/astrobatty/polyfitter/',
      package_dir={'polyfitter':'src'},
      packages=['polyfitter'],
      package_data={"polyfitter": ["transformers/*.pickle"]},
      install_requires=requirements,
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(extensions, compiler_directives={'language_level': 3})
     )
