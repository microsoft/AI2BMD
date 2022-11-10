from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("OGB_ViSNet/datasets/algos.pyx"),
    include_dirs=[numpy.get_include()]
)