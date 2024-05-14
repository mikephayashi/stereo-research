from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("speedup.pyx"),
    annotate=True
)