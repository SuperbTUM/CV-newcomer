from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

compile_args = ['-std=c++11', '-fopenmp']
linker_flags = ['-fopenmp']

module = Extension('scratch',
                   ['scratch.pyx'],
                   language='c++',
                   include_dirs=[np.get_include()],
                   extra_compile_args=compile_args,
                   extra_link_args=linker_flags)

setup(
    name='scratch',
    ext_modules=cythonize(module),
    gdb_debug=False
)
