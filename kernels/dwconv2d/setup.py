from setuptools import setup
from torch.utils import cpp_extension

from Cython.Build import cythonize
setup(
    name ='dwconv2d',   
    ext_modules=[cpp_extension.CUDAExtension('dwconv2d', 
    ['dwconv2d.cpp', 
    'depthwise_fwd/launch.cu',
    ],)
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['Dwconv']
)

# CUDAExtension
# CppExtension
