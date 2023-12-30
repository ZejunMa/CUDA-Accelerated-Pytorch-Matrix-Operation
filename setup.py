import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='zejun_cuda',
    version='1.0',
    author='zejunma',
    author_email='zejunma9@gmail.com',
    description='Pytorch CUDA/C++ Extension',
    long_description='Pytorch CUDA/C++ Extension',
    ext_modules=[
        CUDAExtension(
            name='zejun_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)