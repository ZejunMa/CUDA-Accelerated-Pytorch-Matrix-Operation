from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='zejun_torch_cuda',
    version='1.0',
    author='zejunma',
    author_email='zejunma9@gmail.com',
    description='torch-cuda-acceleration-package',
    long_description='torch-cuda-acceleration-package-developed-from-tutorial',
    ext_modules=[
        CppExtension(
            name='zejun_torch_cuda',
            sources=['interpolation.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)