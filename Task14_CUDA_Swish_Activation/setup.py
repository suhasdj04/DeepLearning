from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='swish_cuda',
    ext_modules=[
        CUDAExtension('swish_cuda', [
            'swish_cpp.cpp',
            'swish_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
