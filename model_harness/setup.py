from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tony_conv',
    ext_modules=[
        CUDAExtension('tony_conv', [
            'approx_kernel_pytorch.cc',
            'approx_kernel_pytorch_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
