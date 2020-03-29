from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='MaskRCNN Extension',
      ext_modules=[CUDAExtension('_C', ['nms.cpp', 'nms_cpu.cpp', 'nms_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})