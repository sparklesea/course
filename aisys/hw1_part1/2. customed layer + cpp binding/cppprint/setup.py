from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

module_name = "cppprint_cpp"

ext_module = CppExtension(
    name=module_name,
    sources=["cppprint.cpp"]
)

setup(
    name=module_name,
    ext_modules=[ext_module],
    cmdclass={
        'build_ext': BuildExtension
    }
)