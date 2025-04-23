from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob
import platform

library_name = "voxelproj"
__version__ = "0.0.1"


def get_extensions():
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    is_windows = platform.system() == "Windows"
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "/O2" if is_windows else "-O3",
        ],
        "nvcc": [
            "-O3",
            # "-DTORCH_USE_CUDA_DSA",
            "--use_fast_math",
            # "-allow-unsupported-compiler",                 # These were needed to get it to work on my laptop
            # "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",  # With older cuda support
            # "-lineinfo",        # Include line information for profiler
            # "-src-in-ptx",      # Embed source in PTX for better profiling
            # "-g",               # Include debug information
        ],
    }

    ext_proj = CUDAExtension(
        f"{library_name}._CU",
        sources + cuda_sources,
        extra_compile_args=extra_compile_args,
    )

    return [ext_proj]


setup(
    name=library_name,
    version=__version__,
    author="Axel Ekman",
    author_email="Axel.Ekman@iki.fi",
    description="Accurate parallel projection",
    long_description="NA",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
