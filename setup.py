from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import platform
import shutil


class CUDABuild(build_ext):
    def run(self):
        # Compile CUDA kernels
        cuda_files = [
            "sp_forward_z0.cu",
            "sp_forward_z2.cu",
            "sp_backward_z0.cu",
            "sp_backward_z2.cu",
        ]  # List your .cu files here

        common_flags = [
            "nvcc",
            "-arch=native",
            "-O3",  # Maximum optimization level
            "--use_fast_math",  # Fast math operations
            "--cubin",
        ]

        is_windows = platform.system() == "Windows"

        if is_windows:
            # Windows-specific compiler flags
            platform_flags = [
                "-Xcompiler=/O2",  # Host compiler optimization for speed
                "-Xcompiler=/MD",  # Multi-threaded DLL runtime
                "-Xcompiler=/Zi",  # Debug information
            ]
        else:
            # Linux/Unix-specific compiler flags
            platform_flags = [
                "-Xcompiler=-O3",  # Host compiler optimization
                "-Xcompiler=-fPIC",  # Position-independent code
                "-Xcompiler=-Wall",  # Enable warnings
                "--compiler-options=-ffast-math",  # Fast math for host code
            ]

        for cuda_file in cuda_files:
            output_file = os.path.splitext(cuda_file)[0] + ".cubin"
            input_path = os.path.join("voxelproj", "cuda", cuda_file)
            output_path = os.path.join("voxelproj", "cuda", output_file)
            print(f"Compiling {cuda_file} to {output_file}...")
            file_args = [input_path, "-o", output_path]

            subprocess.check_call(common_flags + platform_flags + file_args)
        build_ext.run(self)


setup(
    name="voxelproj",
    version="0.1",
    packages=find_packages(),
    package_data={
        "voxelproj": ["cuda/*.ptx"],  # Include compiled PTX files
    },
    cmdclass={
        "build_ext": CUDABuild,
    },
    install_requires=[
        "numpy",
        "pycuda",
    ],
    python_requires=">=3.6",
    author="Axel Ekman",
    author_email="axel.a.ekman@gmail.com",
    description="Exact parallel projections with separabel footprints.",
    keywords="tomgrpahy",
)
