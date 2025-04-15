import os
import cupy as cp
import numpy as np


def get_kernel_cupy(file, kernel_name):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptx_path = os.path.join(module_path, "cuda", file)
    module = cp.RawModule(path=ptx_path)
    kernel = module.get_function(kernel_name)
    return kernel


def create_texture_from_array(array):
    """
    Create a CUDA texture object from a 3D array using CuPy.
    """
    # Convert to CuPy array if it's numpy
    if isinstance(array, np.ndarray):
        gpu_array = cp.array(array, dtype=cp.float32)
    else:
        gpu_array = array.astype(cp.float32) if array.dtype != cp.float32 else array

    # Create a CUDA array for the texture
    channel_desc = cp.cuda.texture.ChannelFormatDescriptor(
        32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat
    )

    # For 3D arrays, need width (fastest changing dimension), height, and depth
    # In CUDA, dimensions are ordered as width, height, depth which corresponds to
    # the reversed order of numpy/cupy array dimensions (which are z, y, x)
    width, height, depth = gpu_array.shape[::-1]

    # Create 3D CUDA array
    cuda_array = cp.cuda.texture.CUDAarray(
        desc=channel_desc,
        width=width,  # x dimension
        height=height,  # y dimension
        depth=depth,  # z dimension
    )

    # Copy data from GPU array to CUDA array
    cuda_array.copy_from(gpu_array)

    # Create resource descriptor
    res_desc = cp.cuda.texture.ResourceDescriptor(
        restype=cp.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_array
    )

    # Create texture descriptor with 3 dimensions
    tex_desc = cp.cuda.texture.TextureDescriptor(
        addressModes=[
            cp.cuda.runtime.cudaAddressModeBorder,  # x dimension
            cp.cuda.runtime.cudaAddressModeBorder,  # y dimension
            cp.cuda.runtime.cudaAddressModeBorder,  # z dimension
        ],
        # filterMode=cp.cuda.runtime.cudaFilterModeLinear,
        filterMode=cp.cuda.runtime.cudaFilterModePoint,
        readMode=cp.cuda.runtime.cudaReadModeElementType,
        normalizedCoords=0,
    )

    # Create and return texture object
    return cp.cuda.texture.TextureObject(res_desc, tex_desc)


def gpu_array_with_default_shape(x=None, shape=None):
    if x is None:
        x_gpu = cp.zeros(shape, dtype=cp.float32)
    elif isinstance(x, tuple):
        # If x is a shape tuple
        output_shape = x
        x_gpu = cp.zeros(output_shape, dtype=cp.float32)
    elif isinstance(x, (np.ndarray, cp.ndarray)):
        # If x is an array
        x_gpu = cp.asarray(x, dtype=cp.float32)
    else:
        raise TypeError("x must be None, a shape tuple, or an array")
    return x_gpu


def forward_z0(x, angles, y, block=None):
    x_gpu = cp.asarray(x, dtype=cp.float32)
    a_gpu = cp.asarray(angles, dtype=cp.float32)

    L_max, H = max(x.shape[0], x.shape[1]), x.shape[2]
    y_gpu = gpu_array_with_default_shape(y, shape=(len(angles), L_max, H))

    Nv = cp.array(x_gpu.shape[::-1], dtype=cp.int32)
    Np = cp.array(y_gpu.shape[::-1], dtype=cp.int32)

    tex_obj = create_texture_from_array(x_gpu)
    kernel = get_kernel_cupy("sp_forward_z0.cubin", "sp_forward_z0")
    if block is None:
        block = (8, 8, 8)
    grid = tuple((s + b - 1) // b for s, b in zip(y_gpu.shape[::-1], block))
    kernel(grid, block, (tex_obj, y_gpu, a_gpu, *Nv.tolist(), *Np.tolist()))

    # Return result as cupy array if input was cupy array
    if isinstance(x, cp.ndarray):
        return y_gpu
    else:
        return y_gpu.get()


def forward_z2(x, angles, y, block=None):
    x_gpu = cp.asarray(x, dtype=cp.float32)
    a_gpu = cp.asarray(angles, dtype=cp.float32)

    L_max, H = max(x.shape[0], x.shape[1]), x.shape[2]
    y_gpu = gpu_array_with_default_shape(y, shape=(H, len(angles), L_max))

    Nv = cp.array(x_gpu.shape[::-1], dtype=cp.int32)
    Np = cp.array(y_gpu.shape[::-1], dtype=cp.int32)

    tex_obj = create_texture_from_array(x_gpu)
    kernel = get_kernel_cupy("sp_forward_z2.cubin", "sp_forward_z2")
    if block is None:
        block = (8, 8, 8)
    grid = tuple((s + b - 1) // b for s, b in zip(y_gpu.shape[::-1], block))
    kernel(grid, block, (tex_obj, y_gpu, a_gpu, *Nv.tolist(), *Np.tolist()))

    # Return result as cupy array if input was cupy array
    if isinstance(x, cp.ndarray):
        return y_gpu
    else:
        return y_gpu.get()


def backward_z0(y, angles, x=None, block=None):
    y_gpu = cp.asarray(y, dtype=cp.float32)
    a_gpu = cp.asarray(angles, dtype=cp.float32)

    _, L, H = y.shape
    x_gpu = gpu_array_with_default_shape(x, shape=(L, L, H))

    Np = cp.array(y_gpu.shape[::-1], dtype=cp.int32)
    Nv = cp.array(x_gpu.shape[::-1], dtype=cp.int32)

    tex_obj = create_texture_from_array(y_gpu)
    kernel = get_kernel_cupy("sp_backward_z0.cubin", "sp_backward_z0")

    if block is None:
        block = (8, 8, 8)
    grid = tuple((s + b - 1) // b for s, b in zip(x_gpu.shape[::-1], block))
    kernel(grid, block, (tex_obj, x_gpu, a_gpu, *Np.tolist(), *Nv.tolist()))

    # Return result as cupy array if input was cupy array
    if isinstance(y, cp.ndarray):
        return x_gpu
    else:
        return x_gpu.get()


def backward_z2(y, angles, x=None, block=None):
    y_gpu = cp.asarray(y, dtype=cp.float32)
    a_gpu = cp.asarray(angles, dtype=cp.float32)

    H, _, L = y.shape
    x_gpu = gpu_array_with_default_shape(x, shape=(L, L, H))

    Np = cp.array(y_gpu.shape[::-1], dtype=cp.int32)
    Nv = cp.array(x_gpu.shape[::-1], dtype=cp.int32)

    tex_obj = create_texture_from_array(y_gpu)
    kernel = get_kernel_cupy("sp_backward_z2.cubin", "sp_backward_z2")

    block = (8, 8, 8)
    grid = tuple((s + b - 1) // b for s, b in zip(x_gpu.shape[::-1], block))
    kernel(grid, block, (tex_obj, x_gpu, a_gpu, *Np.tolist(), *Nv.tolist()))

    # Return result as cupy array if input was cupy array
    if isinstance(y, cp.ndarray):
        return x_gpu
    else:
        return x_gpu.get()
