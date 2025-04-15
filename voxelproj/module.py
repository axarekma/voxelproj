import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


def get_kernel(file, kernel):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptx_path = os.path.join(module_path, "cuda", file)
    # print(f"Fetching {kernel} from {file}")
    # Load the module from the PTX file
    module = cuda.module_from_file(ptx_path)
    return module.get_function(kernel)


def forward_z0(x, angles, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if x.shape[2] != y.shape[2]:
        raise ValueError("Arrays must have the same haeight")

    # Convert to float32 if needed
    x = x.astype(np.float32) if x.dtype != np.float32 else x
    angles = angles.astype(np.float32) if angles.dtype != np.float32 else angles
    result = np.zeros_like(y)

    proj_kernel = get_kernel("sp_forward_z0.ptx", "sp_forward_z0")

    Np = np.array(y.shape[::-1]).astype(np.int32)
    Nv = np.array(x.shape[::-1]).astype(np.int32)

    block_size = (8, 8, 8)
    grid_size = tuple((L + bs - 1) // bs for L, bs in zip(y.shape[::-1], block_size))

    proj_kernel(
        cuda.In(x),
        cuda.Out(result),
        cuda.In(angles),
        *Nv,
        *Np,
        block=block_size,
        grid=grid_size,
    )

    return result


def backward_z0(y, angles, x):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if x.shape[2] != y.shape[2]:
        raise ValueError("Arrays must have the same haeight")

    # Convert to float32 if needed
    y = y.astype(np.float32) if y.dtype != np.float32 else y
    angles = angles.astype(np.float32) if angles.dtype != np.float32 else angles
    result = np.zeros_like(x)

    proj_kernel = get_kernel("sp_backward_z0.ptx", "sp_backward_z0")

    Np = np.array(y.shape[::-1]).astype(np.int32)
    Nv = np.array(x.shape[::-1]).astype(np.int32)

    block_size = (8, 8, 8)
    grid_size = tuple((L + bs - 1) // bs for L, bs in zip(x.shape[::-1], block_size))

    proj_kernel(
        cuda.In(y),
        cuda.Out(result),
        cuda.In(angles),
        *Np,
        *Nv,
        block=block_size,
        grid=grid_size,
    )

    return result


def forward_z2(x, angles, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if x.shape[0] != y.shape[0]:
        raise ValueError("Arrays must have the same haeight")

    # Convert to float32 if needed
    x = x.astype(np.float32) if x.dtype != np.float32 else x
    angles = angles.astype(np.float32) if angles.dtype != np.float32 else angles
    result = np.zeros_like(y)

    proj_kernel = get_kernel("sp_forward_z2.ptx", "sp_forward_z2")

    Np = np.array(y.shape[::-1]).astype(np.int32)
    Nv = np.array(x.shape[::-1]).astype(np.int32)

    block_size = (8, 8, 8)
    grid_size = tuple((L + bs - 1) // bs for L, bs in zip(y.shape[::-1], block_size))

    proj_kernel(
        cuda.In(x),
        cuda.Out(result),
        cuda.In(angles),
        *Nv,
        *Np,
        block=block_size,
        grid=grid_size,
    )

    return result


def backward_z2(y, angles, x):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if x.shape[0] != y.shape[0]:
        raise ValueError("Arrays must have the same height")

    # Convert to float32 if needed
    y = y.astype(np.float32) if y.dtype != np.float32 else y
    angles = angles.astype(np.float32) if angles.dtype != np.float32 else angles
    result = np.zeros_like(x)

    proj_kernel = get_kernel("sp_backward_z2.ptx", "sp_backward_z2")

    Np = np.array(y.shape[::-1]).astype(np.int32)
    Nv = np.array(x.shape[::-1]).astype(np.int32)

    block_size = (8, 8, 8)
    grid_size = tuple((L + bs - 1) // bs for L, bs in zip(x.shape[::-1], block_size))

    proj_kernel(
        cuda.In(y),
        cuda.Out(result),
        cuda.In(angles),
        *Np,
        *Nv,
        block=block_size,
        grid=grid_size,
    )

    return result


def create_texture_object_from_numpy(np_array: np.ndarray) -> cuda.TextureObject:
    """
    Create a CUDA texture object from a numpy array.

    Args:
        np_array (np.ndarray): The input numpy array to be used as the texture.

    Returns:
        cuda.TextureObject: The created texture object.
    """
    # Ensure the input is a numpy array of float32 type (for texture operations)
    if np_array.dtype != np.float32:
        np_array = np_array.astype(np.float32)

    # Allocate device memory for the numpy array (texture data)
    tex_data_gpu = cuda.mem_alloc(np_array.nbytes)

    # Copy the data to the device memory
    cuda.memcpy_htod(tex_data_gpu, np_array)

    # Create the texture descriptor
    tex_desc = cuda.ResourceDescriptor()
    tex_desc.set_array(tex_data_gpu)
    tex_desc.set_flags(cuda.TexFlags.READ_ONLY)  # Read-only texture
    tex_desc.set_format(cuda.TexFormat.FLOAT, cuda.ChannelFormatKind.FLOAT)
    tex_desc.set_address(
        tex_data_gpu, np_array.nbytes, np_array.shape[0], np_array.shape[1]
    )

    # Create the texture object using the descriptor
    tex_proj = cuda.TextureObject(tex_desc)

    return tex_proj


def backward_z2_tex(y, angles, x):
    """
    Perform backward projection using texture memory for projection data.

    Args:
        y: NumPy array containing projection data
        angles: NumPy array containing angles
        x: Either a NumPy array to store results or a shape tuple for output

    Returns:
        NumPy array with reconstruction result
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")

    # Convert angles to float32 if needed
    angles = angles.astype(np.float32) if angles.dtype != np.float32 else angles

    # Handle x parameter (can be shape tuple or array)
    if isinstance(x, tuple):
        output_shape = x
        result = np.zeros(output_shape, dtype=np.float32)
    elif isinstance(x, np.ndarray):
        result = x.astype(np.float32) if x.dtype != np.float32 else x
    else:
        raise TypeError("x must be None, a shape tuple, or a numpy array")

    Np = np.array(y.shape[::-1]).astype(np.int32)
    Nv = np.array(result.shape[::-1]).astype(np.int32)

    # Create texture from the projection data
    tex_obj = create_texture_object_from_numpy(y)

    # Get the kernel
    proj_kernel = get_kernel("sp_backward_z2_tex.ptx", "sp_backward_z2_tex")

    # Set block and grid sizes
    block_size = (8, 8, 8)
    grid_size = tuple(
        (L + bs - 1) // bs for L, bs in zip(result.shape[::-1], block_size)
    )

    # Launch kernel
    proj_kernel(
        tex_obj,
        cuda.Out(result),
        cuda.In(angles),
        *Np,
        *Nv,
        block=block_size,
        grid=grid_size,
    )

    return result
