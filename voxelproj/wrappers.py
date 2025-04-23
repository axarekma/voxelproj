import torch
import numpy as np

from voxelproj._CU import forward_z0, backward_z0
from voxelproj._CU import forward_z2, backward_z2

# from voxelproj._CU import backward_z0_opt


def gpu_array_with_default_shape(x=None, shape=None):
    if x is None:
        x_gpu = torch.zeros(shape, dtype=torch.float32, device="cuda")
    elif isinstance(x, tuple):
        # If x is a shape tuple
        output_shape = x
        x_gpu = torch.zeros(output_shape, dtype=torch.float32, device="cuda")
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        # If x is an array or tensor
        if isinstance(x, np.ndarray):
            x_gpu = torch.tensor(x, dtype=torch.float32, device="cuda")
        else:
            x_gpu = x.to(dtype=torch.float32, device="cuda")
    else:
        raise TypeError("x must be None, a shape tuple, or an array/tensor")
    return x_gpu


def forward(x, angles, y, z_order=0, block_size=[8, 8, 8]):
    # Determine input types for correct output handling
    is_numpy = isinstance(x, np.ndarray)
    is_tensor = isinstance(x, torch.Tensor) and not x.is_cuda
    device = None if not is_tensor else x.device

    return_is_container = isinstance(y, (np.ndarray, torch.Tensor))

    # y_shape_z0 = (n_angles, PX, H)
    # y_shape_z2 = (H, n_angles, PX)
    # x_shape_z0 = (SY, SX, H)
    # x_shape_z2 = (H, SY, SX)
    default_shape = {
        0: (len(angles), max(x.shape[0], x.shape[1]), x.shape[2]),
        2: (x.shape[0], len(angles), max(x.shape[1], x.shape[2])),
    }
    forward_project = {
        0: forward_z0,
        2: forward_z2,
    }
    # Convert inputs to CUDA tensors
    x_gpu = gpu_array_with_default_shape(x)
    a_gpu = gpu_array_with_default_shape(angles)

    # reference might drop here because numpy -> torch transfer
    y_gpu = gpu_array_with_default_shape(y, shape=default_shape[z_order])

    forward_project[z_order](x_gpu, y_gpu, a_gpu, block_size=block_size)

    if return_is_container:
        # Copy result back into user-provided container
        if isinstance(y, np.ndarray):
            np.copyto(y, y_gpu.cpu().numpy())
        elif isinstance(y, torch.Tensor):
            if y.data_ptr() != y_gpu.data_ptr():
                y.to(device)
                y.copy_(y_gpu.to(device))
            else:
                # y is already correct
                pass
        return y
    else:
        # Return a new array in same format as `x`
        if is_numpy:
            return y_gpu.cpu().numpy()
        return y_gpu.to(device)


def backward(y, angles, x, z_order=0, block_size=[8, 8, 8]):
    # Determine input types for correct output handling
    is_numpy = isinstance(y, np.ndarray)
    is_tensor = isinstance(y, torch.Tensor) and not y.is_cuda
    device = None if not is_tensor else y.device

    return_is_container = isinstance(x, (np.ndarray, torch.Tensor))

    # y_shape_z0 = (n_angles, PX, H)
    # y_shape_z2 = (H, n_angles, PX)
    # x_shape_z0 = (SY, SX, H)
    # x_shape_z2 = (H, SY, SX)
    default_shape = {
        0: (y.shape[1], y.shape[1], y.shape[2]),
        2: (y.shape[0], y.shape[2], y.shape[2]),
    }
    backward_project = {
        0: backward_z0,
        2: backward_z2,
    }
    # Convert inputs to CUDA tensors
    y_gpu = gpu_array_with_default_shape(y)
    a_gpu = gpu_array_with_default_shape(angles)

    # reference might drop here because numpy -> torch transfer
    x_gpu = gpu_array_with_default_shape(x, shape=default_shape[z_order])

    # print(" WRAPPER: angles ", a_gpu[0], a_gpu[-1], a_gpu.shape)
    backward_project[z_order](y_gpu, x_gpu, a_gpu, block_size=block_size)

    if return_is_container:
        # Copy result back into user-provided container
        if isinstance(x, np.ndarray):
            np.copyto(x, x_gpu.cpu().numpy())
        elif isinstance(x, torch.Tensor):
            if x.data_ptr() != x_gpu.data_ptr():
                x.to(device)
                x.copy_(x_gpu.to(device))
            else:
                # y is already correct
                pass
        return x
    else:
        # Return a new array in same format as `x`
        if is_numpy:
            return x_gpu.cpu().numpy()
        return x_gpu.to(device)


# def backward_opt(y, angles, x, block=None, z_order=0, block_size=[8, 8, 8]):
#     # Determine input types for correct output handling
#     is_numpy = isinstance(y, np.ndarray)
#     is_tensor = isinstance(y, torch.Tensor) and not y.is_cuda
#     device = None if not is_tensor else y.device

#     return_is_container = isinstance(x, (np.ndarray, torch.Tensor))

#     # y_shape_z0 = (n_angles, PX, H)
#     # y_shape_z2 = (H, n_angles, PX)
#     # x_shape_z0 = (SY, SX, H)
#     # x_shape_z2 = (H, SY, SX)
#     default_shape = {
#         0: (y.shape[1], y.shape[1], y.shape[2]),
#         2: (y.shape[0], y.shape[2], y.shape[2]),
#     }
#     backward_project = {
#         0: backward_z0_opt,
#         # 2: backward_z2,
#     }
#     # Convert inputs to CUDA tensors
#     y_gpu = gpu_array_with_default_shape(y)
#     a_gpu = gpu_array_with_default_shape(angles)

#     # reference might drop here because numpy -> torch transfer
#     x_gpu = gpu_array_with_default_shape(x, shape=default_shape[z_order])

#     # print(" WRAPPER: angles ", a_gpu[0], a_gpu[-1], a_gpu.shape)
#     backward_project[z_order](y_gpu, x_gpu, a_gpu, block_size=block_size)

#     if return_is_container:
#         # Copy result back into user-provided container
#         if isinstance(x, np.ndarray):
#             np.copyto(x, x_gpu.cpu().numpy())
#         elif isinstance(x, torch.Tensor):
#             if x.data_ptr() != x_gpu.data_ptr():
#                 x.to(device)
#                 x.copy_(x_gpu.to(device))
#             else:
#                 # y is already correct
#                 pass
#         return x
#     else:
#         # Return a new array in same format as `x`
#         if is_numpy:
#             return x_gpu.cpu().numpy()
#         return x_gpu.to(device)
