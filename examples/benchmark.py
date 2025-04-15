import voxelproj
import numpy as np
import time
import cupy

LENGTHS = [128, 256, 512]

print(
    "CUDA ARCH",
    cupy.cuda.runtime.getDeviceProperties(0)["major"],
    cupy.cuda.runtime.getDeviceProperties(0)["minor"],
)


def setup_z0(L):
    np.random.seed(876234)
    n_angles = 192
    dtype = "float32"
    angles = np.linspace(0, np.pi, n_angles, endpoint=False).astype(dtype)
    x = np.random.random((L, L, L)).astype(dtype)
    y = np.random.random((n_angles, L, L)).astype(dtype)
    return x, y, angles


def setup_z2(L):
    n_angles = 192
    dtype = "float32"
    angles = np.linspace(0, np.pi, n_angles, endpoint=False).astype(dtype)
    x = np.random.random((L, L, L)).astype(dtype)
    y = np.random.random((L, n_angles, L)).astype(dtype)
    return x, y, angles


def run_func_A0(func, block=None, pretransfer=False):
    print("\nMeasuring forward_z0")
    for L in LENGTHS:
        x, y, angles = setup_z0(L)
        y *= 0
        if pretransfer:
            x = cp.asarray(x)
            y = cp.asarray(y)
            angles = cp.asarray(angles)
        startTime = time.time()
        y = func(x, angles, y=y, block=block)
        checksum = np.sum(y)
        delta_ms = 1000 * (time.time() - startTime)
        print(f"    Forward Projection (L={L}): {delta_ms:.2f} ms (check {checksum})")


def run_func_A2(func, block=None, pretransfer=False):
    print("\nMeasuring forward_z2")
    for L in LENGTHS:
        x, y, angles = setup_z2(L)
        y *= 0
        if pretransfer:
            x = cp.asarray(x)
            y = cp.asarray(y)
            angles = cp.asarray(angles)
        startTime = time.time()
        y = func(x, angles, y=y, block=block)
        checksum = np.sum(y)
        delta_ms = 1000 * (time.time() - startTime)
        print(f"    Forward Projection (L={L}): {delta_ms:.2f} ms (check {checksum})")


def run_func_AT0(func, block=None, pretransfer=False):
    print("\nMeasuring backward_z0")
    for L in LENGTHS:
        x, y, angles = setup_z0(L)
        x *= 0
        if pretransfer:
            x = cp.asarray(x)
            y = cp.asarray(y)
            angles = cp.asarray(angles)
        startTime = time.time()
        x_res = func(y, angles, x=x, block=block)
        checksum = np.sum(x_res)
        delta_ms = 1000 * (time.time() - startTime)
        print(f"    Backward Projection (L={L}): {delta_ms:.2f} ms (check {checksum})")


def run_func_AT2(func, block=None, pretransfer=False):
    print("\nMeasuring backward_z2")
    for L in LENGTHS:
        x, y, angles = setup_z2(L)
        x *= 0
        if pretransfer:
            x = cp.asarray(x)
            y = cp.asarray(y)
            angles = cp.asarray(angles)
        startTime = time.time()
        x_res = func(y, angles, x=x, block=block)
        checksum = np.sum(x_res)
        delta_ms = 1000 * (time.time() - startTime)
        print(f"    Backward Projection (L={L}): {delta_ms:.2f} ms (check {checksum})")


import cupy as cp


def get_max_block_dimensions():
    # Get the current device properties
    device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.device.get_device_id())

    # Extract the maximum block dimensions
    max_block_dim_x = device_props["maxThreadsDim"][0]
    max_block_dim_y = device_props["maxThreadsDim"][1]
    max_block_dim_z = device_props["maxThreadsDim"][2]

    # Get the maximum total threads per block
    max_threads_per_block = device_props["maxThreadsPerBlock"]

    return {
        "max_dim_x": max_block_dim_x,
        "max_dim_y": max_block_dim_y,
        "max_dim_z": max_block_dim_z,
        "max_threads_per_block": max_threads_per_block,
    }


# Get and print the maximum block dimensions
max_dims = get_max_block_dimensions()
print(
    f"Maximum block dimensions: ({max_dims['max_dim_x']}, {max_dims['max_dim_y']}, {max_dims['max_dim_z']})"
)
print(f"Maximum threads per block: {max_dims['max_threads_per_block']}")

if __name__ == "__main__":
    run_func_A0(voxelproj.forward_z0, pretransfer=True)
    run_func_A2(voxelproj.forward_z2, pretransfer=True)
    run_func_AT0(voxelproj.backward_z0, pretransfer=True)
    run_func_AT2(voxelproj.backward_z2, pretransfer=True)

    print("DONE")
