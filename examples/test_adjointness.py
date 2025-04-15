import voxelproj
import numpy as np
import time
import cupy

import matplotlib.pyplot as plt

from test_utils import print_adjoint_info

SEED = 138294

SX = 3
SY = 4
H = 3
PX = 8
n_angles = 31
dtype = "float32"
angles = np.linspace(0, np.pi, n_angles, endpoint=False).astype(dtype)

y_shape_z0 = (n_angles, PX, H)
y_shape_z2 = (H, n_angles, PX)

x_shape_z0 = (SY, SX, H)
x_shape_z2 = (H, SY, SX)

pad = 1


def profile_z0(y):
    return np.sum(np.sum(y, -1), -1)


def profile_z2(y):
    return np.sum(np.sum(y, -1), 0)


def phantom_z0(xi, yi):
    x = np.zeros(x_shape_z0).astype(dtype)
    x[yi, xi, :] = 1
    return x


def phantom_z2(xi, yi):
    x = np.zeros(x_shape_z2).astype(dtype)
    x[:, yi, xi] = 1
    return x


def test_adjoint_z0():
    print("\nVoxelproj z0")
    np.random.seed(SEED)
    x = np.random.rand(*x_shape_z0).astype(dtype)
    y = np.random.rand(*y_shape_z0).astype(dtype)
    Ax = voxelproj.forward_z0(x, angles, y=y_shape_z0)
    ATy = voxelproj.backward_z0(y, angles, x=x_shape_z0)
    print_adjoint_info(x, y, Ax, ATy)


def test_adjoint_z2():
    print("\nVoxelproj z2")
    np.random.seed(SEED)
    x = np.random.rand(*x_shape_z2).astype(dtype)
    y = np.random.rand(*y_shape_z2).astype(dtype)
    Ax = voxelproj.forward_z2(x, angles, y=y_shape_z2)
    ATy = voxelproj.backward_z2(y, angles, x=x_shape_z2)
    print_adjoint_info(x, y, Ax, ATy)


def voxelproj_z0():
    for xi in range(SX):
        for yi in range(SY):
            y = voxelproj.forward_z0(phantom_z0(xi, yi), angles, y=y_shape_z0)
            y_sum = profile_z0(y)
            print(f"({xi},{yi}) mean {np.mean(y_sum):.3f} std {np.std(y_sum):.2e}")


def voxelproj_z2():
    for xi in range(SX):
        for yi in range(SY):
            y = voxelproj.forward_z2(phantom_z2(xi, yi), angles, y=y_shape_z2)
            y_sum = profile_z2(y)
            print(f"({xi},{yi}) mean {np.mean(y_sum):.3f} std {np.std(y_sum):.2e}")


print("Vocel_proj 0 ")
voxelproj_z0()
print("Vocel_proj 2 ")
voxelproj_z2()


test_adjoint_z0()
test_adjoint_z2()
