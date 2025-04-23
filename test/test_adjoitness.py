import numpy as np
import torch
import pytest

from context import voxelproj

# -----------------------------
# Geometry and Shape Definitions
# -----------------------------
SEED = 128437

SX = 5
SY = 6
H = 3

n_angles = 31
angles = np.linspace(0, np.pi, n_angles).astype(np.float32)


@pytest.mark.parametrize("z_order", [0, 2])
def test_adjointness_full(z_order):
    PX = SX + SY
    y_shape = {0: (n_angles, PX, H), 2: (H, n_angles, PX)}
    x_shape = {0: (SY, SX, H), 2: (H, SY, SX)}

    x = np.random.rand(*x_shape[z_order]).astype(np.float32)
    y = np.random.rand(*y_shape[z_order]).astype(np.float32)
    Ax = voxelproj.forward(x, angles, y=y_shape[z_order], z_order=z_order)
    ATy = voxelproj.backward(y, angles, x=x_shape[z_order], z_order=z_order)

    lhs = np.vdot(Ax, y)
    rhs = np.vdot(x, ATy)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("z_order", [0, 2])
def test_adjointness_interior(z_order):
    PX = min(SY, SX) - 1
    y_shape = {0: (n_angles, PX, H), 2: (H, n_angles, PX)}
    x_shape = {0: (SY, SX, H), 2: (H, SY, SX)}

    x = np.random.rand(*x_shape[z_order]).astype(np.float32)
    y = np.random.rand(*y_shape[z_order]).astype(np.float32)
    Ax = voxelproj.forward(x, angles, y=y_shape[z_order], z_order=z_order)
    ATy = voxelproj.backward(y, angles, x=x_shape[z_order], z_order=z_order)

    lhs = np.vdot(Ax, y)
    rhs = np.vdot(x, ATy)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-4)
