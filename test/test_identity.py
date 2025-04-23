import numpy as np
import torch
import pytest

from context import voxelproj

# -----------------------------
# Geometry and Shape Definitions
# -----------------------------
SX = 5
SY = 6
H = 3
PX = SX + SY
n_angles = 7

y_shape = {0: (n_angles, PX, H), 2: (H, n_angles, PX)}
x_shape = {0: (SY, SX, H), 2: (H, SY, SX)}
angles = np.linspace(0, np.pi, n_angles).astype(np.float32)


@pytest.mark.parametrize("z_order", [0, 2])
def test_forward_keeps_id_for_cuda_tensor(z_order):
    x = torch.randn(x_shape[z_order], device="cuda")
    y = torch.zeros(y_shape[z_order], device="cuda")
    y_id = id(y)
    y_out = voxelproj.forward(x, angles, y, z_order=z_order)
    assert id(y_out) == y_id


@pytest.mark.parametrize("z_order", [0, 2])
def test_forward_keeps_id_for_cpu_tensor(z_order):
    x = torch.randn(x_shape[z_order])
    y = torch.zeros(y_shape[z_order])
    y_id = id(y)
    y_out = voxelproj.forward(x, angles, y, z_order=z_order)
    assert id(y_out) == y_id


@pytest.mark.parametrize("z_order", [0, 2])
def test_forward_keeps_id_for_numpy_array(z_order):
    x = np.random.randn(*x_shape[z_order]).astype(np.float32)
    y = np.zeros(y_shape[z_order], dtype=np.float32)
    y_id = id(y)
    y_out = voxelproj.forward(x, angles, y, z_order=z_order)
    assert id(y_out) == y_id


@pytest.mark.parametrize("x_type", ["torch_cpu", "torch_cuda", "numpy"])
@pytest.mark.parametrize("z_order", [0, 2])
def test_forward_returns_new_when_y_none(x_type, z_order):
    if x_type == "torch_cpu":
        x = torch.randn(x_shape[z_order])
    elif x_type == "torch_cuda":
        x = torch.randn(x_shape[z_order], device="cuda")
    else:  # numpy
        x = np.random.randn(*x_shape[z_order]).astype(np.float32)

    y_out = voxelproj.forward(x, angles, None, z_order=z_order)

    if isinstance(x, np.ndarray):
        assert isinstance(y_out, np.ndarray)
    else:
        assert isinstance(y_out, torch.Tensor)
        assert y_out.device == x.device


@pytest.mark.parametrize("z_order", [0, 2])
def test_backward_keeps_id_for_cuda_tensor(z_order):
    x = torch.randn(x_shape[z_order], device="cuda")
    y = torch.zeros(y_shape[z_order], device="cuda")
    x_id = id(x)
    x_out = voxelproj.backward(y, angles, x, z_order=z_order)
    assert id(x_out) == x_id


@pytest.mark.parametrize("z_order", [0, 2])
def test_backward_keeps_id_for_cpu_tensor(z_order):
    x = torch.randn(x_shape[z_order])
    y = torch.zeros(y_shape[z_order])
    x_id = id(x)
    x_out = voxelproj.backward(y, angles, x, z_order=z_order)
    assert id(x_out) == x_id


@pytest.mark.parametrize("z_order", [0, 2])
def test_backward_keeps_id_for_numpy_array(z_order):
    x = np.random.randn(*x_shape[z_order]).astype(np.float32)
    y = np.zeros(y_shape[z_order], dtype=np.float32)
    x_id = id(x)
    x_out = voxelproj.backward(y, angles, x, z_order=z_order)
    assert id(x_out) == x_id


@pytest.mark.parametrize("y_type", ["torch_cpu", "torch_cuda", "numpy"])
@pytest.mark.parametrize("z_order", [0, 2])
def test_backward_returns_new_when_x_none(y_type, z_order):
    if y_type == "torch_cpu":
        y = torch.randn(y_shape[z_order])
    elif y_type == "torch_cuda":
        y = torch.randn(y_shape[z_order], device="cuda")
    else:  # numpy
        y = np.random.randn(*y_shape[z_order]).astype(np.float32)

    x_out = voxelproj.backward(y, angles, None, z_order=z_order)

    if isinstance(y, np.ndarray):
        assert isinstance(x_out, np.ndarray)
    else:
        assert isinstance(x_out, torch.Tensor)
        assert x_out.device == y.device
