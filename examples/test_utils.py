import numpy as np


def print_adjoint_info(x, y, Ax, ATy):
    x = x.astype("float64")
    y = x.astype("float64")
    Ax = x.astype("float64")
    ATy = x.astype("float64")

    lhs = np.vdot(Ax, y)
    rhs = np.vdot(x, ATy)

    # Calculate relative difference
    avg = (abs(lhs) + abs(rhs)) / 2
    rel_diff = abs(lhs - rhs) / avg

    print(f"    <Ax, y> = {lhs:.4e}")
    print(f"    <x, A*y> = {rhs:.4e}")
    print(f"    Relative difference: {rel_diff:.4e}")


def test_single_voxel(x_shape, y_shape, i0, i1, i2, forward=None, backward=None):
    dtype = "float32"
    EPS = 1e-9
    # Basis vector in volume
    x = np.zeros(x_shape, dtype=dtype)
    x[i0, i1, i2] = 1.0
    Ax = forward(x)  # shape: y_shape

    # Iterate over elements in Ax to compare with At(y_basis)
    max_err = 0
    elements = []
    for i in range(y_shape[0]):
        for j in range(y_shape[1]):
            for k in range(y_shape[2]):
                # Basis vector in projection space
                y = np.zeros(y_shape, dtype=dtype)
                y[i, j, k] = 1.0

                At_y = backward(y)  # shape: x_shape
                lhs = Ax[i, j, k]
                rhs = At_y[i0, i1, i2]
                if lhs > EPS or rhs > EPS:
                    row = [i0, i1, i2, i, j, k, lhs, rhs]
                    elements.append(row)

                    # err = abs(lhs - rhs)
                    # max_err = max(max_err, err)
                    # print(
                    #     f"A[{i0},{i1},{i2}] vs AT[{i},{j},{k}] = {lhs:.3e} vs {rhs:.3e} | err = {err:.2e}"
                    # )
    return elements

    # print(f"\nMax error for voxel ({xi},{yi},{zi}) = {max_err:.2e}")
