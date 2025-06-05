# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import einsums as ein
import pytest
import numpy as np
import math


@pytest.mark.parametrize(
    "tensor_type",
    [
        ein.core.RuntimeTensorF,
        ein.core.RuntimeTensorD,
        ein.core.RuntimeTensorC,
        ein.core.RuntimeTensorZ,
    ],
)
def test_creation(tensor_type):
    A = tensor_type("A", [3, 3])
    B = tensor_type([3, 3])
    C = tensor_type(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    A_dims = A.dims()
    B_dims = B.dims()
    C_dims = C.dims()

    assert A.rank() == 2
    assert B.rank() == 2
    assert C.rank() == 2

    assert A_dims[0] == 3
    assert A_dims[1] == 3
    assert A.dim(0) == 3
    assert A.dim(1) == 3

    assert B_dims[0] == 3
    assert B_dims[1] == 3
    assert B.dim(0) == 3
    assert B.dim(1) == 3

    assert C_dims[0] == 3
    assert C_dims[1] == 3
    assert C.dim(0) == 3
    assert C.dim(1) == 3

    A_strides = A.strides()
    B_strides = B.strides()
    C_strides = C.strides()

    assert A_strides[0] == 3
    assert A_strides[1] == 1
    assert A.stride(0) == 3
    assert A.stride(1) == 1

    assert B_strides[0] == 3
    assert B_strides[1] == 1
    assert B.stride(0) == 3
    assert B.stride(1) == 1

    assert C_strides[0] == 3
    assert C_strides[1] == 1
    assert C.stride(0) == 3
    assert C.stride(1) == 1

    assert A.get_name() == "A"
    assert A.name == "A"

    B.set_name("B")
    assert B.get_name() == "B"

    B.name = "B2"
    assert B.get_name() == "B2"

    for dtype in [
        int,
        float,
        complex,
        np.single,
        np.double,
        np.complex64,
        np.complex128,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ]:
        C = tensor_type(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype))

        print(C)
        print(dtype)

        for n, x in enumerate(C):
            assert x == n + 1

    for dtype in [
        int,
        float,
        complex,
        np.single,
        np.double,
        np.complex64,
        np.complex128,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ]:
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=dtype
        )
        C = tensor_type(x[0:3, 0:3])

        print(C)
        print(dtype)

        for n, x in enumerate(C):
            assert x == n + (n // 3) + 1

    # Test errors
    for dtype in ["X", "ZX", "T{}"]:
        with pytest.raises((ValueError, TypeError)):
            x = ein.core.BadBuffer(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            x.set_format(dtype)
            C = tensor_type(x)


@pytest.mark.parametrize(
    "tensor_type",
    [
        ein.core.RuntimeTensorF,
        ein.core.RuntimeTensorD,
        ein.core.RuntimeTensorC,
        ein.core.RuntimeTensorZ,
    ],
)
def test_set(tensor_type):
    A = tensor_type("A", [3, 3])
    B = tensor_type("B", [3, 3])

    A.zero()

    B.set_all(3.0)

    for x in A:
        assert x == 0.0

    for x in B:
        assert x == 3.0

    for i in range(3):
        for j in range(3):
            A[i, j] = 3 * i + j

    for i in range(3):
        for j in range(3):
            assert A[i, j] == 3 * i + j

    B[0, 0:2] = np.array([1.0, 2.0])

    B_expected = [1, 2, 3, 3, 3, 3, 3, 3, 3]

    for x, y in zip(B, B_expected):
        assert x == y

    A.assign(B)

    for x, y in zip(A, B):
        assert x == y


@pytest.mark.parametrize(
    "dtype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "etype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
def test_ops(dtype, etype):
    A = ein.utils.create_random_tensor("A", [3, 3], dtype)
    B = ein.utils.create_random_tensor("B", [3, 3], dtype)
    C = ein.utils.create_random_tensor("C", [3, 3], dtype)
    D = np.array(ein.utils.create_random_tensor("D", [3, 3], etype), dtype=etype)
    D_test = np.array(D, dtype=dtype)
    E_base = np.array(ein.utils.create_random_tensor("E", [10, 10], etype), dtype=etype)
    E = E_base[0:3, 0:3]
    E_test = np.array(E, dtype=dtype)
    F = ein.core.BadBuffer(ein.utils.create_random_tensor("F", [3, 3], dtype))
    F.set_format("X")

    # Multiplication
    A_res = A.copy()

    A_res *= 2

    for x, y in zip(A_res, A):
        assert x == pytest.approx(2 * y)

    A_res *= B

    for x, y, z in zip(A_res, A, B):
        assert x == pytest.approx(2 * y * z)

    A_res *= C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C):
        assert x == pytest.approx(2 * y * z * w)

    A_res *= D

    for x, y, z, w, u in zip(A_res, A, B, C, D_test.flat):
        assert x == pytest.approx(2 * y * z * w * u)

    A_res *= E

    for x, y, z, w, u, v in zip(A_res, A, B, C, D_test.flat, E_test.flat):
        assert x == pytest.approx(2 * y * z * w * u * v)

    with pytest.raises(ValueError):
        A_res *= F

    # Addition
    A_res = A.copy()

    A_res += 2

    for x, y in zip(A_res, A):
        assert x == pytest.approx(2 + y)

    A_res += B

    for x, y, z in zip(A_res, A, B):
        assert x == pytest.approx(2 + y + z)

    A_res += C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C):
        assert x == pytest.approx(2 + y + z + w)

    A_res += D

    for x, y, z, w, u in zip(A_res, A, B, C, D_test.flat):
        assert x == pytest.approx(2 + y + z + w + u)

    A_res += E

    for x, y, z, w, u, v in zip(A_res, A, B, C, D_test.flat, E_test.flat):
        assert x == pytest.approx(2 + y + z + w + u + v)

    with pytest.raises(ValueError):
        A_res += F

    # Division
    A_res = A.copy()

    A_res /= 2

    for x, y in zip(A_res, A):
        assert x == pytest.approx(y / 2)

    A_res /= B

    for x, y, z in zip(A_res, A, B):
        assert x == pytest.approx(y / 2 / z)

    A_res /= C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C):
        assert x == pytest.approx(y / 2 / z / w)

    A_res /= D

    for x, y, z, w, u in zip(A_res, A, B, C, D_test.flat):
        assert x == pytest.approx(y / 2 / z / w / u)

    A_res /= E

    for x, y, z, w, u, v in zip(A_res, A, B, C, D_test.flat, E_test.flat):
        assert x == pytest.approx(y / 2 / z / w / u / v)

    with pytest.raises(ValueError):
        A_res /= F

    # Subtraction
    A_res = A.copy()

    A_res -= 2

    for x, y in zip(A_res, A):
        assert x == pytest.approx(y - 2)

    A_res -= B

    for x, y, z in zip(A_res, A, B):
        assert x == pytest.approx(y - 2 - z)

    A_res -= C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C):
        assert x == pytest.approx(y - 2 - z - w)

    A_res -= D

    for x, y, z, w, u in zip(A_res, A, B, C, D_test.flat):
        assert x == pytest.approx(y - 2 - z - w - u)

    A_res -= E

    for x, y, z, w, u, v in zip(A_res, A, B, C, D_test.flat, E_test.flat):
        assert x == pytest.approx(y - 2 - z - w - u - v)

    with pytest.raises(ValueError):
        A_res -= F


def test_view_creation():
    A = ein.core.RuntimeTensorD("A", [5, 5])
    B = ein.core.RuntimeTensorD([5, 5])
    A_view = A[0:3, 0:3]
    B_view = B[0:3, 0:3]

    A_dims = A_view.dims()
    B_dims = B_view.dims()

    assert A_view.rank() == 2
    assert B_view.rank() == 2

    assert A_dims[0] == 3
    assert A_dims[1] == 3
    assert A_view.dim(0) == 3
    assert A_view.dim(1) == 3

    assert B_dims[0] == 3
    assert B_dims[1] == 3
    assert B_view.dim(0) == 3
    assert B_view.dim(1) == 3

    A_strides = A_view.strides()
    B_strides = B_view.strides()

    assert A_strides[0] == 5
    assert A_strides[1] == 1
    assert A_view.stride(0) == 5
    assert A_view.stride(1) == 1

    assert B_strides[0] == 5
    assert B_strides[1] == 1
    assert B_view.stride(0) == 5
    assert B_view.stride(1) == 1

    A_view.set_name("A view")
    assert A_view.get_name() == "A view"
    assert A_view.name == "A view"

    B_view.set_name("B view")
    assert B_view.get_name() == "B view"

    B.name = "B2"
    assert B_view.get_name() == "B view"


def test_view_set():
    A = ein.core.RuntimeTensorD("A", [5, 5])
    B = ein.core.RuntimeTensorD([5, 5])
    A_view = A[0:3, 0:3]
    B_view = B[0:3, 0:3]

    A.zero()

    B.set_all(3.0)

    for x in A:
        assert x == 0.0

    for x in B:
        assert x == 3.0

    for index in enumerate(ein.utils.TensorIndices(A_view)):
        A_view[index[1]] = index[0]

    for index in ein.utils.TensorIndices(A_view):
        assert A_view[index] == 3 * index[0] + index[1]

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A[index] == 3 * index[0] + index[1]
        else:
            assert A[index] == 0

    B_view[0, 0:2] = np.array([1.0, 2.0])

    B_expected = [
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    ]

    for x, y in zip(B, B_expected):
        assert x == y


@pytest.mark.parametrize(
    "dtype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "etype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
def test_view_ops(dtype, etype):
    A = ein.utils.create_random_tensor("A", [5, 5], dtype)
    B = ein.utils.create_random_tensor("B", [3, 3], dtype)
    C = ein.utils.create_random_tensor("C", [5, 5], dtype)
    D = np.array(ein.utils.create_random_tensor("D", [3, 3], etype), dtype=etype)
    D_test = np.array(D, dtype=dtype)
    E_base = np.array(ein.utils.create_random_tensor("E", [10, 10], etype), dtype=etype)
    E = E_base[0:3, 0:3]
    E_test = np.array(E, dtype=dtype)
    F = ein.core.BadBuffer(ein.utils.create_random_tensor("F", [3, 3], dtype))
    F.set_format("X")

    # Multiplication
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view *= 2

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] * 2)
        else:
            assert A_res[index] == A[index]

    A_view *= B

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            try:
                assert A_res[index] == pytest.approx(A[index] * 2 * B[index])
            except Exception as e:
                raise RuntimeError(f"Index is {index}.") from e
        else:
            assert A_res[index] == A[index]

    A_view *= C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] * 2 * B[index] * C[index])
        else:
            assert A_res[index] == A[index]

    A_view *= D

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] * 2 * B[index] * C[index] * D_test[index]
            )
        else:
            assert A_res[index] == A[index]

    A_view *= E

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] * 2 * B[index] * C[index] * D_test[index] * E_test[index]
            )
        else:
            assert A_res[index] == A[index]

    with pytest.raises(ValueError):
        A_view *= F

    # Addition
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view += 2

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] + 2)
        else:
            assert A_res[index] == A[index]

    A_view += B

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] + 2 + B[index])
        else:
            assert A_res[index] == A[index]

    A_view += C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] + 2 + B[index] + C[index])
        else:
            assert A_res[index] == A[index]

    A_view += D

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] + 2 + B[index] + C[index] + D_test[index]
            )
        else:
            assert A_res[index] == A[index]

    A_view += E

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] + 2 + B[index] + C[index] + D_test[index] + E_test[index]
            )
        else:
            assert A_res[index] == A[index]

    with pytest.raises(ValueError):
        A_view += F

    # Subtraction
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view -= 2

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] - 2)
        else:
            assert A_res[index] == A[index]

    A_view -= B

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] - 2 - B[index])
        else:
            assert A_res[index] == A[index]

    A_view -= C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] - 2 - B[index] - C[index])
        else:
            assert A_res[index] == A[index]

    A_view -= D

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] - 2 - B[index] - C[index] - D_test[index]
            )
        else:
            assert A_res[index] == A[index]

    A_view -= E

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] - 2 - B[index] - C[index] - D_test[index] - E_test[index]
            )
        else:
            assert A_res[index] == A[index]

    with pytest.raises(ValueError):
        A_view -= F

    # Division
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view /= 2

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] / 2)
        else:
            assert A_res[index] == A[index]

    A_view /= B

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] / 2 / B[index])
        else:
            assert A_res[index] == A[index]

    A_view /= C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(A[index] / 2 / B[index] / C[index])
        else:
            assert A_res[index] == A[index]

    A_view /= D

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] / 2 / B[index] / C[index] / D_test[index]
            )
        else:
            assert A_res[index] == A[index]

    A_view /= E

    for index in ein.utils.TensorIndices(A):
        if index[0] < 3 and index[1] < 3:
            assert A_res[index] == pytest.approx(
                A[index] / 2 / B[index] / C[index] / D_test[index] / E_test[index]
            )
        else:
            assert A_res[index] == A[index]

    with pytest.raises(ValueError):
        A_view /= F


def test_iterators():
    A = ein.utils.create_random_tensor("A", [10, 10])
    B = A.copy()

    B *= 2

    # Test tensor iterators.
    for x, y in zip(A, B):
        assert 2 * x == y

    # Test tensor view iterators
    A_view = A[0:5, 0:5]
    B_view = B[0:5, 0:5]

    for x, y in zip(A_view, B_view):
        assert 2 * x == y

    # Test reverse tensor iterators.
    for x, y in zip(reversed(A), reversed(B)):
        assert 2 * x == y

    # Test reverse tensor view iterators.
    for x, y in zip(reversed(A_view), reversed(B_view)):
        assert 2 * x == y
