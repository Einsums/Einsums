# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np
import scipy as sp
import math

pytestmark = [
    pytest.mark.parametrize(["dtype"], [(np.float64,), (np.complex128,)]),
    pytest.mark.parametrize("array", ["einsums", "numpy"]),
]


@pytest.fixture
def set_big_memory():
    ein.core.GlobalConfigMap.get_singleton().set_str("buffer-size", "1GB")


@pytest.mark.parametrize(["length"], [(10,), (1000,)])
def test_sumsq(length, dtype, array):
    lst = ein.utils.random_tensor_factory("vector", [length], dtype, array)

    sumsq = ein.core.sum_square(lst)

    check = sum(abs(x) ** 2 for x in lst)

    assert check == pytest.approx(sumsq)


@pytest.mark.parametrize(
    ["a", "b", "c"],
    [(10, 10, 10), pytest.param(100, 100, 100, marks=pytest.mark.slow), (11, 13, 17)],
)
def test_gemm(set_big_memory, a, b, c, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)
    B = ein.utils.random_tensor_factory("B", [b, c], dtype, array)
    C = ein.utils.tensor_factory("C", [a, c], dtype, array)

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)], dtype=dtype)

    ein.core.gemm("N", "N", 1.0, A, B, 0.0, C)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a):
        for j in range(c):
            for k in range(b):
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a):
        for j in range(c):
            assert C[i, j] == pytest.approx(C_actual[i, j])


@pytest.mark.parametrize(
    ["a", "b"], [(10, 10), pytest.param(1000, 1000, marks=pytest.mark.slow), (11, 13)]
)
def test_mat_vec_prod(set_big_memory, a, b, dtype, array):

    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)
    B = ein.utils.random_tensor_factory("B", [b], dtype, array)
    C = ein.utils.tensor_factory("C", [a], dtype, array)

    ein.core.gemv("N", 1.0, A, B, 0.0, C)

    C_actual = np.array([0.0 for i in range(b)], dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a):
        for j in range(b):
            C_actual[i] += A[i, j] * B[j]

    for i in range(a):
        assert C[i] == pytest.approx(C_actual[i])


@pytest.mark.parametrize(["width"], [(10,), (100,)])
def test_syev(width, dtype, array):
    A = ein.utils.random_tensor_factory("Test tensor", [width, width], dtype, array)
    W = ein.utils.tensor_factory("Eigenvalues", [width], dtype = ein.utils.remove_complex(dtype))

    if ein.utils.is_complex(dtype) :
        for i in range(width) :
            A[i, i] = A[i, i].real
            for j in range(i) :
                A[i, j] = A[j, i].conjugate()
    else :
        for i in range(width) :
            for j in range(i) :
                A[i, j] = A[j, i]


    A_copy = np.array(A)

    ein.core.syev('V', A, W)

    evals, evecs = np.linalg.eigh(A_copy)

    pair = [(evals[i], evecs[:, i]) for i in range(width)]

    pair = sorted(pair, key = lambda x : x[0])

    evals, evecs = [p[0] for p in pair], np.array([p[1] for p in pair]).T

    for i in range(width) :
        assert W[i] == pytest.approx(evals[i])

    for i in range(width) :
        scale = evecs[0, i] / A[0, i]
        for j in range(width) :
            assert A[j, i] * scale == pytest.approx(evecs[j, i])





@pytest.mark.parametrize(["width"], [(10,), (50,)])
def test_geev(width, dtype, array):
    A = ein.utils.random_tensor_factory("A", [width, width], dtype=dtype, method=array)
    A_copy = A.copy()

    got_evals = ein.utils.tensor_factory(
        "Got eigenvalues", [width], dtype=ein.utils.add_complex(dtype), method=array
    )
    got_evecs = ein.utils.tensor_factory(
        "Got eigenvectors",
        [width, width],
        dtype=ein.utils.add_complex(dtype),
        method=array,
    )

    evals, evecs = np.linalg.eig(A_copy)
    ein.core.geev(A, got_evals, None, got_evecs)

    for i in range(width):
        min_pos = i
        for j in range(i, width):
            if got_evals[j].real < got_evals[min_pos].real:
                min_pos = j
            elif (
                got_evals[j].real == got_evals[min_pos].real
                and got_evals[j].imag < got_evals[min_pos].imag
            ):
                min_pos = j
        if min_pos != i:
            got_evals[i], got_evals[min_pos] = got_evals[min_pos], got_evals[i]
            for j in range(width):
                temp = got_evecs[j, i]
                got_evecs[j, i] = got_evecs[j, min_pos]
                got_evecs[j, min_pos] = temp

    for i in range(width):
        min_pos = i
        for j in range(i, width):
            if evals[j].real < evals[min_pos].real:
                min_pos = j
            elif (
                evals[j].real == evals[min_pos].real
                and evals[j].imag < evals[min_pos].imag
            ):
                min_pos = j
        if min_pos != i:
            evals[i], evals[min_pos] = evals[min_pos], evals[i]
            for j in range(width):
                temp = evecs[j, i]
                evecs[j, i] = evecs[j, min_pos]
                evecs[j, min_pos] = temp

    for i in range(width):
        assert got_evals[i] == pytest.approx(evals[i])

    for i in range(width):
        scale = evecs[0, i] / got_evecs[0, i]
        for j in range(width):
            assert got_evecs[j, i] * scale == pytest.approx(evecs[j, i])


@pytest.mark.parametrize(["a", "b"], [(10, 1), (10, 10), (100, 100)])
def test_gesv(a, b, dtype, array):
    A = ein.utils.random_definite_tensor_factory("A", a, dtype=dtype, method=array)
    B = ein.utils.random_tensor_factory("B", [a, b], dtype, array)

    print(A.strides)

    A_copy = A.copy()
    B_copy = B.copy()

    ein.core.gesv(A, B)

    expected_B = np.linalg.solve(A_copy, B_copy)

    for i in range(a):
        for j in range(b):
            assert B[i, j] == pytest.approx(expected_B[i, j])


@pytest.mark.parametrize(["a", "b", "c"], [(10, 10, 10), (11, 13, 17)])
def test_scale(a, b, c, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b, c], dtype, array)

    A_copy = A.copy()

    scale_factor = ein.utils.random.random()

    ein.core.scale(scale_factor, A)

    for i in range(a):
        for j in range(b):
            for k in range(c):
                assert scale_factor * A_copy[i, j, k] == pytest.approx(A[i, j, k])


@pytest.mark.parametrize(["a", "b"], [(10, 1), (1, 10), (10, 10), (100, 100)])
def test_scale_row(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = A.copy()

    scale = ein.utils.random.random()
    row = ein.utils.random.randint(0, a - 1)

    ein.core.scale_row(row, scale, A)

    for i in range(a):
        for j in range(b):
            if i == row:
                assert A_copy[i, j] * scale == pytest.approx(A[i, j])
            else:
                assert A_copy[i, j] == A[i, j]


@pytest.mark.parametrize(["a", "b"], [(10, 1), (1, 10), (10, 10), (100, 100)])
def test_scale_col(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = A.copy()

    scale = ein.utils.random.random()
    col = ein.utils.random.randint(0, b - 1)

    ein.core.scale_column(col, scale, A)

    for i in range(a):
        for j in range(b):
            if j == col:
                assert A_copy[i, j] * scale == pytest.approx(A[i, j])
            else:
                assert A_copy[i, j] == A[i, j]


@pytest.mark.parametrize("a", [10, 100])
def test_axpy(a, dtype, array):
    X = ein.utils.random_tensor_factory("X", [a], dtype, array)
    Y = ein.utils.random_tensor_factory("Y", [a], dtype, array)

    alpha = ein.utils.random.random()

    Y_test = np.zeros([a], dtype=dtype)

    for i in range(a):
        Y_test[i] = alpha * X[i] + Y[i]

    ein.core.axpy(alpha, X, Y)

    for i in range(a):
        assert Y_test[i] == pytest.approx(Y[i])


@pytest.mark.parametrize("a", [10, 100])
def test_axpby(a, dtype, array):
    X = ein.utils.random_tensor_factory("X", [a], dtype, array)
    Y = ein.utils.random_tensor_factory("Y", [a], dtype, array)

    alpha = ein.utils.random.random()
    beta = ein.utils.random.random()

    Y_test = np.zeros([a], dtype=dtype)

    for i in range(a):
        Y_test[i] = alpha * X[i] + beta * Y[i]

    ein.core.axpby(alpha, X, beta, Y)

    for i in range(a):
        assert Y_test[i] == pytest.approx(Y[i])


@pytest.mark.parametrize(["a", "b"], [(10, 10), (100, 100), (11, 13)])
def test_ger(a, b, dtype, array):
    X = ein.utils.random_tensor_factory("X", [a], dtype, array)
    Y = ein.utils.random_tensor_factory("Y", [b], dtype, array)
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    alpha = ein.utils.random.random()

    A_test = np.zeros([a, b], dtype=dtype)

    for i in range(a):
        for j in range(b):
            A_test[i, j] = A[i, j] + alpha * X[i] * Y[j]

    ein.core.ger(alpha, X, Y, A)

    for i in range(a):
        for j in range(b):
            assert A[i, j] == pytest.approx(A_test[i, j])


@pytest.mark.parametrize("a", [10, 100])
def test_invert(a, dtype, array):
    A = ein.utils.random_definite_tensor_factory("A", a, dtype=dtype, method=array)

    A_copy = A.copy()

    ein.core.invert(A)

    test = ein.utils.tensor_factory("test", [a, a], dtype, array)

    ein.core.gemm("n", "n", 1.0, A_copy, A, 0.0, test)

    for i in range(a):
        assert test[i, i] == pytest.approx(1.0)
        for j in range(i):
            assert test[i, j] == pytest.approx(0.0)
            assert test[j, i] == pytest.approx(0.0)


@pytest.mark.parametrize(["a", "b"], [(10, 10), (100, 100), (11, 13)])
def test_norm(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    assert ein.core.norm(ein.core.FROBENIUS, A) == pytest.approx(
        np.linalg.norm(A, "fro")
    )
    assert ein.core.norm(ein.core.ONE, A) == pytest.approx(np.linalg.norm(A, 1))
    assert ein.core.norm(ein.core.INFINITY, A) == pytest.approx(np.linalg.norm(A, np.inf))


@pytest.mark.parametrize("a", [10, 100])
def test_vec_norm(a, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)

    assert ein.core.vec_norm(A) == pytest.approx(np.linalg.norm(A))


@pytest.mark.parametrize("a", [10, 100])
def test_dot(a, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("A", [a], dtype, array)

    test = sum(x * y for x, y in zip(A, B))

    got = ein.core.dot(A, B)

    assert got == pytest.approx(test)


@pytest.mark.parametrize("a", [10, 100])
def test_true_dot(a, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("A", [a], dtype, array)

    test = sum(x.conjugate() * y for x, y in zip(A, B))

    got = ein.core.true_dot(A, B)

    assert got == pytest.approx(test)


@pytest.mark.parametrize(["a", "b"], [(10, 10), (11, 13)])
def test_svd(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype=dtype)

    U, S, V = ein.core.svd(A)

    U_test, S_test, V_test = np.linalg.svd(A_copy, compute_uv=True)

    for i in range(a):
        for j in range(a):
            assert abs(U[i, j]) == pytest.approx(abs(U_test[i, j]))

    for i in range(b):
        for j in range(b):
            assert abs(V[i, j]) == pytest.approx(abs(V_test[i, j]))

    for i in range(min(a, b)):
        assert S[i] == pytest.approx(S_test[i])


@pytest.mark.parametrize(["a", "b"], [(10, 10), (11, 13)])
def test_nullspace(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype=dtype)

    Null = ein.core.svd_nullspace(A)

    for i in range(Null.shape[1]):
        assert ein.core.vec_norm(Null[:, i]) == pytest.approx(1.0)

    Null_expected = sp.linalg.null_space(A_copy, lapack_driver="gesvd")

    for i in range(Null_expected.shape[1]):
        scale = 0
        for j in range(b):
            scale = Null_expected[j, i]
            if abs(scale) > 1e-12:
                break
        Null_expected[:, i] /= scale
        norm = np.linalg.norm(Null_expected[:, i])
        Null_expected[:, i] /= norm

    assert Null.shape[1] == Null_expected.shape[1]

    for j in range(Null_expected.shape[1]):
        scale = Null_expected[0, j] / Null[0, j]
        for i in range(b):
            assert Null[i, j] * scale == pytest.approx(Null_expected[i, j])


@pytest.mark.parametrize(["a", "b"], [(10, 10), (50, 50), (11, 13)])
def test_sdd(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype=dtype)

    U, S, V = ein.core.svd_dd(A)

    U_test, S_test, V_test = np.linalg.svd(A_copy, compute_uv=True)

    for i in range(a):
        for j in range(a):
            assert abs(U[i, j]) == pytest.approx(abs(U_test[i, j]))

    for i in range(b):
        for j in range(b):
            assert abs(V[i, j]) == pytest.approx(abs(V_test[i, j]))

    for i in range(min(a, b)):
        assert S[i] == pytest.approx(S_test[i])


@pytest.mark.parametrize(["a", "b"], [(10, 10), (50, 50), (11, 13), (13, 11)])
def test_qr(a, b, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    Q, R = ein.core.qr(A)

    A_test = ein.utils.tensor_factory("A test", [a, b], dtype, array)

    ein.core.gemm('n', 'n', 1.0, Q, R, 0.0, A_test)

    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            assert A_test[i, j] == pytest.approx(A[i, j])


@pytest.mark.parametrize("dims", [[10, 10], [10, 10, 10], [11, 12, 13], [100]])
def test_direct_prod(dims, dtype, array):
    A = ein.utils.random_tensor_factory("A", dims, dtype, array)
    B = ein.utils.random_tensor_factory("B", dims, dtype, array)
    C = ein.utils.random_tensor_factory("C", dims, dtype, array)

    alpha = ein.utils.random.random()
    beta = ein.utils.random.random()

    C_copy = C.copy()

    C_copy *= beta
    temp = A.copy()
    temp *= B
    temp *= alpha
    C_copy += temp

    ein.core.direct_product(alpha, A, B, beta, C)

    for exp, got in zip(C_copy, C):
        assert got == pytest.approx(exp, rel=1e-4)


@pytest.mark.parametrize("a", [10, 25])
def test_det(a, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, a], dtype, array)

    A_numpy = A.copy()

    assert ein.core.det(A) == pytest.approx(np.linalg.det(A_numpy))
