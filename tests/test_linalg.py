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


@pytest.mark.parametrize(["length"], [(10,), (1000,)])
def test_sumsq(length, dtype, array):
    lst = ein.utils.random_tensor_factory("vector", [length], dtype, array)

    scale = 0
    sumsq = 0

    sumsq, scale = ein.core.sum_square(lst)

    check = sum(abs(x) ** 2 for x in lst)

    assert check == pytest.approx(scale**2 * sumsq)


@pytest.mark.parametrize(
    ["a", "b", "c"],
    [(10, 10, 10), pytest.param(100, 100, 100, marks=pytest.mark.slow), (11, 13, 17)],
)
def test_gemm(a, b, c, dtype, array):
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
def test_mat_vec_prod(a, b, dtype, array):

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

    # Make A symmetric/hermitian.
    for i in range(width):
        A[i, i] = A[i, i].real
        for j in range(i + 1, width):
            A[i, j] = A[j, i].conjugate()

    A_copy = np.array(A.copy(), dtype=dtype)

    got_vals = ein.utils.tensor_factory(
        "Eigenvalues", [width], ein.utils.remove_complex(dtype), array
    )

    ein.core.syev("V", A, got_vals)

    expected_vals, expected_vecs = np.linalg.eigh(A_copy)

    eiglist = [
        (expected_vals[i], expected_vecs[:, i]) for i in range(len(expected_vals))
    ]

    eiglist = sorted(eiglist, key=lambda x: x[0])

    expected_vals = [val[0] for val in eiglist]
    expected_vecs = np.array([val[1] for val in eiglist], dtype=dtype)

    # Go through each vector, divide by the first element, then renormalize.
    for i in range(width):
        div = A[i, 0]
        for j in range(width):
            A[i, j] /= div
        norm = np.linalg.norm(list(A[i, :]))
        for j in range(width):
            A[i, j] /= norm

    # The algorithm is very unstable for 32-bit floats.
    if dtype == np.float32 or dtype == np.complex64:
        for exp, res in zip(expected_vals, got_vals):
            assert exp == pytest.approx(res, 1e-3)
        # Don't even check the eigenvectors. They are really bad!
        # for i in range(width):
        #     for j in range(width):
        #         assert (expected_vecs[i, j] == pytest.approx(A[i, j], rel=1e-2)) or (
        #             expected_vecs[i, j] == pytest.approx(-A[i, j], rel=1e-2)
        #        )
    else:
        for exp, res in zip(expected_vals, got_vals):
            assert exp == pytest.approx(res)
        for i in range(width):
            for j in range(width):
                assert (expected_vecs[i, j] == pytest.approx(A[i, j])) or (
                    expected_vecs[i, j] == pytest.approx(-A[i, j])
                )

    # for i in range(width) :
    #     for j in range(width) :
    #         assert(A[i, j] == pytest.approx(expected_vecs[i, j]))


@pytest.mark.parametrize(["width"], [(10,), (50,)])
def test_geev(width, dtype, array):
    A = ein.utils.random_tensor_factory("Test tensor", [width, width], dtype, array)

    A_copy = np.array(A.copy(), dtype=dtype)

    got_vals = ein.utils.tensor_factory(
        "Eigenvalues", [width], ein.utils.add_complex(dtype), array
    )

    A_vecs = ein.utils.tensor_factory(
        "Test eigenvectors", [width, width], ein.utils.add_complex(dtype), array
    )

    ein.core.geev("N", "V", A, got_vals, None, A_vecs)

    expected_vals, expected_vecs = np.linalg.eig(A_copy)

    # eiglist = [
    #     (expected_vals[i], expected_vecs[:, i]) for i in range(len(expected_vals))
    # ]

    # eiglist = sorted(eiglist, key=lambda x: x[0])

    # expected_vals = [val[0] for val in eiglist]
    # expected_vecs = np.array([val[1] for val in eiglist], dtype=dtype)

    # Go through each vector, divide by the first element, then renormalize.
    for i in range(width):
        # div = A_vecs[i, 0]
        # div_ind = 0
        # while div == pytest.approx(0.0) and div_ind < width:
        #     div = A_vecs[i, div_ind]
        #     div_ind += 1

        # for j in range(width):
        #     A_vecs[i, j] /= div
        norm = np.linalg.norm(A_vecs[:, i])
        for j in range(width):
            A_vecs[j, i] /= norm

    print(got_vals)
    print(expected_vals)

    # The algorithm is very unstable for 32-bit floats.
    if dtype == np.float32 or dtype == np.complex64:
        for exp, res in zip(expected_vals, got_vals):
            assert exp == pytest.approx(res, 1e-3)
        # Don't even check the eigenvectors. They are really bad!
        # for i in range(width):
        #     for j in range(width):
        #         assert (expected_vecs[i, j] == pytest.approx(A[i, j], rel=1e-2)) or (
        #             expected_vecs[i, j] == pytest.approx(-A[i, j], rel=1e-2)
        #        )
    else:
        for exp, res in zip(expected_vals, got_vals):
            assert exp == pytest.approx(res)
        # In my experience, the eigenvectors become very unstable above 30x30.
        if width < 30:
            for i in range(width):
                for j in range(width):
                    assert (expected_vecs[i, j] == pytest.approx(A_vecs[i, j])) or (
                        expected_vecs[i, j] == pytest.approx(-A_vecs[i, j])
                    )

    # for i in range(width) :
    #     for j in range(width) :
    #         assert(A[i, j] == pytest.approx(expected_vecs[i, j]))


@pytest.mark.parametrize(["a", "b"], [(10, 1), (10, 10), (100, 100)])
def test_gesv(a, b, dtype, array):
    A = ein.utils.random_definite_tensor_factory("A", a, dtype = dtype, method = array)
    B = ein.utils.random_tensor_factory("B", [a, b], dtype, array)

    A_copy = A.copy()
    B_copy = B.copy()

    ein.core.gesv(A, B)

    expected_B = np.linalg.solve(A_copy, B_copy)

    print(B)
    print(expected_B)

    for i in range(a):
        for j in range(b):
            assert B[i, j] == pytest.approx(expected_B[i, j])


@pytest.mark.parametrize(["a", "b", "c"], [(10, 10, 10), (11, 13, 17)])
def test_scale(a, b, c, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a, b, c], dtype, array)

    A_copy = A.copy()

    scale_factor = ein.utils.random.random()
    print(scale_factor)

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

    assert ein.core.norm(ein.core.Frobenius, A) == pytest.approx(
        np.linalg.norm(A, "fro")
    )
    assert ein.core.norm(ein.core.Infinity, A) == pytest.approx(np.linalg.norm(A, 1))
    assert ein.core.norm(ein.core.One, A) == pytest.approx(
        np.linalg.norm(A, np.inf)
    )

@pytest.mark.parametrize("a", [10, 100])
def test_vec_norm(a, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)

    assert ein.core.vec_norm(A) == pytest.approx(np.linalg.norm(A))

@pytest.mark.parametrize("a", [10, 100])
def test_dot(a, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("A", [a], dtype, array)

    test = sum(x * y for x, y in zip(A, B))

    got = ein.core.dot(A, B)

    assert got == pytest.approx(test)
    
@pytest.mark.parametrize("a", [10, 100])
def test_true_dot(a, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("A", [a], dtype, array)

    test = sum(x.conjugate() * y for x, y in zip(A, B))

    got = ein.core.true_dot(A, B)

    assert got == pytest.approx(test)

@pytest.mark.parametrize(["a", "b"], [(10, 10), (11, 13)])
def test_svd(a, b, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype = dtype)

    U, S, V = ein.core.svd(A)

    U_test, S_test, V_test = np.linalg.svd(A_copy, compute_uv = True)

    for i in range(a) :
        for j in range(a) :
            assert abs(U[i, j]) == pytest.approx(abs(U_test[i, j]))

    for i in range(b) :
        for j in range(b) :
            assert abs(V[i, j]) == pytest.approx(abs(V_test[i, j]))
        
    for i in range(min(a, b)) :
        assert S[i] == pytest.approx(S_test[i])
    
@pytest.mark.parametrize(["a", "b"], [(10, 10), (11, 13)])
def test_nullspace(a, b, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype = dtype)

    Null = ein.core.svd_nullspace(A)

    for i in range(Null.shape[1]) :
        assert ein.core.vec_norm(Null[:, i]) == pytest.approx(1.0)


    Null_expected = sp.linalg.null_space(A_copy, lapack_driver = "gesvd")

    for i in range(Null_expected.shape[1]) :
        scale = 0
        for j in range(b) :
            scale = Null_expected[j, i]
            if abs(scale) > 1e-12 :
                break
        Null_expected[:, i] /= scale
        norm = np.linalg.norm(Null_expected[:, i])
        Null_expected[:, i] /= norm

    assert Null.shape[1] == Null_expected.shape[1]

    for i in range(b) :
        for j in range(Null_expected.shape[1]) :
            assert Null[i, j] == pytest.approx(Null_expected[i, j])

@pytest.mark.parametrize(["a", "b"], [(10, 10), (50, 50), (11, 13)])
def test_sdd(a, b, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype = dtype)

    U, S, V = ein.core.svd_dd(A)

    U_test, S_test, V_test = np.linalg.svd(A_copy, compute_uv = True)

    for i in range(a) :
        for j in range(a) :
            assert abs(U[i, j]) == pytest.approx(abs(U_test[i, j]))

    for i in range(b) :
        for j in range(b) :
            assert abs(V[i, j]) == pytest.approx(abs(V_test[i, j]))
        
    for i in range(min(a, b)) :
        assert S[i] == pytest.approx(S_test[i])

@pytest.mark.parametrize(["a", "b"], [(10, 10), (50, 50), (11, 13), (13, 11)])
def test_qr(a, b, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)

    A_copy = np.array(A.copy(), dtype = dtype)

    R = ein.utils.tensor_factory("R", [min(a, b), b], dtype, array)
    R[:, :] = 0

    QR, tau = ein.core.qr(A)

    for i in range(min(a, b)) :
        for j in range(i, b) :
            R[i, j] = QR[i, j]

    Q = ein.core.q(QR, tau)

    Q_expected, R_expected = np.linalg.qr(A_copy)

    for i in range(Q_expected.shape[0]) :
        for j in range(Q_expected.shape[1]) :
            assert Q[i, j] == pytest.approx(Q_expected[i, j])
    
    for i in range(R_expected.shape[0]) :
        for j in range(R_expected.shape[1]) :
            assert R[i, j] == pytest.approx(R_expected[i, j])

@pytest.mark.parametrize("dims", [[10, 10], [10, 10, 10], [11, 12, 13], [100]])
def test_direct_prod(dims, dtype, array) :
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

    for exp, got in zip(C_copy, C) :
        assert got == pytest.approx(exp, rel = 1e-4)

@pytest.mark.parametrize("a", [10, 25])
def test_det(a, dtype, array) :
    A = ein.utils.random_tensor_factory("A", [a, a], dtype, array)

    A_numpy = A.copy()

    assert ein.core.det(A) == pytest.approx(np.linalg.det(A_numpy))