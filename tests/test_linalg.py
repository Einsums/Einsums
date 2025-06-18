# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np
import math

pytestmark = pytest.mark.parametrize(
    ["dtype"], [(np.float64,), (np.complex128,)]
)


@pytest.mark.parametrize(["length"], [(10,), (1000,)])
def test_sumsq(length, dtype):
    lst = ein.utils.create_random_tensor("vector", [length], dtype)

    scale = 0
    sumsq = 0

    sumsq, scale = ein.core.sum_square(lst)

    check = sum(abs(x) ** 2 for x in lst)

    assert check == pytest.approx(scale**2 * sumsq)


@pytest.mark.parametrize(
    ["a", "b", "c"],
    [(10, 10, 10), pytest.param(100, 100, 100, marks=pytest.mark.slow), (11, 13, 17)],
)
def test_gemm(a, b, c, dtype):
    A = np.array([[np.random.rand() for i in range(b)] for j in range(a)], dtype=dtype)
    B = np.array([[np.random.rand() for i in range(c)] for j in range(b)], dtype=dtype)
    C = np.array([[0.0 for i in range(c)] for j in range(a)], dtype=dtype)

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
def test_mat_vec_prod(a, b, dtype):
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)], dtype=dtype)
    B = np.array([np.random.rand() for i in range(a)], dtype=dtype)
    C = np.array([0.0 for i in range(b)], dtype=dtype)

    ein.core.gemv("N", 1.0, A, B, 0.0, C)

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i])


@pytest.mark.parametrize(["width"], [(10,), (100,)])
def test_syev(width, dtype):
    A = ein.utils.create_random_tensor("Test tensor", [width, width], dtype=dtype)

    # Make A symmetric/hermitian.
    for i in range(width):
        A[i, i] = A[i, i].real
        for j in range(i + 1, width):
            A[i, j] = A[j, i].conjugate()

    A_copy = A.copy()

    got_vals = ein.utils.create_tensor(
        "Eigenvalues", [width], dtype=ein.utils.remove_complex(dtype)
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
