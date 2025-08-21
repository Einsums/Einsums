# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np

pytestmark = [
    pytest.mark.parametrize(
        ["a", "b", "c"],
        [
            (10, 10, 10),
            pytest.param(100, 100, 100, marks=pytest.mark.slow),
            (11, 13, 17),
        ],
    ),
    pytest.mark.parametrize(
        ["dtype", "rel"],
        [
            (np.float64, 1e-6),
            (np.complex128, 1e-6),
        ],
    ),
    pytest.mark.parametrize(["array"], [("numpy",), ("einsums",)]),
]


def test_mat_prod(a, b, c, dtype, rel, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)
    B = ein.utils.random_tensor_factory("B", [b, c], dtype, array)
    C = ein.utils.tensor_factory("C", [a, c], dtype, array)

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)], dtype=dtype)

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert type(plan) is ein.core.EinsumGemmPlan

    plan.execute(0.0, C, 1.0, A, B)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a):
        for j in range(c):
            for k in range(b):
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a):
        for j in range(c):
            assert C[i, j] == pytest.approx(C_actual[i, j], rel=rel)

def test_mat_prod_list(a, b, c, dtype, rel, array):
    A = [ein.utils.random_tensor_factory(f"A {i}", [a, b], dtype, array) for i in range(10)]
    B = [ein.utils.random_tensor_factory(f"B {i}", [b, c], dtype, array) for i in range(10)]
    C = [ein.utils.tensor_factory(f"C {i}", [a, c], dtype, array) for i in range(10)]

    C_actual = [np.array([[0.0 for i in range(c)] for j in range(a)], dtype=dtype) for i in range(10)]

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert type(plan) is ein.core.EinsumGemmPlan

    plan.execute(0.0, C, 1.0, A, B)

    # Numpy hates doing matrix multiplication with einsums imported
    for item in range(10) :
        for i in range(a):
            for j in range(c):
                for k in range(b):
                    C_actual[item][i, j] += A[item][i, k] * B[item][k, j]

        for i in range(a):
            for j in range(c):
                assert C[item][i, j] == pytest.approx(C_actual[item][i, j], rel=rel)

@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_mat_prod_gpu_copy(a, b, c, dtype, rel, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)
    B = ein.utils.random_tensor_factory("B", [b, c], dtype, array)
    C = ein.utils.tensor_factory("C", [a, c], dtype, array)

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)], dtype=dtype)

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert type(plan) is ein.core.EinsumGemmPlan

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a):
        for j in range(c):
            for k in range(b):
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a):
        for j in range(c):
            assert C[i, j] == pytest.approx(C_actual[i, j], rel=rel)


@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_mat_prod_gpu_map(a, b, c, dtype, rel, array):
    A = ein.utils.random_tensor_factory("A", [a, b], dtype, array)
    B = ein.utils.random_tensor_factory("B", [b, c], dtype, array)
    C = ein.utils.tensor_factory("C", [a, c], dtype, array)

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)], dtype=dtype)

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert type(plan) is ein.core.EinsumGemmPlan

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    # C_actual = np.matmul(A, B)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a):
        for j in range(c):
            for k in range(b):
                C_actual[i, j] += dtype(dtype(A[i, k]) * dtype(B[k, j]))

    for i in range(a):
        for j in range(c):
            assert C[i, j] == pytest.approx(C_actual[i, j], rel=rel)
