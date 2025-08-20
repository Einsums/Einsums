# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np

pytestmark = [
    pytest.mark.parametrize(
        ["a", "b"],
        [(10, 10), pytest.param(1000, 1000, marks=pytest.mark.slow), (11, 13)],
    ),
    pytest.mark.parametrize(
        ["dtype", "rel"], [(np.float64,1e-6), (np.complex128,1e-6)]
    ),
    pytest.mark.parametrize(["array"], [("numpy",), ("einsums",)]),
]


def test_mat_vec_prod(a, b, dtype, rel, array):
    A = ein.utils.random_tensor_factory("A", [b, a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a], dtype, array)
    C = ein.utils.tensor_factory("C", [b], dtype, array)

    plan = ein.core.compile_plan("i", "ij", "j")

    assert type(plan) is ein.core.EinsumGemvPlan

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = np.array([0.0 for i in range(b)], dtype = dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i], rel = rel)

    # Do the swapped version.
    C = ein.utils.tensor_factory("C", [b], dtype, array)

    plan = ein.core.compile_plan("i", "j", "ij")

    assert type(plan) is ein.core.EinsumGemvPlan

    plan.execute(0.0, C, 1.0, B, A)

    C_actual = np.array([0.0 for i in range(b)], dtype = dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i], rel = rel)

def test_mat_vec_prod_list(a, b, dtype, rel, array):
    A = [ein.utils.random_tensor_factory(f"A {i}", [b, a], dtype, array) for i in range(10)]
    B = [ein.utils.random_tensor_factory(f"B {i}", [a], dtype, array) for i in range(10)]
    C = [ein.utils.tensor_factory(f"C {i}", [b], dtype, array) for i in range(10)]

    plan = ein.core.compile_plan("i", "ij", "j")

    assert type(plan) is ein.core.EinsumGemvPlan

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = [np.array([0.0 for i in range(b)], dtype = dtype) for i in range(10)]

    # Numpy hates doing matrix multiplication with einsums imported
    for item in range(10) :
        for i in range(b):
            for j in range(a):
                C_actual[item][i] += A[item][i, j] * B[item][j]

        for i in range(b):
            assert C[item][i] == pytest.approx(C_actual[item][i], rel = rel)

    # Do the swapped version.
    C = [ein.utils.tensor_factory(f"C {i}", [b], dtype, array) for i in range(10)]

    plan = ein.core.compile_plan("i", "j", "ij")

    assert type(plan) is ein.core.EinsumGemvPlan

    plan.execute(0.0, C, 1.0, B, A)

    C_actual = [np.array([0.0 for i in range(b)], dtype = dtype) for i in range(10)]

    # Numpy hates doing matrix multiplication with einsums imported
    for item in range(10) :
        for i in range(b):
            for j in range(a):
                C_actual[item][i] += A[item][i, j] * B[item][j]

        for i in range(b):
            assert C[item][i] == pytest.approx(C_actual[item][i], rel = rel)



@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_mat_vec_prod_gpu_copy(a, b, dtype, rel, array):
    A = ein.utils.random_tensor_factory("A", [b, a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a], dtype, array)
    C = ein.utils.tensor_factory("C", [b], dtype, array)

    plan = ein.core.compile_plan("i", "ij", "j")

    assert type(plan) is ein.core.EinsumGemvPlan

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = np.array([0.0 for i in range(b)], dtype = dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i], rel = rel)

    # Do the swapped version.
    C = ein.utils.tensor_factory("C", [b], dtype, array)

    plan = ein.core.compile_plan("i", "j", "ij")

    assert type(plan) is ein.core.EinsumGemvPlan

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan.execute(0.0, C_view, 1.0, B_view, A_view)

    C_view.update_D2H()

    C_actual = np.array([0.0 for i in range(b)], dtype = dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i], rel = rel)


@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_mat_vec_prod_gpu_map(a, b, dtype, rel, array):
    A = ein.utils.random_tensor_factory("A", [b, a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a], dtype, array)
    C = ein.utils.tensor_factory("C", [b], dtype, array)

    plan = ein.core.compile_plan("i", "ij", "j")

    assert type(plan) is ein.core.EinsumGemvPlan

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = np.array([0.0 for i in range(b)], dtype = dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i], rel = rel)

    # Do the swapped version.
    C = ein.utils.tensor_factory("C", [b], dtype, array)

    plan = ein.core.compile_plan("i", "j", "ij")

    assert type(plan) is ein.core.EinsumGemvPlan

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan.execute(0.0, C_view, 1.0, B_view, A_view)

    C_actual = np.array([0.0 for i in range(b)], dtype = dtype)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b):
        for j in range(a):
            C_actual[i] += A[i, j] * B[j]

    for i in range(b):
        assert C[i] == pytest.approx(C_actual[i], rel = rel)
