# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize(["a", "b", "c"], [(10, 10, 10), pytest.param(100, 100, 100, marks = pytest.mark.slow), (11, 13, 17)])

def test_mat_prod(a, b, c) :
    A = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    B = np.array([[np.random.rand() for i in range(c)] for j in range(b)])
    C = np.array([[0.0 for i in range(c)] for j in range(a)])

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)])

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert(type(plan) is ein.core.EinsumGemmPlan)

    plan.execute(0.0, C, 1.0, A, B)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a) :
        for j in range(c) :
            for k in range(b) :
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a) :
        for j in range(c) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_mat_prod_gpu_copy(a, b, c) :
    A = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    B = np.array([[np.random.rand() for i in range(c)] for j in range(b)])
    C = np.array([[0.0 for i in range(c)] for j in range(a)])

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)])

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert(type(plan) is ein.core.EinsumGemmPlan)

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a) :
        for j in range(c) :
            for k in range(b) :
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a) :
        for j in range(c) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_mat_prod_gpu_map(a, b, c) :
    A = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    B = np.array([[np.random.rand() for i in range(c)] for j in range(b)])
    C = np.array([[0.0 for i in range(c)] for j in range(a)])

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)])

    plan = ein.core.compile_plan("ij", "ik", "kj")

    assert(type(plan) is ein.core.EinsumGemmPlan)

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a) :
        for j in range(c) :
            for k in range(b) :
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a) :
        for j in range(c) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)