# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize(["a", "b"], [(10, 10), pytest.param(1000, 1000, marks = pytest.mark.slow), (11, 13)])

def test_generic(a: int, b: int) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    plan = ein.core.compile_plan("ij", "ji", "ij")

    assert(type(plan) is ein.core.EinsumGenericPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = A.T * B

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_generic_gpu_copy(a: int, b: int) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan = ein.core.compile_plan("ij", "ji", "ij")

    assert(type(plan) is ein.core.EinsumGenericPlan)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = A.T * B

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_generic_gpu_map(a: int, b: int) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan = ein.core.compile_plan("ij", "ji", "ij")

    assert(type(plan) is ein.core.EinsumGenericPlan)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = A.T * B

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

