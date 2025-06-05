# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize("a", [10, pytest.param(1000, marks = pytest.mark.slow)])

def test_dot(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    plan = ein.core.compile_plan("", "i", "i")

    assert(type(plan) is ein.core.EinsumDotPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_dot_copy(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan = ein.core.compile_plan("", "i", "i")

    assert(type(plan) is ein.core.EinsumDotPlan)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_dot_map(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan = ein.core.compile_plan("", "i", "i")

    assert(type(plan) is ein.core.EinsumDotPlan)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

