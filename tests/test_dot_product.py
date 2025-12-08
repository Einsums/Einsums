# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein
import numpy as np

pytestmark = [
    pytest.mark.parametrize("a", [10, pytest.param(1000, marks=pytest.mark.slow)]),
    pytest.mark.parametrize(
        ["dtype"], [(np.float64,), (np.complex128,)]
    ),
    pytest.mark.parametrize(["array"], [("numpy",), ("einsums",)]),
]


def test_dot(a: int, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a], dtype, array)
    C = ein.utils.tensor_factory("C", [1], dtype, array)

    plan = ein.core.compile_plan("", "i", "i")

    assert type(plan) is ein.core.EinsumDotPlan

    assert C[0] == dtype(0.0)

    plan.execute(0.0, C, 0.25, A, B)

    C_actual = sum(0.25 * a_ * b_ for a_, b_ in zip(A, B))

    assert C[0] == pytest.approx(C_actual)

def test_dot_list(a: int, dtype, array):
    A = [ein.utils.random_tensor_factory(f"A {i}", [a], dtype, array) for i in range(10)]
    B = [ein.utils.random_tensor_factory(f"B {i}", [a], dtype, array) for i in range(10)]
    C = [ein.utils.tensor_factory(f"C {i}", [1], dtype, array) for i in range(10)]

    plan = ein.core.compile_plan("", "i", "i")

    assert type(plan) is ein.core.EinsumDotPlan

    assert C[0] == dtype(0.0)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = [sum(a_ * b_ for a_, b_ in zip(a, b)) for a, b in zip(A, B)]

    for i in range(10) :
        assert C[i][0] == pytest.approx(C_actual[i])

@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_dot_copy(a: int, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a], dtype, array)
    C = ein.utils.tensor_factory("C", [1], dtype, array)

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan = ein.core.compile_plan("", "i", "i")

    assert type(plan) is ein.core.EinsumDotPlan

    plan.execute(0.0, C_view, 0.25, A_view, B_view)

    C_view.update_D2H()

    C_actual = sum(0.25 * a_ * b_ for a_, b_ in zip(A, B))

    assert C[0] == pytest.approx(C_actual)


@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_dot_map(a: int, dtype, array):
    A = ein.utils.random_tensor_factory("A", [a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a], dtype, array)
    C = ein.utils.tensor_factory("C", [1], dtype, array)

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan = ein.core.compile_plan("", "i", "i")

    assert type(plan) is ein.core.EinsumDotPlan

    plan.execute(0.0, C_view, 0.25, A_view, B_view)

    C_actual = sum(0.25 * a_ * b_ for a_, b_ in zip(A, B))

    assert C[0] == pytest.approx(C_actual)
