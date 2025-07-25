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
    pytest.mark.parametrize(["dtype"], [(np.float64,), (np.complex128,)]),
    pytest.mark.parametrize(["array"], [("numpy",), ("einsums",)]),
]


@pytest.fixture
def set_big_memory():
    ein.core.GlobalConfigMap.get_singleton().set_str("buffer-size", "1GB")
    ein.core.GlobalConfigMap.get_singleton().set_str("gpu-buffer-size", "1GB")


def test_generic(set_big_memory, a: int, b: int, dtype, array):
    A = ein.utils.random_tensor_factory("A", [b, a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a, b], dtype, array)
    C = ein.utils.tensor_factory("C", [a, b], dtype, array)

    plan = ein.core.compile_plan("ij", "ji", "ij")

    assert type(plan) is ein.core.EinsumGenericPlan

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = np.array(A).T * B

    for i in range(a):
        for j in range(b):
            assert C[i, j] == pytest.approx(C_actual[i, j])


@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_generic_gpu_copy(set_big_memory, a: int, b: int, dtype, array):
    A = ein.utils.random_tensor_factory("A", [b, a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a, b], dtype, array)
    C = ein.utils.tensor_factory("C", [a, b], dtype, array)

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan = ein.core.compile_plan("ij", "ji", "ij")

    assert type(plan) is ein.core.EinsumGenericPlan

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = A.T * B

    for i in range(a):
        for j in range(b):
            assert C[i, j] == pytest.approx(C_actual[i, j])


@pytest.mark.skipif(
    not ein.core.gpu_enabled(), reason="Einsums not built with GPU support!"
)
def test_generic_gpu_map(set_big_memory, a: int, b: int, dtype, array):
    A = ein.utils.random_tensor_factory("A", [b, a], dtype, array)
    B = ein.utils.random_tensor_factory("B", [a, b], dtype, array)
    C = ein.utils.tensor_factory("C", [a, b], dtype, array)

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan = ein.core.compile_plan("ij", "ji", "ij")

    assert type(plan) is ein.core.EinsumGenericPlan

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = A.T * B

    for i in range(a):
        for j in range(b):
            assert C[i, j] == pytest.approx(C_actual[i, j])
