import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize(["a"], [tuple(10), pytest.param(1000, marks = pytest.mark.slow)])

def test_dot(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    plan = ein.compile_plan("", "i", "i")

    assert(type(plan) is ein.EinsumDotPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

@pytest.mark.skipif(not ein.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_dot_copy(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    A_view = ein.GPUView(A, ein.COPY)
    B_view = ein.GPUView(B, ein.COPY)
    C_view = ein.GPUView(C, ein.COPY)

    plan = ein.compile_plan("", "i", "i")

    assert(type(plan) is ein.EinsumDotPlan)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

@pytest.mark.skipif(not ein.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_dot_map(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    A_view = ein.GPUView(A, ein.MAP)
    B_view = ein.GPUView(B, ein.MAP)
    C_view = ein.GPUView(C, ein.MAP)

    plan = ein.compile_plan("", "i", "i")

    assert(type(plan) is ein.EinsumDotPlan)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

