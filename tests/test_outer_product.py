import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize(["a", "b"], [(10, 10), pytest.param(1000, 1000, marks = pytest.mark.slow), (11, 13)])

def test_outer_prod(a, b) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(b)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    plan = ein.compile_plan("ij", "i", "j")

    assert(type(plan) is ein.EinsumGerPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = np.outer(A, B)

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

@pytest.mark.skipif(not ein.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_outer_prod_gpu_copy(a, b) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(b)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    plan = ein.compile_plan("ij", "i", "j")

    assert(type(plan) is ein.EinsumGerPlan)

    A_view = ein.GPUView(A, ein.COPY)
    B_view = ein.GPUView(B, ein.COPY)
    C_view = ein.GPUView(C, ein.COPY)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = np.outer(A, B)

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

@pytest.mark.skipif(not ein.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_outer_prod_gpu_map(a, b) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(b)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    plan = ein.compile_plan("ij", "i", "j")

    assert(type(plan) is ein.EinsumGerPlan)

    A_view = ein.GPUView(A, ein.MAP)
    B_view = ein.GPUView(B, ein.MAP)
    C_view = ein.GPUView(C, ein.MAP)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = np.outer(A, B)

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)