import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize(["a", "b"], [(10, 10), pytest.param(1000, 1000, marks = pytest.mark.slow), (11, 13)])

def test_mat_vec_prod(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0 for i in range(b)])

    plan = ein.core.compile_plan("i", "ij", "j")

    assert(type(plan) is ein.core.EinsumGemvPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b) :
        for j in range(a) :
            C_actual[i] += A[i, j] * B[j]

    for i in range(b) :
        assert(abs(C[i] - C_actual[i]) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_mat_vec_prod_gpu_copy(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0 for i in range(b)])

    plan = ein.core.compile_plan("i", "ij", "j")

    assert(type(plan) is ein.core.EinsumGemvPlan)

    A_view = ein.core.GPUView(A, ein.core.COPY)
    B_view = ein.core.GPUView(B, ein.core.COPY)
    C_view = ein.core.GPUView(C, ein.core.COPY)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b) :
        for j in range(a) :
            C_actual[i] += A[i, j] * B[j]

    for i in range(b) :
        assert(abs(C[i] - C_actual[i]) < 1e-6)

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_mat_vec_prod_gpu_map(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0 for i in range(b)])

    plan = ein.core.compile_plan("i", "ij", "j")

    assert(type(plan) is ein.core.EinsumGemvPlan)

    A_view = ein.core.GPUView(A, ein.core.MAP)
    B_view = ein.core.GPUView(B, ein.core.MAP)
    C_view = ein.core.GPUView(C, ein.core.MAP)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b) :
        for j in range(a) :
            C_actual[i] += A[i, j] * B[j]

    for i in range(b) :
        assert(abs(C[i] - C_actual[i]) < 1e-6)
