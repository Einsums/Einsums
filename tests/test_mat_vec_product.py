import pytest
import einsums as ein
import numpy as np

pytestmark = pytest.mark.parametrize(["a", "b"], [(10, 10), pytest.param(1000, 1000, marks = pytest.mark.slow), (11, 13)])

def test_mat_vec_prod(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0 for i in range(b)])

    plan = ein.compile_plan("i", "ij", "j")

    assert(type(plan) is ein.EinsumGemvPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b) :
        for j in range(a) :
            C_actual[i] += A[i, j] * B[j]

    for i in range(b) :
        assert(abs(C[i] - C_actual[i]) < 1e-6)

@pytest.mark.skipif(not ein.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_mat_vec_prod_gpu_copy(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0 for i in range(b)])

    plan = ein.compile_plan("i", "ij", "j")

    assert(type(plan) is ein.EinsumGemvPlan)

    A_view = ein.GPUView(A, ein.COPY)
    B_view = ein.GPUView(B, ein.COPY)
    C_view = ein.GPUView(C, ein.COPY)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_view.update_D2H()

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b) :
        for j in range(a) :
            C_actual[i] += A[i, j] * B[j]

    for i in range(b) :
        assert(abs(C[i] - C_actual[i]) < 1e-6)

@pytest.mark.skipif(not ein.gpu_enabled(), reason = "Einsums not built with GPU support!")
def test_mat_vec_prod_gpu_map(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0 for i in range(b)])

    plan = ein.compile_plan("i", "ij", "j")

    assert(type(plan) is ein.EinsumGemvPlan)

    A_view = ein.GPUView(A, ein.MAP)
    B_view = ein.GPUView(B, ein.MAP)
    C_view = ein.GPUView(C, ein.MAP)

    plan.execute(0.0, C_view, 1.0, A_view, B_view)

    C_actual = np.array([0.0 for i in range(b)])

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(b) :
        for j in range(a) :
            C_actual[i] += A[i, j] * B[j]

    for i in range(b) :
        assert(abs(C[i] - C_actual[i]) < 1e-6)
