import pytest
import einsums_py as ein
import numpy as np

def generic_tester(a: int, b: int) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    C = np.array([[0.0 for i in range(b)] for j in range(a)])

    plan = ein.compile_plan("ij", "ji", "ij")

    assert(type(plan) is ein.EinsumGenericPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = A.T * B

    for i in range(a) :
        for j in range(b) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

def dot_tester(a: int) :
    A = np.array([np.random.rand() for i in range(a)])
    B = np.array([np.random.rand() for i in range(a)])
    C = np.array([0.0])

    plan = ein.compile_plan("", "i", "i")

    assert(type(plan) is ein.EinsumDotPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = sum(a_ * b_ for a_, b_ in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

def direct_prod_tester(a, b) :
    A = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    B = np.array([[np.random.rand() for i in range(a)] for j in range(b)])
    C = np.array([[0.0 for i in range(a)] for j in range(b)])

    plan = ein.compile_plan("ij", "ij", "ij")

    assert(type(plan) is ein.EinsumDirectProductPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = A * B

    for i in range(b) :
        for j in range(a) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

def outer_prod_tester(a, b) :
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

def mat_vec_prod_tester(a, b) :
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

def mat_prod_tester(a, b, c) :
    A = np.array([[np.random.rand() for i in range(b)] for j in range(a)])
    B = np.array([[np.random.rand() for i in range(c)] for j in range(b)])
    C = np.array([[0.0 for i in range(c)] for j in range(a)])

    C_actual = np.array([[0.0 for i in range(c)] for j in range(a)])

    plan = ein.compile_plan("ij", "ik", "kj")

    assert(type(plan) is ein.EinsumGemmPlan)

    plan.execute(0.0, C, 1.0, A, B)

    # Numpy hates doing matrix multiplication with einsums imported
    for i in range(a) :
        for j in range(c) :
            for k in range(b) :
                C_actual[i, j] += A[i, k] * B[k, j]

    for i in range(a) :
        for j in range(c) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)


def test_generic() :
    generic_tester(10, 10)

def test_dot() :
    dot_tester(10)

def test_direct_prod() :
    direct_prod_tester(10, 10)

def test_outer_prod() :
    outer_prod_tester(10, 10)

def test_mat_vec_prod() :
    mat_vec_prod_tester(10, 10)

def test_mat_prod() :
    mat_prod_tester(10, 10, 10)

@pytest.mark.slow
def test_generic_large() :
    generic_tester(1000, 1000)

@pytest.mark.slow
def test_dot_large() :
    dot_tester(100000)

@pytest.mark.slow
def test_direct_prod_large() :
    direct_prod_tester(1000, 1000)

@pytest.mark.slow
def test_outer_prod_large() :
    outer_prod_tester(1000, 1000)

@pytest.mark.slow
def test_mat_vec_prod_large() :
    mat_vec_prod_tester(1000, 1000)

@pytest.mark.slow
def test_mat_prod_large() :
    mat_prod_tester(100, 100, 100)

def test_generic_different() :
    generic_tester(11, 13)

def test_direct_prod_different() :
    direct_prod_tester(11, 13)

def test_outer_prod_different() :
    outer_prod_tester(11, 13)

def test_mat_vec_prod_different() :
    mat_vec_prod_tester(11, 13)

def test_mat_prod_different() :
    mat_prod_tester(11, 13, 17)