import einsums_py as ein
import pytest
import numpy as np

def generic_tester(unit) :
    A = np.array([[np.random.rand() for i in range(10)] for j in range(10)])
    B = np.array([[np.random.rand() for i in range(10)] for j in range(10)])
    C = np.array([[0.0 for i in range(10)] for j in range(10)])

    plan = ein.compile_plan("ij", "ji", "ij", unit)

    assert(type(plan) is ein.EinsumGenericPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = A.T * B

    for i in range(10) :
        for j in range(10) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)

def dot_tester(unit) :
    A = np.array([1.0 for i in range(10)])
    B = np.array([np.random.rand() for i in range(10)])
    C = np.array([0.0])

    plan = ein.compile_plan("", "i", "i", unit)

    assert(type(plan) is ein.EinsumDotPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = sum(a * b for a, b in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)

def direct_prod_tester(unit) :
    A = np.array([[np.random.rand() for i in range(10)] for j in range(10)])
    B = np.array([[np.random.rand() for i in range(10)] for j in range(10)])
    C = np.array([[0.0 for i in range(10)] for j in range(10)])

    plan = ein.compile_plan("ij", "ij", "ij", unit)

    assert(type(plan) is ein.EinsumDirectProductPlan)

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = A * B

    for i in range(10) :
        for j in range(10) :
            assert(abs(C[i, j] - C_actual[i, j]) < 1e-6)


def test_generic() :
    generic_tester(ein.CPU)

    if ein.gpu_enabled() :
        generic_tester(ein.GPU_MAP)
        generic_tester(ein.GPU_COPY)

def test_dot() :
    dot_tester(ein.CPU)

    if ein.gpu_enabled() :
        dot_tester(ein.GPU_MAP)
        dot_tester(ein.GPU_COPY)

def test_direct_prod() :
    direct_prod_tester(ein.CPU)

    if ein.gpu_enabled() :
        direct_prod_tester(ein.GPU_MAP)
        direct_prod_tester(ein.GPU_COPY)