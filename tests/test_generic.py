import pytest
import einsums as ein
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

