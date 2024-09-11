import einsums_py as ein
import pytest
import numpy as np


def test_dot() :
    A = np.array([np.random.rand() for i in range(10)])
    B = np.array([np.random.rand() for i in range(10)])
    C = np.array([0.0])

    plan = ein.compile_plan("", "i", "i")

    plan.execute(0.0, C, 1.0, A, B)

    C_actual = sum(a * b for a, b in zip(A, B))

    assert(abs(C[0] - C_actual) < 1e-6)
