# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import einsums as ein
import pytest
import numpy as np
import math


@pytest.mark.parametrize(
    "tensor_type", [
        ein.core.TiledTensorF,
        ein.core.TiledTensorD,
        ein.core.TiledTensorC,
        ein.core.TiledTensorZ])
@ein.utils.labeled_section
def test_creation(tensor_type):
    A = tensor_type("A", [[2, 1], [2, 1]])
    B = tensor_type("B", 2, [2, 1])
    
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    
    assert A.shape[0] == 3
    assert A.shape[1] == 3
    
    assert B.shape[0] == 3
    assert B.shape[1] == 3
    
    assert A.num_filled() == 0
    assert B.num_filled() == 0
    
    assert len(A) == 0
    assert len(B) == 0
    
    assert A.name == "A"
    
    assert A.rank() == 2
    assert B.rank() == 2
    
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0
    A[2, 2] = 5.0
    
    assert A.num_filled() == 2
    assert A.shape[0] == 3
    assert A.shape[1] == 3
    assert len(A) == 2
    
    B[0, 0] = 1.0
    B[0, 1] = 2.0
    B[1, 0] = 3.0
    B[1, 1] = 4.0
    B[2, 2] = 5.0
    
    assert B.num_filled() == 2
    assert B.shape[0] == 3
    assert B.shape[1] == 3
    assert len(B) == 2
    
    for i in range(3):
        for j in range(3):
            assert A[i, j] == B[i, j]   

    
@pytest.mark.parametrize(
    "dtype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "etype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
@ein.utils.labeled_section
def test_ops(dtype, etype):
    ein.log_debug("Making A")
    A = ein.utils.create_random_tiled_tensor("A", [[2, 0, 1], [2, 0, 1]], dtype=dtype)
    ein.log_debug("Making B")
    B = ein.utils.create_random_tiled_tensor("B", [[2, 0, 1], [2, 0, 1]], dtype=dtype)
    
    ein.log_debug("Done")
    
    rel = 1e-6
    if dtype in ein.utils.__singles or dtype in ein.utils.__complex_singles or etype in ein.utils.__singles or etype in ein.utils.__complex_singles:
        rel = 1e-3
        
    # Multiplication
    ein.log_debug("Checking multiplication.")
        
    A_res = A.copy()
    
    A_res *= 2
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(2 * A[i, j], rel=rel)
            
    A_res *= B
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(2 * A[i, j] * B[i, j], rel=rel)
    
    # Division
    ein.log_debug("Checking division.")
    A_res = A.copy()
    
    A_res /= 2
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(A[i, j] / 2, rel=rel)
            
    A_res /= B
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(A[i, j] / 2 / B[i, j], rel=rel)
    
    # Addition
    ein.log_debug("Checking addition.")
    print(repr(A), flush=True)
    A_res = A.copy()
    print(repr(A_res), flush=True)
    
    A_res += 2
    
    print(repr(A_res), flush=True)
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(2 + A[i, j], rel=rel)
            
    A_res += B
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(2 + A[i, j] + B[i, j], rel=rel)
            
    # Subtraction
    ein.log_debug("Checking subtraction.")
    A_res = A.copy()
    
    A_res -= 2
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(A[i, j] - 2, rel=rel)
            
    A_res -= B
    
    for i in range(3):
        for j in range(3):
            assert A_res[i, j] == pytest.approx(A[i, j] - 2 - B[i, j], rel=rel)
            
