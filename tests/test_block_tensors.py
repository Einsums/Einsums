# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import einsums as ein
import pytest
import numpy as np
import math


@pytest.mark.parametrize(
    "tensor_type", [
        ein.core.BlockTensorF,
        ein.core.BlockTensorD,
        ein.core.BlockTensorC,
        ein.core.BlockTensorZ])
@ein.utils.labeled_section
def test_creation(tensor_type):
    A = tensor_type("A", 2, [2, 1])
    B = tensor_type(2, [2, 1])
    C = tensor_type(2)
    
    assert A.dim() == 3
    assert B.dim() == 3
    assert C.dim() == 0
    
    assert A.num_blocks() == 2
    assert B.num_blocks() == 2
    assert C.num_blocks() == 0
    assert len(A) == 2
    assert len(B) == 2
    assert len(C) == 0
    
    assert A.name == "A"
    
    assert A.rank() == 2
    assert B.rank() == 2
    assert C.rank() == 2
    
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0
    A[2, 2] = 5.0
    
    B[0, 0] = 1.0
    B[0, 1] = 2.0
    B[1, 0] = 3.0
    B[1, 1] = 4.0
    B[2, 2] = 5.0
    
    C.push_block(np.array([[5]]))
    C.insert_block(0, np.array([[1, 2], [3, 4]]))
    
    assert C.dim() == 3
    assert C.num_blocks() == 2
    assert len(C) == 2
    assert C.rank() == 2
    
    for i in range(3):
        for j in range(3):
            assert A[i, j] == B[i, j]
            assert A[i, j] == C[i, j]    

    
@pytest.mark.parametrize(
    "dtype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "etype", [float, complex, np.float32, np.float64, np.complex64, np.complex128]
)
@ein.utils.labeled_section
def test_ops(dtype, etype):
    ein.log_debug("Making A")
    A = ein.utils.create_random_block_tensor("A", 2, [2, 0, 1], dtype=dtype)
    ein.log_debug("Making B")
    B = ein.utils.create_random_block_tensor("B", 2, [2, 0, 1], dtype=dtype)
    ein.log_debug("Making C")
    C = ein.utils.create_random_block_tensor("C", 2, [2, 0, 1], dtype=etype)
    ein.log_debug("Making C_test")
    C_test = ein.utils.create_block_tensor("C", 2, [], dtype=dtype)
    
    ein.log_debug("Adding blocks to C_test")
    for block in C:
        ein.log_debug(f"Adding a block to C_test with dimension {block.dim(0)}.")
        C_test.push_block(block)
    
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
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(2 * A[i, j], rel=rel)
            
    A_res *= B
    
    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(2 * A[i, j] * B[i, j], rel=rel)
    
    A_res *= C

    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(2 * A[i, j] * B[i, j] * C_test[i, j], rel=rel)
            
    # Division
    ein.log_debug("Checking division.")
    A_res = A.copy()
    
    A_res /= 2
    
    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(A[i, j] / 2, rel=rel)
            
    A_res /= B
    
    for i in range(3):
        for j in range(3):
            if B[i, j] != 0 and A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(A[i, j] / 2 / B[i, j], rel=rel)
    
    A_res /= C

    for i in range(3):
        for j in range(3):
            if B[i, j] != 0 and C[i, j] != 0 and A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(A[i, j] / 2 / B[i, j] / C_test[i, j], rel=rel)
    
    # Addition
    ein.log_debug("Checking addition.")
    print(repr(A), flush=True)
    A_res = A.copy()
    print(repr(A_res), flush=True)
    
    A_res += 2
    
    print(repr(A_res), flush=True)
    
    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(2 + A[i, j], rel=rel)
            
    A_res += B
    
    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(2 + A[i, j] + B[i, j], rel=rel)
    
    A_res += C

    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(2 + A[i, j] + B[i, j] + C_test[i, j], rel=rel)
            
    # Subtraction
    ein.log_debug("Checking subtraction.")
    A_res = A.copy()
    
    A_res -= 2
    
    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(A[i, j] - 2, rel=rel)
            
    A_res -= B
    
    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(A[i, j] - 2 - B[i, j], rel=rel)
    
    A_res -= C

    for i in range(3):
        for j in range(3):
            if A.is_inside_block(i, j):
                assert A_res[i, j] == pytest.approx(A[i, j] - 2 - B[i, j] - C_test[i, j], rel=rel)
            
