import einsums as ein
import pytest
import numpy as np

def test_creation() :
    A = ein.core.RuntimeTensor("A", [3, 3])
    B = ein.core.RuntimeTensor([3, 3])
    C = ein.core.RuntimeTensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))

    A_dims = A.dims()
    B_dims = B.dims()
    C_dims = C.dims()

    assert A.rank() == 2
    assert B.rank() == 2
    assert C.rank() == 2

    assert A_dims[0] == 3
    assert A_dims[1] == 3
    assert A.dim(0) == 3
    assert A.dim(1) == 3

    assert B_dims[0] == 3
    assert B_dims[1] == 3
    assert B.dim(0) == 3
    assert B.dim(1) == 3

    assert C_dims[0] == 3
    assert C_dims[1] == 3
    assert C.dim(0) == 3
    assert C.dim(1) == 3

    A_strides = A.strides()
    B_strides = B.strides()
    C_strides = C.strides()

    assert A_strides[0] == 3
    assert A_strides[1] == 1
    assert A.stride(0) == 3
    assert A.stride(1) == 1

    assert B_strides[0] == 3
    assert B_strides[1] == 1
    assert B.stride(0) == 3
    assert B.stride(1) == 1

    assert C_strides[0] == 3
    assert C_strides[1] == 1
    assert C.stride(0) == 3
    assert C.stride(1) == 1

    assert A.get_name() == "A"
    assert A.name == "A"

    B.set_name("B")
    assert B.get_name() == "B"

    B.name = "B2"
    assert B.get_name() == "B2"

def test_set() :
    A = ein.core.RuntimeTensor("A", [3, 3])
    B = ein.core.RuntimeTensor("B", [3, 3])

    A.zero()

    B.set_all(3.0)

    for x in A :
        assert x == 0.0
    
    for x in B :
        assert x == 3.0

    for i in range(3) :
        for j in range(3) :
            A[i, j] = 3 * i + j

    for i in range(3) :
        for j in range(3) :
            assert A[i, j] == 3 * i + j

    B[0, 0:2] = np.array([1.0, 2.0])

    B_expected = [1, 2, 3, 3, 3, 3, 3, 3, 3]

    for x, y in zip(B, B_expected) :
        assert x == y

def test_ops() :
    A = ein.utils.create_random_tensor("A", [3, 3])
    B = ein.utils.create_random_tensor("B", [3, 3])
    C = ein.utils.create_random_tensor("C", [3, 3])
    D = np.array(ein.utils.create_random_tensor("C", [3, 3]))

    # Multiplication
    A_res = A.copy()

    A_res *= 2

    for x, y in zip(A_res, A) :
        assert x == 2 * y
    
    A_res *= B

    for x, y, z in zip(A_res, A, B) :
        assert x == 2 * y * z

    A_res *= C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C) :
        assert x == 2 * y * z * w
    
    A_res *= D

    for x, y, z, w, u in zip(A_res, A, B, C, D.flat) :
        assert x == 2 * y * z * w * u
    
    # Addition
    A_res = A.copy()

    A_res += 2

    for x, y in zip(A_res, A) :
        assert x == 2 + y
    
    A_res += B

    for x, y, z in zip(A_res, A, B) :
        assert x == 2 + y + z

    A_res += C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C) :
        assert x == 2 + y + z + w

    A_res += D

    for x, y, z, w, u in zip(A_res, A, B, C, D.flat) :
        assert x == 2 + y + z + w + u

    # Division
    A_res = A.copy()

    A_res /= 2

    for x, y in zip(A_res, A) :
        assert x == y / 2
    
    A_res /= B

    for x, y, z in zip(A_res, A, B) :
        assert x == y / 2 / z

    A_res /= C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C) :
        assert x == y / 2 / z / w

    A_res /= D

    for x, y, z, w, u in zip(A_res, A, B, C, D.flat) :
        assert x == y / 2 / z / w / u

    # Subtraction
    A_res = A.copy()

    A_res -= 2

    for x, y in zip(A_res, A) :
        assert x == y - 2
    
    A_res -= B

    for x, y, z in zip(A_res, A, B) :
        assert x == y - 2 - z

    A_res -= C[0:3, 0:3]

    for x, y, z, w in zip(A_res, A, B, C) :
        assert x == y - 2 - z - w
    
    A_res -= D

    for x, y, z, w, u in zip(A_res, A, B, C, D.flat) :
        assert x == y - 2 - z - w - u



    
def test_view_creation() :
    A = ein.core.RuntimeTensor("A", [5, 5])
    B = ein.core.RuntimeTensor([5, 5])
    A_view = A[0:3, 0:3]
    B_view = B[0:3, 0:3]

    A_dims = A_view.dims()
    B_dims = B_view.dims()

    assert A_view.rank() == 2
    assert B_view.rank() == 2

    assert A_dims[0] == 3
    assert A_dims[1] == 3
    assert A_view.dim(0) == 3
    assert A_view.dim(1) == 3

    assert B_dims[0] == 3
    assert B_dims[1] == 3
    assert B_view.dim(0) == 3
    assert B_view.dim(1) == 3

    A_strides = A_view.strides()
    B_strides = B_view.strides()

    assert A_strides[0] == 5
    assert A_strides[1] == 1
    assert A_view.stride(0) == 5
    assert A_view.stride(1) == 1

    assert B_strides[0] == 5
    assert B_strides[1] == 1
    assert B_view.stride(0) == 5
    assert B_view.stride(1) == 1

    A_view.set_name("A view")
    assert A_view.get_name() == "A view"
    assert A_view.name == "A view"

    B_view.set_name("B view")
    assert B_view.get_name() == "B view"

    B.name = "B2"
    assert B_view.get_name() == "B view"

def test_view_set() :
    A = ein.core.RuntimeTensor("A", [5, 5])
    B = ein.core.RuntimeTensor([5, 5])
    A_view = A[0:3, 0:3]
    B_view = B[0:3, 0:3]

    A.zero()

    B.set_all(3.0)

    for x in A :
        assert x == 0.0
    
    for x in B :
        assert x == 3.0

    for index in enumerate(ein.utils.TensorIndices(A_view)) :
        A_view[index[1]] = index[0]

    for index in ein.utils.TensorIndices(A_view) :
        assert A_view[index] == 3 * index[0] + index[1]

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A[index] == 3 * index[0] + index[1]
        else :
            assert A[index] == 0

    B_view[0, 0:2] = np.array([1.0, 2.0])

    B_expected = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    for x, y in zip(B, B_expected) :
        assert x == y

def test_view_ops() :
    A = ein.utils.create_random_tensor("A", [5, 5])
    B = ein.utils.create_random_tensor("B", [3, 3])
    C = ein.utils.create_random_tensor("C", [5, 5])
    D = np.array(ein.utils.create_random_tensor("D", [3, 3]))

    # Multiplication
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view *= 2

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] * 2
        else :
            assert A_res[index] == A[index]
    
    A_view *= B

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] * 2 * B[index]
        else :
            assert A_res[index] == A[index]

    A_view *= C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] * 2 * B[index] * C[index]
        else :
            assert A_res[index] == A[index]
    
    A_view *= D

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] * 2 * B[index] * C[index] * D[index]
        else :
            assert A_res[index] == A[index]
    
    # Addition
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view += 2

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] + 2
        else :
            assert A_res[index] == A[index]
    
    A_view += B

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] + 2 + B[index]
        else :
            assert A_res[index] == A[index]

    A_view += C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] + 2 + B[index] + C[index]
        else :
            assert A_res[index] == A[index]
    
    A_view += D

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] + 2 + B[index] + C[index] + D[index]
        else :
            assert A_res[index] == A[index]

    # Subtraction
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view -= 2

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] - 2
        else :
            assert A_res[index] == A[index]
    
    A_view -= B

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] - 2 - B[index]
        else :
            assert A_res[index] == A[index]

    A_view -= C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] - 2 - B[index] - C[index]
        else :
            assert A_res[index] == A[index]
    
    A_view -= D

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] - 2 - B[index] - C[index] - D[index]
        else :
            assert A_res[index] == A[index]

    # Division
    A_res = A.copy()
    A_view = A_res[0:3, 0:3]

    A_view /= 2

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] / 2
        else :
            assert A_res[index] == A[index]
    
    A_view /= B

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] / 2 / B[index]
        else :
            assert A_res[index] == A[index]

    A_view /= C[0:3, 0:3]

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] / 2 / B[index] / C[index]
        else :
            assert A_res[index] == A[index]
    
    A_view /= D

    for index in ein.utils.TensorIndices(A) :
        if index[0] < 3 and index[1] < 3 :
            assert A_res[index] == A[index] / 2 / B[index] / C[index] / D[index]
        else :
            assert A_res[index] == A[index]
    
