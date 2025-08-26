..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_LinearAlgebra:

Linear Algebra
==============

This module contains the user-facing definitions of various linear algebra routines.

See the :ref:`API reference <modules_Einsums_LinearAlgebra_api>` of this module for more
details.

Public API
----------

Here are some of the public symbols that are available to use. More can be found in the :ref:`API reference <modules_Einsums_LinearAlgebra_api>`.

.. cpp:function:: template<TensorConcept AType> void sum_square(const AType &A, RemoveComplexT<typename AType::ValueType> *scale, RemoveComplexT<typename AType::ValueType> *sumsq)

    Computes the sum of the squares of the elements of a tensor. This uses the following formula.

    .. math::

        scale_{out}^2 * sum_{out} = A_1^2 + A_2^2 + \cdots + A_n^2 + scale_{in}^2 * sum_{in}

    On exit, the value of :code:`scale` will normally be 1. However, it may be different if the 
    result of the sum would have caused either an overflow or an underflow.

    Currently only supports :code:`BasicTensor`s of rank 1.

    :param A[in]: The tensor to evaluate.
    :param scale[inout]: The scale factor. It is both an input and output parameter.
    :param sumsq[inout]: The starting point and result. It is both an input and output parameter.
    :tparam AType: The type of the tensor A.

    .. versionadded:: 1.0.0

.. cpp:function:: template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U> gemm(U alpha, AType const &A, BType const &B, U beta, CType *C)

    Compute the matrix product between two matrices.

    Currently supports the following:
    
    For all of these, :code:`U, T = float, double, std::complex<float>, std::complex<double>`.

    For all of these, the rank is always rank 2.

    All of the stored types must match, though the prefactor's type does not. It will be converted
    to match if it does not.

    * :code:`BasicTensor * BasicTensor = BasicTensor`
    * :code:`BlockTensor * BlockTensor = BlockTensor`
    * :code:`TiledTensor * TiledTensor = TiledTensor`
    * :code:`OtherTensor * OtherTensor = OtherTensor`

    :param alpha[in]: Scale factor for the matrix product.
    :param A[in]: The left matrix in the product.
    :param B[in]: The right matrix in the product.
    :param beta[in]: The scale factor to apply to the C matrix on input.
    :param C[inout]: The result. If :code:`beta` is not zero, then the value of :code:`C` on input will be scaled and added to the result.

    :tparam TransA: If true, transpose the A matrix.
    :tparam TransB: If true, transpose the B matrix.
    :tparam AType: The type of the A matrix.
    :tparam BType: The type of the B matrix.
    :tparam CType: The type  of the C matrix.
    :tparam U: The type of the scale factors. It will be converted if needed.

    .. versionadded:: 1.0.0

.. cpp:function:: template <MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U> gemm(char transA, char transB, U alpha, AType const &A, BType const &B, U beta, CType *C)

    Compute the matrix product between two matrices.

    Currently supports the following:
    
    For all of these, :code:`U, T = float, double, std::complex<float>, std::complex<double>`.

    For all of these, the rank is always rank 2.

    All of the stored types must match, though the prefactor's type does not. It will be converted
    to match if it does not.

    * :code:`BasicTensor * BasicTensor = BasicTensor`
    * :code:`BlockTensor * BlockTensor = BlockTensor`
    * :code:`TiledTensor * TiledTensor = TiledTensor`
    * :code:`OtherTensor * OtherTensor = OtherTensor`

    :param transA[in]: Whether to transpose the A matrix. Case insensitive. Can be either 'n', 't', or 'c'.
    :param transB[in]: Whether to transpose the A matrix. Case insensitive. Can be either 'n', 't', or 'c'.
    :param alpha[in]: Scale factor for the matrix product.
    :param A[in]: The left matrix in the product.
    :param B[in]: The right matrix in the product.
    :param beta[in]: The scale factor to apply to the C matrix on input.
    :param C[inout]: The result. If :code:`beta` is not zero, then the value of :code:`C` on input will be scaled and added to the result.

    :tparam AType: The type of the A matrix.
    :tparam BType: The type of the B matrix.
    :tparam CType: The type  of the C matrix.
    :tparam U: The type of the scale factors. It will be converted if needed.

    .. versionadded:: 1.0.0

.. cpp:function:: template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, typename U> auto gemm(U const alpha, AType const &A, BType const &B) -> RemoveViewT<AType>

    Compute the matrix product between two matrices. This is a wrapper around the previous :code:`gemm`,
    but instead of returning its result in an output argument, it returns its result as an output
    value. It supports all the same combinations as the other definition of :code:`gemm`, but if it is passed
    a view, it will remove that view.

    :param alpha[in]: The scale factor on the matrix product.
    :param A[in]: The left matrix.
    :param B[in]: The right matrix.

    :tparam TransA: Whether to transpose the A matrix.
    :tparam TransB: Whether to transpose the B matrix.

    :return: The matrix product scaled by :code:`alpha`.

    .. versionadded:: 1.0.0

.. cpp:function:: template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType> void symm_gemm(AType const &A, BType const &B, CType *C)

    This function computes :math:`OP(B)^T OP(A) OP(B) = C`. It supports the same arguments
    as :code:`gemm`, since it normally calls :code:`gemm` in the back.

    :param A[in]: The middle tensor.
    :param B[in]: The tensor that will be multiplied on either side.
    :param C[out]: The output of the operation.
    
    :tparam TransA: Whether to transpose A.
    :tparam TransB: Whether to transpose the second instance of B. The first instance will always be the opposite.
    :tparam AType: The matrix type of A.
    :tparam BType: The matrix type of B.
    :tparam CType: The matrix type of the output.

    .. versionadded:: 1.0.0

.. cpp:function:: template <bool TransA, MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U> void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y)

    Computes the matrix-vector product.

    Currently supports the following:

    For each of the arguments, :code:`U,T = float, double, std::complex<float>, std::complex<double>`.

    The stored types of each of the tensors must match.

    The rank of :code:`A` is 2 and the rank of :code:`X` and :code:`Y` is 1.

    * BasicTensor * BasicTensor = BasicTensor
    * BlockTensor * BasicTensor = BasicTensor
    * TiledTensor * BasicTensor = BasicTensor
    * TiledTensor * TiledTensor = BasicTensor
    * TiledTensor * BasicTensor = TiledTensor
    * TiledTensor * TiledTensor = TiledTensor
    * OtherTensor * OtherTensor = OtherTensor

    :param alpha[in]: The scale factor on the product.
    :param A[in]: The matrix in the product.
    :param z[in]: The vector in the product.
    :param beta[in]: The scale factor on the result vector.
    :param y[inout]: The result vector. If :code:`beta` is not zero, then the value of this on entry will be scaled and added to the result.

    :tparam TransA: Whether to transpose the matrix.
    :tparam AType: The type of the matrix.
    :tparam XType: The type of the input vector.
    :tparam YType: The type of the output vector.
    :tparam U: The type of the scale factors. If it is not the same as the types stored by the tensors, it will be cast to match.

    .. versionadded:: 1.0.0

.. cpp:function:: template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType> void syev(AType *A, WType *W)

    Computes the eigendecomposition of a symmetrix matrix.

    Supports the following:

    :code:`A` and :code:`W` need to have the same stored type, and that type needs to be real.

    :code:`A` needs to be rank 2 and :code:`W` needs to be rank 1.

    * BasicTensor to BasicTensor
    * BlockTensor to BasicTensor

    :param A: On entry, it is the matrix to decompose. On exit, it contains the eigenvectors in its columns, if told to compute the eigenvectors.
    :param W: On exit, it contains the eigenvalues.

    :tparam ComputeEigenvectors: If true, the eigenvectors will overwrite the :code:`A` matrix.
    :tparam AType: The type of the matrix.
    :tparam WType: The type of the vector.

    .. versionadded:: 1.0.0

.. cpp:function:: template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType> void heev(AType *A, WType *W)

    Computes the eigendecomposition of a Hermitian matrix.

    Supports the following:

    :code:`A` needs to be complex, and :code:`W` needs to be real. The types of the components of :code:`A` need to be the same as the
    type of the values of :code:`W`. For instance, :code:`std::complex<float>` and :code:`float`.

    :code:`A` needs to be rank 2 and :code:`W` needs to be rank 1.

    * BasicTensor to BasicTensor
    * BlockTensor to BasicTensor

    :param A: On entry, it is the matrix to decompose. On exit, it contains the eigenvectors in its columns, if told to compute the eigenvectors.
    :param W: On exit, it contains the eigenvalues.

    :tparam ComputeEigenvectors: If true, the eigenvectors will overwrite the :code:`A` matrix.
    :tparam AType: The type of the matrix.
    :tparam WType: The type of the vector.

    .. versionadded:: 1.0.0

.. cpp:function:: template <MatrixConcept AType, VectorConcept WType> void geev(AType *A, WType *W, AType *lvecs, AType *rvecs)

    Compute the eingendecomposition of a general matrix. If a real matrix has a complex eigenvalue, it will
    always come in a conjugate pair. In this case, the columns of the eigenvector matrix will 
    act as the real and imaginary parts. The first column of the two will be the real part,
    and the second column will be the imaginary part of the first eigenvector. The imaginary
    part of the second eigenvector will be the negative of this vector. This only applies to
    real inputs. If the input is complex, then the eigenvectors will be stored as normal.

    Supports the following:

    :code:`AType` needs to be rank2 and :code:`W` needs to be rank 1.

    :code:`W` needs to store complex values. :code:`A` can be real or complex. The stored
    types much match in precision, so :code:`std::complex<float>` will match either :code:`float`
    or :code:`std::complex<float>`.

    * BasicTensor to BasicTensor
    * BlockTensor to BasicTensor values and BlockTensor vectors

    :param A[inout]: The matrix to decompose. It will be overwritten on exit.
    :param W[out]: The eigenvalues of the matrix.
    :param lvecs[out]: If specified, it will contain the left eigenvectors.
    :param rvecs[out]: If specified, it will contain the right eigenvectors.

    :tparam ComputeLeftrightEigenvectors: If true, the eigenvectors will be computed.
        .. versionremoved:: 2.0.0

    :tparam AType: The type of the matrix and the vector outputs.
    :tparam WType: The type of the value output.

    .. versionadded:: 1.0.0

    .. versionchanged:: 2.0.0
        The option to compute eigenvectors is no longer a template argument. It is now decided by whether the appropriate parameter is
        a null pointer or not. For instance, if :code:`lvecs` is a null pointer then it will not be computed.

.. cpp:function:: template<MatrixConcept AType, TensorConcept BType> int gesv(AType *A, BType *B)

    Solves a system of linear equations.

    :param A[inout[]]: The coefficient matrix. On exit, it contains the upper and lower triangular factors, even if the return value is greater than zero. The
    elements of the lower triangular factor are all 1, so they are not stored.
    :param B[inout]: The constant matrix. If this function returns 0, then on exit, this will contain the solutions.
    :return: If the return value is greater than 0, then the input matrix is singular. If it is less than zero,
    then the input contains an invalid value, such as infinity. If it is zero, then the system could be solved.

    .. versionadded:: 1.0.0

    .. versionchanged:: 2.0.0
        Fixed a bug with how the matrices were being handled. The B tensor is now able to be a vector as well.

.. cpp:function:: template<TensorConcept AType> void scale(typename AType::ValueType scale, AType *A)

    Scales a tensor by a scalar value.

    :param scale: The scale factor to apply.
    :param A: The tensor to scale.

    .. versionadded:: 1.0.0

.. cpp:function:: template<MatrixConcept> void scale_row(size_t row, typename AType::ValueType scale, AType *A)
.. cpp:function:: template<MatrixConcept> void scale_column(size_t col, typename AType::ValueType scale, AType *A)

    Scale a row or column of a matrix.

    :param row,col[in]: The row or column to scale.
    :param scale[in]: The scale factor.
    :param A[inout]: The matrix to scale.

    .. versionadded:: 1.0.0

.. cpp:function:: template<MatrixConcept AType> auto pow(AType const &a, typename AType::ValueType alpha, \
         typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon()) -> RemoveViewT<AType>
    
    Take the matrix power. This is equivalent to diagonalizing the matrix, raising the eigenvalues to the given power,
    then recombining the matrix.

    :param a[in]: The matrix to exponentiate.
    :param alpha[in]: The power to raise the matrix to.
    :param cutoff[in]: If an eigenvalue is below this parameter after exponentiation, then set it to be zero.
    :return: The result of raising the matrix to a power.

    .. versionadded:: 1.0.0

.. cpp:function:: template<TensorConcept AType, TensorConcept BType> auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType>

    Compute the dot product between two tensors. This form does not conjugate either element if complex.

    :param A,B[in]: The tensors to dot together.
    :return: The dot product between the two tensors.
    
    .. versionadded:: 1.0.0

.. cpp:function:: template<TensorConcept AType, TensorConcept BType> auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType>

    Compute the dot product between two tensors. This form conjugates the first parameter if complex.

    :param A,B[in]: The tensors to dot together.
    :return: The dot product between the two tensors.
    
    .. versionadded:: 1.0.0

.. cpp:function:: template<TensorConcept AType, TensorConcept BType, TensorConcept CType> auto dot(AType const &A, BType const &B, CType const &C) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType>

    Computes the dot product between three tensors.

    :param A,B,C[in]: The tensors to dot together.
    :return: The dot product between the three tensors.
    
    .. versionadded:: 1.0.0

.. cpp:function:: template<TensorConcept XType, TensorConcept YType> void axpy(typename XType::ValueType alpha, XType const &X, YType *Y)

    Adds two tensors together. The values from the first tensor will be scaled during addition. 
    This is equivalent to :math:`\mathbf{Y} = \alpha\mathbf{X} + \mathbf{Y}`.

    :param alpha[in]: The scale factor for the input tensor.
    :param X[in]: The input tensor.
    :param Y[out]: The accumulated tensor.

    .. versionadded:: 1.0.0

.. cpp:function:: template<TensorConcept XType, TensorConcept YType> void axpby(typename XType::ValueType alpha, XType const &X, typename XType::ValueType beta, YType *Y)

    Adds two tensors together. The values from the both tensors will be scaled during addition. 
    This is equivalent to :math:`\mathbf{Y} = \alpha\mathbf{X} + \beta\mathbf{Y}`.

    :param alpha[in]: The scale factor for the input tensor.
    :param X[in]: The input tensor.
    :param beta[in]: The scale factor for the accumulated tensor.
    :param Y[out]: The accumulated tensor.

    .. versionadded:: 1.0.0

.. cpp:function:: template<MatrixConcept AType, VectorConcept XYType> void ger(typename AType::ValueType alpha, XYType const &X, XYType const &Y, AType *A)
.. cpp:function:: template<MatrixConcept AType, VectorConcept XYType> void gerc(typename AType::ValueType alpha, XYType const &X, XYType const &Y, AType *A)

    Computes the outer product of two vectors and adds it to the output tensor. Equivalent to :math:`\mathbf{A} = \alpha\mathbf{XY}^T + \mathbf{A}`,
    or :math:`\mathbf{A} = \alpha\mathbf{XY}^H + \mathbf{A}` for :cpp:func:`gerc`.

    :param alpha[in]: The amount to scale the outer product.
    :param X[in]: The left vector.
    :param Y[in]: The right vector.
    :param A[out]: The output matrix.

    .. versionadded:: 1.0.0
        Added :cpp:func:`ger`.
    
    .. versionadded:: 2.0.0
        Added :cpp:func:`gerc`.

.. cpp:function:: template<MatrixConcept TensorType> int getrf(TensorType *A, std::vector<blas::int_t> *pivot)

    Computes the LU factorization of a general :math:`m` by :math:`n` matrix.

    :param A[inout]: The matrix to factorize. On exit, it contains the L and U matrices. The diagonal elements of the
    L matrix are not stored, since they are all 1.
    :param pivot[out]: The pivot table. Indicates which rows were swapped.
    :return: If 0, then the procedure succeded. If positive, the procedure succeeded, but the matrix was singular, so it
    should not be used to solve systems of equations. If negative, the procedure failed.

    .. versionadded:: 1.0.0

.. cpp:function:: template<MatrixConcept TensorType> int getri(TensorType *A, std::vector<blas::int_t> const &pivot)

    Computes the inverse of a matrix using the data obtained from :cpp:func:`getrf`.

    :param A[inout]: The matrix to invert after being processed by :cpp:func:`getrf`. On exit, it contains the inverse of the original matrix.
    :param pivot[in]: The pivot table. Indicates which rows were swapped.
    :return: If 0, then the procedure succeded. Otherwise, the procedure failed.

    .. versionadded:: 1.0.0

.. cpp:function:: template<MatrixConcept TensorType> void invert(TensorType *A)

    Combines :cpp:func:`getrf` and :cpp:func:`getri` into one function call, calculating the inverse of the matrix.

    :param A[inout]: The matrix to invert. On exit, it contains the inverse.
    
    .. versionadded:: 1.0.0

.. cpp:enum:: Norm : char

    Allows selecting of the kind of norm to perform.

    .. cpp:enumerator:: Norm::MAXABS = 'M'

        Use the maximum absolute value of the tensor as the norm.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

    .. cpp:enumerator:: Norm::ONE = '1'

        Use the 1-norm of the matrix, or the maximum column sum. For a vector, use the 1-norm, or sum of the absolute
        values.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

    .. cpp:enumerator:: Norm::INFTY = 'I'

        Use the infinity norm of the matrix, or the maximum row sum. For a vector, use the infinity-norm, or maximum
        absolute value.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

    .. cpp:enumerator:: Norm::FROBENIUS = 'F'

        Use the Frobenius norm of the matrix, or the square root of the sum of squares of the elements. For a
        vector, use the Euclidean norm.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

    .. cpp:enumerator:: Norm::TWO = '2'

        Use the induced 2-norm of a matrix, also called the spectral norm. For a vector, use the Euclidean norm.

        .. versionadded:: 2.0.0

.. cpp:function:: template<TensorConcept AType> auto norm(Norm norm_type, AType const &a) -> RemoveComplexT<typename AType::ValueType>

    Computes the norm of a matrix. The norm can be selected by the first argument, and can come from 
    :cpp:enum:`Norm`.

    :param norm_type[in]: The type of norm to compute.
    :param a[in]: The tensor to process.

    :return: The requested norm.
    
    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        Can now handle vectors.

.. cpp:function:: template <TensorConcept AType> auto vec_norm(AType const &a) -> RemoveComplexT<typename AType::ValueType>

    Compute the Euclidean norm of a vector. This is the usual geometric norm.


    :param a[in]: The tensor to process.
    
    :return: The square root of the sum of the squares of the elements.

    .. versionadded:: 1.0.0

.. cpp:function:: template<MatrixConcept AType> auto svd(AType const &_A, Vectors jobu = Vectors::ALL, Vectors jobvt = Vectors::ALL) -> std::tuple<std::optional<Tensor<typename AType::ValueType, 2>>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>, \
                                        Tensor<typename AType::ValueType, 2>>

    Compute the singular value decomposition of a matrix. The order of the elements in the returned tuple
    is the unitary matrix is first, the rectangular diagonal matrix is second, and the other unitary matrix, transposed. Uses the QR algorithm.

    :param _A[in]: The matrix to decompose.
    :param jobu,jobv[in]: Whether to compute the vectors from the singular value decomposition.

    :return: A tuple which contains the requested tensors.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        The job calculations can now be specified. If a tensor is not computed, the optional object will be empty.

.. cpp:function:: template<MatrixConcept AType> auto svd_nullspace(AType const &_A) -> Tensor<typename AType::ValueType, 2>

    Compute the nullspace of a matrix using singular value decomposition.

    :param _A[in]: The tensor to process.
    
    :return: The nullspace of the input matrix. 

    .. versionadded:: 1.0.0

.. cpp:enum:: Vectors : char

    Allows selecting of the vectors to use for singular value decomposition.

    .. cpp:enumerator:: Vectors::ALL = 'A'

        Compute all vectors.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

    .. cpp:enumerator:: Vectors::SOME = 'S'

        Computes only some of the vectors. The number computed is the same as the smallest dimension
        of the input matrix.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

    .. cpp:enumerator:: Vectors::Overwrite = 'O'

        Overwrites the input matrix with some of the vectors.

        .. versionadded:: 1.0.0
        .. versionremoved:: 2.0.0

    .. cpp:enumerator:: Vectors::NONE = 'N'

        None of the vectors are computed.

        .. versionadded:: 1.0.0
        
        ..versionchanged:: 2.0.0
            Name is now all caps.

.. cpp:function:: template<MatrixConcept AType> auto svd_dd(AType const &_A, Vectors job = Vectors::ALL) \
    -> std::tuple<std::optional<Tensor<typename AType::ValueType, 2>>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>, \
                  std::optional<Tensor<typename AType::ValueType, 2>>> 

    Compute the singular value decomposition of a matrix, optionally selecting the vectors to compute. The order of the elements in the returned tuple
    is the unitary matrix is first, the rectangular diagonal matrix is second, and the other unitary matrix, transposed. Uses the divide-and-conquer algorithm.

    :param _A[in]: The matrix to decompose.
    :param job[in]: Whether to compute the vectors from the singular value decomposition.

    :return: A tuple which contains the requested tensors.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        The job calculations can now be specified. If a tensor is not computed, the optional object will be empty.

.. cpp:function:: template<MatrixConcept AType> auto qr(AType const &_A) -> std::tuple<Tensor<typename AType::ValueType, 2>, Tensor<typename AType::ValueType, 2>>

    Compute the QR decomposition of the input matrix.

    :param _A[in]: The matrix to decompose.

    :return: The Q and R matrices in a tuple.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        The function now returns the Q and R matrices directly.

.. cpp:function:: template<MatrixConcept AType, VectorConcept TauType> auto q(AType const &qr, TauType const &tau) -> Tensor<typename AType::ValueType, 2>

    Use the data from :cpp:func:`qr` to compute the Q matrix.

    .. versionadded:: 1.0.0
    .. versionremoved:: 2.0.0

.. cpp:function:: template <TensorConcept AType, TensorConcept BType, TensorConcept CType, typename T> void direct_product(T alpha, AType const &A, BType const &B, T beta, CType *C)

    Compute the direct product. This is essentially the element-wise product between two matrices.

    :param alpha[in]: The scale factor for the product.
    :param A,B[in]: The tensors to multiply.
    :param beta[in]: The scale factor for the accumulation tensor.
    :param C[out]: The accumulation tensor.

    .. versionadded:: 1.0.0

.. cpp:function:: template <MatrixConcept AType> typename AType::ValueType det(AType const &A)

    Computes the determinant of a matrix.

    :param A[in]: The matrix to analyze.
    
    :return: The determinant of the matrix.

    .. versionadded:: 1.0.0