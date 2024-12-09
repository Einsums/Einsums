..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_LinearAlgebra:

Linear Algebra
==============

This module contains the user-facing definitions of various linear algebra routines.

See the :ref:`API reference <modules_LinearAlgebra_api>` of this module for more
details.

Here are some of the public symbols that are available to use.

.. cpp:function:: template<TensorConcept AType> void sum_square(const AType &A, RemoveComplexT<typename AType::ValueType> *scale, RemoveComplexT<typename AType::ValueType> *sumsq)

    Computes the sum of the squares of the elements of a tensor. This uses the following formula.

    .. math::

        scale_{out}^2 * sum_{out} = A_1^2 + A_2^2 + \cdots + A_n^2 + scale_{in}^2 * sum_{in}

    On exit, the value of :code:`scale` will normally be 1. However, it may be different if the 
    result of the sum would have caused either an overflow or an underflow.

    Currently only supports :code:`BasicTensor`s of rank 1.

    :param A: The tensor to evaluate.
    :param scale: The scale factor. It is both an input and output parameter.
    :param sumsq: The starting point and result. It is both an input and output parameter.
    :tparam AType: The type of the tensor A.

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

    :param alpha: Scale factor for the matrix product.
    :param A: The left matrix in the product.
    :param B: The right matrix in the product.
    :param beta: The scale factor to apply to the C matrix on input.
    :param C: The result. If :code:`beta` is not zero, then the value of :code:`C` on input will be scaled and added to the result.

    :tparam TransA: If true, transpose the A matrix.
    :tparam TransB: If true, transpose the B matrix.
    :tparam AType: The type of the A matrix.
    :tparam BType: The type of the B matrix.
    :tparam CType: The type  of the C matrix.
    :tparam U: The type of the scale factors. It will be converted if needed.

.. cpp:function:: template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, typename U> auto gemm(U const alpha, AType const &A, BType const &B) -> RemoveViewT<AType>

    Compute the matrix product between two matrices. This is a wrapper around the previous :code:`gemm`,
    but instead of returning its result in an output argument, it returns its result as an output
    value. It supports all the same combinations as the other definition of :code:`gemm`, but if it is passed
    a view, it will remove that view.

    :param alpha: The scale factor on the matrix product.
    :param A: The left matrix.
    :param B: The right matrix.
    :return: The matrix product scaled by :code:`alpha`.

.. cpp:function:: template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType> void symm_gemm(AType const &A, BType const &B, CType *C)

    This function computes :math:`OP(B)^T OP(A) OP(B) = C`. It supports the same arguments
    as :code:`gemm`, since it normally calls :code:`gemm` in the back.

    :param A: The middle tensor.
    :param B: The tensor that will be multiplied on either side.
    :param C: The output of the operation.
    
    :tparam TransA: Whether to transpose A.
    :tparam TransB: Whether to transpose the second instance of B. The first instance will always be the opposite.
    :tparam AType: The matrix type of A.
    :tparam BType: The matrix type of B.
    :tparam CType: The matrix type of the output.

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

    :param alpha: The scale factor on the product.
    :param A: The matrix in the product.
    :param z: The vector in the product.
    :param beta: The scale factor on the result vector.
    :param y: The result vector. If :code:`beta` is not zero, then the value of this on entry will be scaled and added to the result.

    :tparam TransA: Whether to transpose the matrix.
    :tparam AType: The type of the matrix.
    :tparam XType: The type of the input vector.
    :tparam YType: The type of the output vector.
    :tparam U: The type of the scale factors. If it is not the same as the types stored by the tensors, it will be cast to match.

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

.. cpp:function:: template <bool ComputeLeftRightEigenvectors = true, MatrixConcept AType, VectorConcept WType> void geev(AType *A, WType *W, AType *lvecs, AType *rvecs)

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

    :param A: The matrix to decompose. It will be overwritten on exit.
    :param W: The eigenvalues of the matrix.
    :param lvecs: If specified, it will contain the left eigenvectors.
    :param rvecs: If specified, it will contain the right eigenvectors.

    :tparam ComputeLeftrightEigenvectors: If true, the eigenvectors will be computed.
    :tparam AType: The type of the matrix and the vector outputs.
    :tparam WType: The type of the value output.

.. todo::

    Finish this file.

