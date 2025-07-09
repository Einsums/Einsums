..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.linearalgebra :

************************
Linear Algebra Functions
************************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

These are the linear algebra functions that have been made available to Python.

.. py:currentmodule:: einsums.core

.. py:class:: Norm

    Enumeration of different types of matrix norms.

    .. py:attribute:: MAXABS
        :value: 'M'

        Finds the largest absolute value of the elements in a matrix.

    .. py:attribute:: ONE
        :value: '1'

        Finds the one-norm of a matrix. This is equivalent to finding the maximum column sum of the matrix.

    .. py:attribute:: INFINITY
        :value: 'I'

        Finds the infinity-norm of a matrix. This is equivalent to finding the maximum row sum of the matrix.

    .. py:attribute:: FROBENIUS
        :value: 'F'

        Finds the Frobenius norm of a matrix. This is the square root of the sum of the squares of the elements.
        Similar to the vector norm, but applied to matrices.

.. py:class:: Vectors

    Enumerations of different operations for :py:func:`einsums.core.svd_dd`.

    .. py:attribute:: ALL
        :value: 'A'

        Gets all of the vectors from the singular value decomposition.

    .. py:attribute:: SOME
        :value: 'S'

        Only some of the vectors are returned. The number of vectors is the minimum of the dimensions of the
        input matrix.

    .. py:attribute:: OVERWRITE
        :value: 'O'

        Overwrite some of the singular vectors into the input matrix. See the LAPACK documentation for ``gesdd``
        for more details.

    .. py:attribute:: NONE
        :value: 'N'

        Do not compute the vectors for the singular value decomposition.


.. py:function:: sum_square(A) -> float, float

    Computes the sum of the squares of the values in a vector without overflow. To use the return from this
    function, use the following.

    >>> A = ein.core.create_random_tensor("A", dims, dtype)
    >>> sum_sq, scale = ein.core.sum_square(A)
    >>> actual = sum_sq * scale ** 2

    :param A: The vector to analyze.
    :return: The square of the sum of the vector and the scale factor to apply to it to avoid overflow.
    :raises einsums.core.rank_error: if the tensor passed in is not rank-1.
    :raises ValueError: if the data stored by the vector is not real or complex floating point.

.. py:function:: gemm(transA: str, transB: str, alpha, A, B, beta, C)

    Performs a matrix multiplication.

    .. math::

        \mathbf{C} := \beta \mathbf{C} + \alpha \mathbf{A B}

    :param transA: Whether to transpose the ``A`` matrix. ``'T'`` transposes, ``'N'`` does not.
    :param transB: Whether to transpose the ``B`` matrix. ``'T'`` transposes, ``'N'`` does not.
    :param alpha: The scale factor for the matrix product.
    :param A: The left matrix.
    :param B: The right matrix.
    :param beta: The scale factor on the accumulation matrix. If zero, then the output matrix will not be
    mixed into the matrix product.
    :param C: The output matrix.
    :raises einsums.core.rank_error: if any of the tensors is not rank-2.
    :raises einsums.core.tensor_compat_error: if the rows and columns of the matrices are incompatible.
    :raises ValueError: if the storage types of the matrices are not compatible.

.. py:function:: gemv(transA: str, alpha, A, X, beta, Y)

        Performs a matrix-vector multiplication.

    .. math::

        \mathbf{Y} := \beta \mathbf{Y} + \alpha \mathbf{A X}

    :param transA: Whether to transpose the ``A`` matrix. ``'T'`` transposes, ``'N'`` does not.
    :param alpha: The scale factor for the matrix product.
    :param A: The matrix.
    :param X: The input vector.
    :param beta: The scale factor on the accumulation vector. If zero, then the output vector will not be
    mixed into the matrix-vector product.
    :param X: The output vector.
    :raises einsums.core.rank_error: if the ``X`` or ``Y`` tensors are not rank-1 or the 
    ``A`` matrix is not rank-2.
    :raises einsums.core.tensor_compat_error: if the rows and columns of the matrix are incompatible with
    the vectors.
    :raises ValueError: if the storage types of the matrices are not compatible.

.. py:function:: syev(A, W)
.. py:function:: heev(A, W)

    Perform the eigendecomposition of a real symmetric or complex hermitian matrix. That is, solve the following.

    .. math::

        \mathbf{Av} = \mathbf{v}\lambda

    :param A: The input matrix. At exit, it will be overwritten by the eigenvectors.
    :param W: The output vector for the eigenvalues. This must hold only real values, since
    the eigenvalues of symmetric/hermitian matrices are always real.
    :raises einsums.core.rank_error: if ``A`` is not a matrix or ``W`` is not a vector.
    :raises ValueError: if the storage types of ``A`` and ``W`` are incompatible or one of the arguments
    passed to the underlying library call had an illegal value. This second case should hopefully never happen.
    :raisees RuntimeError: if the algorithm does not converge, or the memory for internal buffers could not
    be allocated.

.. py:function:: geev(jobvl: str, jobvr: str, A, W, Vl, Vr)

    Perform the eigendecomposition of a general matrix. The left and right eigenvectors are able to 
    be computed. That is, solve the following to get the left eigenvectors.

    .. math::

        \mathbf{uA} = \mathbf{u}\lambda

    And for the right eigenvectors, the following.

    .. math::

        \mathbf{Av} = \mathbf{v}\lambda

    :param jobvl: Whether to compute the left eigenvectors. Pass ``'V'`` to compute them, ``'N'`` to not.
    If the vectors are computed, then a tensor needs to be passed to ``Vl``.
    :param jobvr: Whether to compute the right eigenvectors. Pass ``'V'`` to compute them, ``'N'`` to not.
    If the vectors are computed, then a tensor needs to be passed to ``Vr``.
    :param A: The matrix to decompose.
    :param W: The output for the eigenvalues. This needs to be complex.
    :param Vl: The output for the left eigenvectors. If ``jobvl = 'N'``, then this is not referenced and can
    be set to ``None``.
    :param Vr: The output for the right eigenvectors. If ``jobvr = 'N'``, then this is not referenced and can
    be set to ``None``.
    :raises einsums.core.rank_error: if the input matrix or any of the referenced output matrices are not rank-2,
    or the ``W`` vector is not rank-1.
    :raises einsums.core.tensor_compat_error: if ``A`` is not a square matrix or any of the outputs don't have
    the proper dimensions.
    :raises TypeError: if a set of eigenvectors is requested, but the output tensor is ``None``.
    :raises ValueError: if the storage types of any of the tensors is incompatible.

.. py:function:: gesv(A, B)

    Solve a linear system like the following.

    .. math::

        \mathbf{Ax} = \mathbf{B}

    :param A: The coefficient matrix. Needs to be square.
    :param B: The result matrix. It can have multiple columns representing different linear systems with
    the same coefficients. It will be overwritten by the values of ``x`` in the equation above.
    :raises einsums.core.rank_error: if ``A`` is not rank-2 or ``B`` is not rank-1 or rank-2.
    :raises einsums.core.dimension_error: if ``A`` is not square.
    :raises einsums.core.tensor_compat_error: if the number of rows in ``B`` does not match the number of
    rows in ``A``.
    :raises ValueError: if the storage types of ``A`` and ``B`` are invalid or an invalid value was passed
    to the underlying library function. This second case should not happen.
    :raises RuntimeError: if the input matrix is singular, meaning no solutions could be found.

.. py:function:: scale(alpha, A)

    Scale a tensor by a scale factor.

    :param alpha: The scale factor.
    :param A: The tensor to scale.
    :raises ValueError: if ``A`` does not store real or complex floating point data.

.. py:function:: scale_row(row: int, alpha, A)

    Scales a row of a matrix by a scale factor.

    :param row: Which row to scale. If ``row`` is negative, it will be treated from the end of the matrix.
    For instance, ``row = -1`` will scale the last row.
    :param alpha: The scale factor.
    :param A: The matrix to scale.
    :raises einsums.core.rank_error: if ``A`` is not a matrix.
    :raises IndexError: if the requested row is outside of the range of the matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: scale_column(col: int, alpha, A)

    Scales a column of a matrix by a scale factor.

    :param col: Which column to scale. If ``col`` is negative, it will be treated from the end of the matrix.
    For instance, ``col = -1`` will scale the last column.
    :param alpha: The scale factor.
    :param A: The matrix to scale.
    :raises einsums.core.rank_error: if ``A`` is not a matrix.
    :raises IndexError: if the requested column is outside of the range of the matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: dot(A, B)

    Performs the unconjugated dot product. For real arguments, this is the dot product.
    For complex arguments, it is not. This is equivalent to the following formula.
    The reason this is still called ``dot`` is because of how useful this function is,
    even if the result isn't what mathematicians would call the dot product.

    .. math::
        
        out = \sum_i A_i B_i
    
    One thing to note is that if the same tensor is passed for both ``A`` and ``B``, then 
    this will not return the norm-squared for complex arguments. It will for real arguments
    simply because real arguments don't feel the effects of conjugation. This function is
    used when you want to take two tensors, multiply them element-wise, and sum the result.
    If you need the geometric dot product, consider using :py:func:`true_dot` instead.
    This is not an inner product for complex arguments, since it does not follow
    conjugate symmetry or positive-definiteness.

    :param A: One input tensor.
    :param B: The other input tensor.
    :return: The unconjugated dot product.
    :raises einsums.core.rank_error: if the input tensors do not have the same rank.
    :raises einsums.core.dimension_error: if the input tensors do not have the same dimensions.
    :raises ValueError: if the tensors do not store the same data type or the stored data type is
    not real or complex floating point.

.. py:function:: true_dot(A, B)

    Performs the possibly conjugated dot product. This is the true dot product. That is,
    if the same tensor is passed to both arguments, the norm-squared will be returned.
    This is equivalent to the following formula.

    .. math::
        
        out = \sum_i A_i^* B_i

    For real arguments, this is equivalent to :py:func:`dot`. The difference lies in the
    complex behavior. This is a true inner product for complex arguments.
    
    :param A: One input tensor. The values of this tensor will be conjugated before being used.
    This is done after accessing the elements, so there will be no change to this tensor.
    :param B: The other input tensor.
    :return: The true dot product.
    :raises einsums.core.rank_error: if the input tensors do not have the same rank.
    :raises einsums.core.dimension_error: if the input tensors do not have the same dimensions.
    :raises ValueError: if the tensors do not store the same data type or the stored data type is
    not real or complex floating point.

.. py:function:: axpy(alpha, x, y)

    Performs a scale and add operation. It is similar to :py:func:`axpby` where the ``beta`` 
    argument is set to 1. In mathematical notation, this performs the following.

    .. math::

        \mathbf{y} := \mathbf{y} + \alpha \mathbf{x}

    :param alpha: The scale factor for the input tensor.
    :param x: The input tensor.
    :param y: The output tensor. Its value is used as an input as well.
    :raises einsums.core.rank_error: if the tensors do not have the same rank.
    :raises ValueError: if the tensors do not have the same storage type or the tensors
    do not store real or complex floating point data.
    :raises einsums.core.tensor_compat_error: if the tensors do not have the same dimensions.

.. py:function:: axpby(alpha, x, beta, y)

    Performs a scale and add operation. It is similar to :py:func:`axpy`, but the ``y`` tensor
    is also scaled before being accumulated. In mathematical notation, this performs the following.    
    
    .. math::

        \mathbf{y} := \beta \mathbf{y} + \alpha \mathbf{x}

    :param alpha: The scale factor for the input tensor.
    :param x: The input tensor.
    :param beta: The scale factor for the output tensor on input.
    :param y: The output tensor. Its value is used as an input as well.
    :raises einsums.core.rank_error: if the tensors do not have the same rank.
    :raises ValueError: if the tensors do not have the same storage type or the tensors
    do not store real or complex floating point data.
    :raises einsums.core.tensor_compat_error: if the tensors do not have the same dimensions.

.. py:function:: ger(alpha, x, y, A)

    Performs an outer product update. This is essentially equivalent to the following.

    .. math::

        \mathbf{A} := \mathbf{A} + \alpha \mathbf{x} \mathbf{y}^T

    Or, in index notation, the following.

    .. math::

        A_{ij} := A_{ij} + \alpha x_i y_j
    
    :param alpha: The scale factor for the outer product.
    :param x: The left input vector.
    :param y: The right input vector.
    :param A: The output matrix.
    :raises einsums.core.rank_error: if ``x`` and ``y`` are not rank-1 or ``A`` is not rank-2.
    :raises einsums.core.dimension_error: if the inputs do not have compatible dimensions.
    :raises ValueError: if the inputs do not store the same data type or the data type stored
    is not real or complex floating point data.

.. py:function:: getrf(A) -> list[int]

    Performs LU decomposition. Essentially performs the following.

    .. math::

        A = PLU

    In the equation above, ``P`` is the pivot matrix, which is represented by the ``pivot`` argument,
    ``L`` is a lower triangular matrix with 1 in all diagonal entries, and ``U`` is an upper triangular
    matrix. On exit, the upper triangle of ``A`` will contain ``U`` and the lower triangle of ``A``
    will contain ``L``. The diagonal entries of ``L`` are not stored since they are all 1. The diagonal
    entries of ``A`` will be the diagonal entries of ``U``. The pivot vector will contain a list of
    the pivots that took place. A pivot vector that looks like ``[3, 4, 4, 4]`` means that first, the
    first row was swapped with the third row, then the second with the fourth, then the third with the
    fourth, and the fourth row was not moved. Since this ultimately calls Fortran, these pivot values
    are 1-indexed, the ``3`` in the list would actually refer to ``A[2, :]``. To extract the data from
    this, use :py:func:`extract_plu`.

    This function will give a warning if the matrix is singular.

    :param A: The matrix to decompose.
    :returns: The pivot list. Pass this into :py:func:`extract_plu` to get the matrix factors.
    :raises einsums.core.rank_error: if the tensor is not rank-2.
    :raises ValueError: if the tensor does not store real or complex floating point data or if an invalid
    argument is passed to the internal ``getrf`` call. This last case should not happen.

.. py:function:: extract_plu(A, pivot: list[int]) -> tuple

    Extracts the matrices from a call to :py:func:`getrf`.

    :param A: The matrix to process after a call to :py:func:`getrf`.
    :param pivot: The return value from :py:func:`getrf`.
    :return: Gives the pivot matrix, the lower triangular factor, and the upper triangular factor
    in that order.
    :raises einsums.core.rank_error: if the input tensor is not a matrix.
    :raises RuntimeError: if the pivot list is not formatted correctly.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: getri(A, pivot: list[int])

    Computes the matrix inverse based on the data returned from :py:func:`getrf`. This does not
    take the inverse of ``A``. The input must have been modified by :py:func:`getrf` before
    calling this function.

    :param A: The matrix output from :py:func:`getrf`. After calling this function, this will
    contain the matrix inverse.
    :param pivot: The pivot list from :py:func:`getrf`.
    :raises einsums.core.rank_error: if the input tensor is not a matrix.
    :raises einsums.core.dimension_error: if the matrix is not square.
    :raises ValueError: if the pivot list has not been formatted properly, the input tensor
    does not store real or complex floating point data, or an illegal argument is passed to
    the underlying ``getri`` call. This last one should not happen.
    :raises RuntimeError: if the matrix is singular.

.. py:function:: invert(A)

    Computes the matrix inverse. This calls :py:func:`getrf` and :py:func:`getri` under the hood
    so that you don't have to worry about them.

    :param A: The matrix to invert. After calling this function, this will contain the inverse matrix.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises einsums.core.dimension_error: if the input is not a square matrix.
    :raises ValueError: if the pivot list has not been formatted properly, the input tensor
    does not store real or complex floating point data, or an illegal argument is passed to
    one of the underlying LAPACK calls. This last one should not happen.
    :raises RuntimeError: if the matrix is singular.

.. py:function:: norm(norm_type: einsums.core.Norm, A)

    Computes the norm of a matrix. Does not handle vectors.

    :param norm_type: The kind of norm to take.
    :param A: The matrix to use.
    :return: The norm of the matrix.
    :raises einsums.core.rank_error: if the input tensor is not a matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: vec_norm(A)

    Computes the norm of a vector.

    :param A: The vector to use.
    :return: The norm of the vector.
    :raises einsums.core.rank_error: if the input is not a vector.
    :raises ValueError: if the vector does not store real or complex floating point data.

.. py:function:: svd(A) -> tuple

    Performs singular value decomposition on a matrix.

    :param A: The matrix to decompose.
    :return: The left singular vectors, the singular value vector, and the right singular vectors in a tuple.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises ValueError: if the matrix does not store real or complex floating point values, or an
    illegal argument is passed to the underlying LAPACK call. This last one should never happen.
    :raises RuntimeError: if the SVD iterations did not converge.

.. py:function:: svd_nullspace(A)

    Computes the nullspace of a matrix using singular value decomposition.

    :param A: The matrix to use.
    :return: The nullspace basis as a matrix.
    :raises einsums.core.rank_error: if the input tensor is not a matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data, or an 
    invalid argument was passed to the underlying LAPACK call. This last one should not happen.
    :raises RuntimeError: if the SVD iterations did not converge.

.. py:function:: svd_dd(A, job: einsums.core.Vectors = einsums.core.ALL) -> tuple

    Performs singular value decomposition on a matrix using the divide and conquer algorithm.

    :param A: The matrix to decompose.
    :param job: Determines which vectors to compute.
    :return: The left singular vectors, the singular value vector, and the right singular vectors in a tuple.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises ValueError: if the matrix does not store real or complex floating point values, or an
    illegal argument is passed to the underlying LAPACK call. This last one should never happen.
    :raises RuntimeError: if the SVD iterations did not converge.

.. py:function:: truncated_svd(A, k: int) -> tuple

    Computes the singular value decomposition, but truncates the number of singular values.

    :param A: The matrix to decompose.
    :param k: The number of singular values to use.
    :return: A tuple containing the left singular vectors, the list of singular values, and the right singular
    vectors.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: truncated_syev(A, k: int) -> tuple

    Computes the eigendecomposition, but truncates the number of eigenvalues.

    :param A: The matrix to decompose.
    :param k: The number of eigenvalues to use.
    :return: A tuple containing the eigenvectors and the list of eigenvalues.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises einsums.core.dimension_error: if the input is not a square matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: pseudoinverse(A, tol: float)

    Computes the pseudoinverse of a matrix.

    :param A: The matrix to pseudoinvert.
    :param tol: The tolerance on the singular values.
    :return: The pseudoinverse of the input matrix.
    :raises einsums.core.rank_error: if the input tensor is not a matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data.

.. py:function:: solve_continuous_lyapunov(A, Q)

    Solves a continuous Lyapunov equation. This is an equation of the following form.

    .. math::

        \mathbf{AX} + \mathbf{XA}^H + \mathbf{Q} = 0

    :param A: The A matrix.
    :param Q: The Q matrix.
    :return: The ``X`` matrix that solves this equation.
    :raises einsums.core.rank_error: if either input is not a matrix.
    :raises einsums.core.dimension_error: if the inputs are not square matrices or are not compatible
    with each other.
    :raises ValueError: if the input tensors do not have the same storage type, the inputs do not store
    real or complex floating point data, or an invalid argument is passed to one of the underlying
    LAPACK functions. This last case should hopefully not happen.
    :raises RuntimeError: if the Schur decomposition step fails to converge.

.. py:function:: qr(A) -> tuple

    Perform QR decomposition. The information to get Q and R are returned.

    :param A: The matrix to decompose.
    :return: A tuple to be passed to :py:func:`q` and :py:func:`r`.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises ValueError: if the matrix input does not store real or complex floating point data,
    or an invalid value was passed to the underlying LAPACK call. The second case should not happen.

.. py:function:: q(QR, tau)

    Extract the Q factor from the return from :py:func:`qr`.

    :param QR: The first returned value from :py:func:`qr`.
    :param tau: The second returned value from :py:func:`qr`.
    :return: The Q matrix from the decomposition.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises ValueError: if the matrix inputs do not store the same data type, 
    do not store real or complex floating point data,
    or an invalid value was passed to the underlying LAPACK call. The second case should not happen.

.. py:function:: r(QR, tau)

    Extract the R factor from the return from :py:func:`qr`.

    :param QR: The first returned value from :py:func:`qr`.
    :param tau: The second returned value from :py:func:`qr`. Unused, but present to make it look like
    the corresponding :py:func:`q` call.
    :return: The R matrix from the decomposition.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises ValueError: if the matrix inputs do not store real or complex floating point data.

.. py:function:: direct_product(alpha, A, B, beta, C)

    Performs the following formula.

    .. math::

        C_i := \beta C_i + \alpha A_i B_i

    :param alpha: The scale factor for the product.
    :param A: The first tensor in the product.
    :param B: The second tensor in the product.
    :param beta: The scale factor for the accumulation tensor.
    :param C: The accumulation tensor.
    :raises einsums.core.rank_error: if the tensors do not have the same rank.
    :raises einsums.core.dimension_error: if the tensors have different dimensions.
    :raises ValueError: if the tensors do not store the same data type or they do not
    store real or complex floating point data.

.. py:function:: det(A)

    Computes the determinant of a matrix.

    :param A: The matrix to use.
    :return: The determinant of the matrix.
    :raises einsums.core.rank_error: if the input is not a matrix.
    :raises einsums.core.dimension_error: if the input is not a square matrix.
    :raises ValueError: if the matrix does not store real or complex floating point data, or an
    argument passed to the underlying LAPACK call was invalid. The second case should not happen.
