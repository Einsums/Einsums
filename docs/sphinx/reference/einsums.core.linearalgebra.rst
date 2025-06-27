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