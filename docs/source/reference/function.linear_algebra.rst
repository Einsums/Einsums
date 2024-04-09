..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _function.linear_algebra:

Linear Algebra
==============

.. sectionauthor:: Justin M. Turney

This module contains functions for performing linear algebra operations on tensors.

BLAS Level 1 Routines
---------------------

.. doxygenfunction:: einsums::linear_algebra::dot(const Type<T, 1> &A, const Type<T, 1> &B)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::dot(const Type<T, Rank> &A, const Type<T, Rank> &B)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::dot(const Type<T, Rank> &A, const Type<T, Rank> &B, const Type<T, Rank> &C)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::scale(T scale, AType<T, ARank>* A)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::scale_column(size_t col, T scale, AType<T, ARank>* A)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::scale_row(size_t row, T scale, AType<T, ARank>* A)
    :project: Einsums

BLAS Level 2 Routines
---------------------

.. doxygenfunction:: einsums::linear_algebra::axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y)
    :project: Einsums

.. For some reason gemv does not want to be documented
.. doxygenfunction:: einsums::linear_algebra::gemv(const T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const T beta, YType<T, XYRank> *y)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A)
    :project: Einsums

BLAS Level 3 Routines
---------------------

.. The LONG function signature is needed because there are multiple functions named gemm and it's how to differentiate them in the documentation.
.. doxygenfunction:: einsums::linear_algebra::gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C)(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B)
    :project: Einsums

LAPACK Routines
---------------

.. doxygenfunction:: einsums::linear_algebra::geev
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::gesv
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::getri(TensorType<T, TensorRank> *A, const std::vector<blas_int> &pivot)

.. doxygenfunction:: einsums::linear_algebra::getrf(TensorType<T, TensorRank> *A, std::vector<blas_int> *pivot)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::heev
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::invert(TensorType<T, TensorRank> *A)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::pow
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::sum_square
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::svd
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::svd_nullspace
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::syev(AType<T, ARank> *A, WType<T, WRank> *W)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::syev(const AType<T, ARank> &A)
    :project: Einsums
