..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _function.linear_algebra:

Linear Algebra (BLAS, LAPACK, etc.)
===================================

.. sectionauthor:: Justin M. Turney

This module contains functions for performing high-level linear algebra operations on tensors.

The functions in this module are designed to be used with the `Tensor` class and its subclasses.

BLAS Level 1 Functions
----------------------

BLAS Level 1 routines perform operations of both addition and reduction on vectors of data.
Typical operations include scaling and dot products.

.. doxygenfunction:: einsums::linear_algebra::dot(const AType &A, const BType &B)

.. doxygenfunction:: einsums::linear_algebra::dot(const AType &A, const BType &B, const CType &C)

.. doxygenfunction:: einsums::linear_algebra::scale(T scale, AType* A)

.. doxygenfunction:: einsums::linear_algebra::scale_column(size_t col, T scale, AType* A)

.. doxygenfunction:: einsums::linear_algebra::scale_row(size_t row, T scale, AType* A)

BLAS Level 2 Functions
----------------------

BLAS Level 2 routines perform matrix-vector operations, such as matrix-vector multiplication, rank-1
and rank-2 matrix updates, and solution of triangular systems.

.. doxygenfunction:: einsums::linear_algebra::axpy(T alpha, const XType &X, YType *Y)

.. doxygenfunction:: einsums::linear_algebra::axpby(T alpha, const XType &X, T beta, YType *Y)

.. doxygenfunction:: einsums::linear_algebra::gemv(const U alpha, const AType &A, const XType &z, const U beta, YType *y)

.. doxygenfunction:: einsums::linear_algebra::ger(T alpha, const XYType &X, const XYType &Y, AType *A)

BLAS Level 3 Functions
----------------------

BLAS Level 3 routines perform matrix-matrix operations, such as matrix-matrix multiplication, rank-k update, and
solutions of triangular systems.

.. The LONG function signature is needed because there are multiple functions named gemm and it's how to differentiate them in the documentation.
.. doxygenfunction:: einsums::linear_algebra::gemm(const U alpha, const AType &A, const BType &B, const U beta, CType *C)(const T alpha, const AType &A, const BType &B, const T beta, CType *C)

.. doxygenfunction:: einsums::linear_algebra::gemm(const U alpha, const AType &A, const BType &B)

LAPACK Functions
----------------

LAPACK routines can be divided into the following groups according to the operations they perform:

* Routines for solving systems of linear equations, factoring and inverting matrices, and estimating condition numbers.
* Routines for solving least squares problems, eigenvalue and singular value problems, and Sylvester's equations.
* Auxiliary and utility routines used to perform certain subtasks, common low-level computation or related tasks.

LAPACK Linear Equation Computational Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
      These functions assume Fortran, column-major ordering.

.. doxygenfunction:: einsums::linear_algebra::getri(TensorType *A, const std::vector<blas_int> &pivot)

.. doxygenfunction:: einsums::linear_algebra::getrf(TensorType *A, std::vector<blas_int> *pivot)

To be classified
^^^^^^^^^^^^^^^^

.. doxygenfunction:: einsums::linear_algebra::geev

.. doxygenfunction:: einsums::linear_algebra::gesv

.. doxygenfunction:: einsums::linear_algebra::heev

.. doxygenfunction:: einsums::linear_algebra::invert(TensorType *A)

.. Keep the next two: Norm and norm, together if any relocation happens.
.. doxygenenum:: einsums::linear_algebra::Norm
.. doxygenfunction:: einsums::linear_algebra::norm

.. doxygenfunction:: einsums::linear_algebra::pow

.. doxygenfunction:: einsums::linear_algebra::sum_square

.. doxygenfunction:: einsums::linear_algebra::svd

.. doxygenfunction:: einsums::linear_algebra::svd_nullspace

.. doxygenfunction:: einsums::linear_algebra::syev(AType *A, WType *W)

.. doxygenfunction:: einsums::linear_algebra::syev(const AType &A)
