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

BLAS Level 2 Routines
---------------------

.. For some reason gemv does not want to be documented
.. doxygenfunction:: einsums::linear_algebra::gemv(const T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const T beta, YType<T, XYRank> *y)
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

.. doxygenfunction:: einsums::linear_algebra::heev
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::sum_square
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::syev(AType<T, ARank> *A, WType<T, WRank> *W)
    :project: Einsums

.. doxygenfunction:: einsums::linear_algebra::syev(const AType<T, ARank> &A)
    :project: Einsums
