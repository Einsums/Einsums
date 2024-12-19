..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _function.low_level.linear_algebra:

Low-Level Linear Algebra (BLAS, LAPACK, etc.)
=============================================

.. sectionauthor:: Justin M. Turney

The low-level linear algebra functions are a collection of functions that provide a direct interface to the underlying
linear algebra libraries (e.g., BLAS, LAPACK, etc.). These functions are intended to be used by developers who need to
access the low-level linear algebra routines directly. The functions in this module are not intended to be used by
end-users, as they do not provide a high-level interface to the linear algebra routines.

Before using the low-level linear algebra functions, you should be familiar with the underlying linear algebra libraries.

The low-level linear algebra functions are organized into the following categories:

* BLAS Level 1 Functions
* BLAS Level 2 Functions
* BLAS Level 3 Functions
* LAPACK Functions

Before using any of the functions listed in this module, the Einsums BLAS subsystem needs to be initialized.
This is done with the following function:

.. doxygenfunction:: einsums::blas::initialize

The BLAS subsystem should be finalized when you are done using the low-level linear algebra functions.

.. doxygenfunction:: einsums::blas::finalize

BLAS Level 1 Functions
----------------------

BLAS Level 2 Functions
----------------------

BLAS Level 3 Functions
----------------------

BLAS Level 3 routines perform matrix-matrix operations, such as matrix-matrix multiplication, rank-k update, and
solutions of triangular systems.

The following BLAS Level 3 functions are available:

.. doxygenfunction:: einsums::blas::gemm

LAPACK Functions
----------------

To Be Classified
----------------
