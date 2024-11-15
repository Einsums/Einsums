..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_LinearAlgebra:

Linear Algebra
==============

.. sectionauthor:: Justin M. Turney

This module contains functions for performing high-level linear algebra operations on tensors.

The functions in this module are designed to be used with the `Tensor` class and its subclasses.

BLAS Level 1 Functions
----------------------

BLAS Level 1 routines perform operations of both addition and reduction on vectors of data.
Typical operations include scaling and dot products.

.. doxygenfunction:: einsums::linear_algebra::dot(AType const &A, BType const &B)
    :project: LinearAlgebra

.. doxygenfunction:: einsums::linear_algebra::dot(AType const &A, BType const &B, CType const &C)
    :project: LinearAlgebra

.. doxygenfunction:: einsums::linear_algebra::scale
    :project: LinearAlgebra

.. doxygenfunction:: einsums::linear_algebra::scale_column
    :project: LinearAlgebra

.. doxygenfunction:: einsums::linear_algebra::scale_row
    :project: LinearAlgebra


See the :ref:`API reference <modules_LinearAlgebra_api>` of this module for more
details.

..
    This doesnt work.

See details of the function in :cpp:func:`einsums::linear_algebra::dot(AType const&, BType const&, CType const&)`.

..
    This does

See details of the function in :cpp:func:`einsums::linear_algebra::dot`.
