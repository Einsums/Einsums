..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.errors :

***********************
Einsums Internal Errors
***********************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

This page gives the internal errors used by Einsums that are available to Python.

.. py:currentmodule:: einsums.core

.. py:exception:: rank_error

    This is thrown when a tensor is passed to an operation, but the operation can not operate
    on tensors of its rank. For instance, :py:func:`einsums.core.gemm` can only operate on
    rank-2 tensors. If a rank-1 tensor were to be passed in, then this error will be thrown.

    .. versionadded:: 1.0.0

.. py:exception:: dimension_error

    This is thrown when tensors have incompatible dimensions. For instance, :py:func:`einsums.core.geev`
    can only operate on square matrices. If a non-square matrix were to be passed, this error will be thrown.

    .. versionadded:: 1.0.0

.. py:exception:: tensor_compat_error

    This is thrown when tensor arguments have incompatible dimensions with each other. For instance,
    passing two *m* by *n* tensors with no transpositions to :py:func:`einsums.core.gemm` will throw this error.

    .. versionadded:: 1.0.0

.. py:exception:: num_argument_error

    This is thrown when the wrong number of arguments are passed to a function that can take several
    arguments. It has specializations :py:exc:`too_many_args` and :py:exc:`not_enough_args`.

    .. versionadded:: 1.0.0

.. py:exception:: not_enough_args

    This is thrown when a function that can take a variable number of arguments doesn't receive enough
    arguments.

    .. versionadded:: 1.0.0

.. py:exception:: too_many_args

    This is thrown when a function that can take a variable number of arguments receives more than expected.

    .. versionadded:: 1.0.0

.. py:exception:: access_denied

    Indicates that an operation was halted due to access restrictions, such as trying to write to read-only data.

    .. versionadded:: 1.0.0

.. py:exception:: todo_error

    This indicates that a piece of code has not been implemented but will be in the future.

    .. versionadded:: 1.0.0

.. py:exception:: not_implemented

    This indicates that a code path has not been implemented and will likely not be implemented in the future.

    .. versionadded:: 1.0.0

.. py:exception:: bad_logic

    General runtime error indicating that some program logic failed.

    .. versionadded:: 1.0.0

.. py:exception:: uninitialized_error

    Indicates that the code was trying to use uninitialized data.

    .. versionadded:: 1.0.0

.. py:exception:: system_error

    Indicates that an error happened when making a system call.

    .. versionadded:: 1.0.0

.. py:exception:: enum_error

    Indicates that an invalid enumeration value was used.

    .. versionadded:: 1.0.0

.. py:exception:: complex_conversion_error

    Thrown when trying to convert a complex number to a real number. Instead, the input
    data should be transformed into a real value in a way that makes sense for the operation
    being performed. This is often either the magnitude or the real part.

    .. versionadded:: 2.0.0