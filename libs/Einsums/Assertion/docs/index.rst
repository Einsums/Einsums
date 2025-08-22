..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Assertion:

=========
Assertion
=========

This module contains several macros for handling assertions.

See the :ref:`API reference <modules_Einsums_Assertion_api>` of this module for more
details.

Here is a list of public utilities defined in this module.


.. c:macro:: EINSUMS_ASSERT(expr)

    Asserts that the expression is true. Does not print a custom message on failure.
    If the expression evaluates to false, this will also abort the program.
    This will only evaluate when the program is set to debug. Otherwise it will have no effect.

    :param expr: The expression to test.

    .. versionadded:: 1.0.0

.. c:macro:: EINSUMS_ASSERT_MSG(expr, msg)

    Asserts that the expression is true. If it is false, then this prints a custom message and aborts the execution.
    This will only evaluate when the program is set to debug. Otherwise it will have no effect.

    :param expr: The expression to check.
    :param msg: The message to print on failure.

    .. versionadded:: 1.0.0

-------------
Example Usage
-------------

As an example, suppose we want to make sure that the result of an operation is correct. We can use the assert macros to
make sure that it is correct.

.. code:: C++

    Tensor<1, double> vec = create_random_tensor("vector", 10);
    double out = dot(vec, vec);

    // The dot product of a vector with itself can not be negative.
    // If it is, we are in an invalid state, so we should probably exit.
    EINSUMS_ASSERT_MSG(out >= 0, "The dot product can not be negative! What happened!?");

The :cpp:macro:`EINSUMS_ASSERT` macro is similar, but it doesn't take any debug info. 

.. code:: C++

    // No debug info this time.
    EINSUMS_ASSERT(out >= 0);
