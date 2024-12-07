
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Assertion:

=========
Assertion
=========

This module contains several macros for handling assertions.

See the :ref:`API reference <modules_Assertion_api>` of this module for more
details.

Here is a list of public utilities defined in this module.


.. c:macro:: EINSUMS_ASSERT(expr)

    Asserts that the expression is true. Does not print a custom message on failure.

    :param expr: The expression to test.

.. c:macro:: EINSUMS_ASSERT_MSG(expr, msg)

    Asserts that the expression is true. If it is false, then this prints a custom message.

    :param expr: The expression to check.
    :param msg: The message to print on failure.

