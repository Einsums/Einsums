..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.logging :

**************
Python Logging
**************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

The logging facilities for Einsums are available in Python.

.. py:currentmodule:: einsums.core


.. py:function:: log(level: int, message: str)

    Logs a message with the given level. The levels are as follows:

    0: Trace messages. These are only shown when running the Debug configuration. These are
    used for putting messages in the code to see which parts are actually being run.

    1: Debug messages. These are only shown when running the Debug configuration. These are
    for giving information about the values of variables or other program state information.

    2: Informational messages. These are things that are useful to know but don't necessarily
    warrant extra attention.

    3: Warnings. These indicate that the program may be in a bad state, but it is recoverable.
    
    4: Errors. These indicate that the program is in a bad state, but it is not recoverable.

    5: Critical messages. For when error messages aren't severe enough.

    Each of these levels also has a specialized call.

    .. versionadded:: 1.1.0

.. py:function:: log_trace(message: str)

    Logs a trace message. Only shows up when Einsums is built in the Debug configuration.

    .. versionadded:: 1.1.0

.. py:function:: log_debug(message: str)

    Logs a debug message. Only shows up when Einsums is built in the Debug configuration.

    .. versionadded:: 1.1.0

.. py:function:: log_info(message: str)

    Logs an informational message.

    .. versionadded:: 1.1.0

.. py:function:: log_warn(message: str)

    Logs a warning.

    .. versionadded:: 1.1.0

.. py:function:: log_error(message: str)

    Logs an error.

    .. versionadded:: 1.1.0

.. py:function:: log_critical(message: str)

    Logs a critical message.

    .. versionadded:: 1.1.0