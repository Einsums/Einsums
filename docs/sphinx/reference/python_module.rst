..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _python_module:

*********************
Einsums Python Module
*********************

This is the reference documentation for the Einsums Python module. It contains many sub-modules as well.

einsums
-------

.. py:currentmodule:: einsums
    
.. versionadded:: 1.0.0

This module contains various facilities for interfacing with the Einsums library. It also provides the following functions.

.. py:function:: log(level: int, msg: str, always_print: bool = False, stack_info: typing.Optional[inspect.FrameInfo]=None)
    Log a message with a given level of urgency. Level 0 is for trace statements, 1 is for debugging statements,
    2 is for general information, 3 is for recoverable warnings, 4 is for unrecoverable errors, 5 is for critical
    errors.
    
    .. versionadded:: 1.1.6

    :param level: The level to log at.
    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.
    :param stack_info: A ``FrameInfo`` object or ``None``, which tells the logger the function that raised the message.
    If none, then the logger will use the frame info of the function that called it.

.. py:function:: log_trace(msg: str, always_print: bool = False)
    Log a trace message. These have the lowest urgency. These are intended to indicate which code path is being taken
    or that a certain procedure has completed successfully.
    
    .. versionadded:: 1.1.6

    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.
    
    .. versionadded:: 1.1.6

.. py:function:: log_debug(msg: str, always_print: bool = False)
    Log a debugging message. These should give information that is helpful for developers trying to debug your system.
    
    .. versionadded:: 1.1.6

    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.

.. py:function:: log_info(msg: str, always_print: bool = False)
    Log an informational message. These should give information about the execution that is relevant to a general user.
    
    .. versionadded:: 1.1.6

    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.

.. py:function:: log_warn(msg: str, always_print: bool = False)
    Log a warning message. These indicate that something has gone wrong, but the issue is recoverable.
    
    .. versionadded:: 1.1.6

    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.

.. py:function:: log_error(msg: str, always_print: bool = False)
    Log an error message. These indicate that something has gone wrong in an unrecoverable way. They are often
    followed by an exception being thrown.
    
    .. versionadded:: 1.1.6

    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.

.. py:function:: log_critical(msg: str, always_print: bool = False)
    Log a critical error message. These have the highest urgency. They indicate that something has gone wrong
    in an unrecoverable way, and often with implications outside of the scope of the program execution.
    
    .. versionadded:: 1.1.6

    :param msg: The message to print.
    :param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
    global log level limit. If true, the global log level limit is ignored.

If you're having trouble importing Einsums, there is an enviroment variable, :envvar:`EINSUMS_DEBUG_IMPORT`, which
turns on debug statements during the import process.

.. envvar:: EINSUMS_DEBUG_IMPORT

    If this is set to a truthy value, debugging statements will be turned on while importing Einsums. If not set,
    or set to a falsy value, debugging statements will be turned off. If set to an unrecognized value, the default
    action will be taken, which will usually be no debugging statements. Truthy values are considered to be 
    :code:`["on", "yes", true", "1"]`, while falsy values are considered to be :code:`["off", "no", "false", "0"]`.
    These checks are case-insensitive, so :code:`"on"`, :code:`"On"`, and :code:`"ON"` are all considered truthy.
    
    .. versionadded:: 1.1.6


.. py:function:: initialize()

    Initializes the Einsums library. This is an internal function that is called automatically. 
    It does not need to be called directly, and instead it is called when you import :code:`einsums`.

    .. versionadded:: 1.0.0
    .. versionremoved:: 2.0.0
        Some users were using this in their code. Its name was changed to make it clear that it is called internally.

.. py:function:: __initialize()

    Initializes the Einsums library. This is an internal function that is called automatically. 
    It does not need to be called directly, and instead it is called when you import :code:`einsums`.

    .. versionadded:: 2.0.0

einsums.core
------------

This module contains the C++ wrappings. It is documented in :ref:`einsums.core`.

.. versionadded:: 1.0.0

einsums.utils
-------------

This module contains several helpful classes and functions. It is documented in :ref:`einsums.utils`.

.. versionadded:: 1.0.0

einsums.gpu_except
------------------

This module contains all of the possible exceptions that can be thrown by HIP, hipBlas, and hipSolver.
There are hundreds of these, so it is pretty big. It is its own module so as to avoid cluttering the 
namespaces of the other modules.

.. versionadded:: 1.0.0

.. toctree::
    :maxdepth: 3

    einsums.core
    einsums.utils
    einsums.gpu_except
    
