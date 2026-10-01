..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _arguments:

######################
Command Line Arguments
######################

There are several command line arguments that can control the behavior of Einsums. This is the documentation
for these arguments.

===============
Basic Arguments
===============

.. option:: --einsums:log-level <level>

    Set the level to see in the logger. Lower values provide more information. By default, it is set to 
    3 for the release build and 2 for the debug build.

    * 0: Tracing messages. Very verbose.
    * 1: Debugging messages.
    * 2: Information messages
    * 3: Warnings.
    * 4: Errors.
    * 5: Critical errors.

    .. versionadded:: 1.0.0
    .. versionchanged:: 1.1.0
        This option now also sets the HIP log level if Einsums was built with GPU support.

.. option:: --einsums:log-destination [cerr | cout]

    Set whether the logger will log to standard output or standard error.

    .. versionadded:: 1.0.0

.. option:: --einsums:no-profiler-report

    Tells Einsums not to output the profiling information.

    .. versionadded:: 1.0.0

.. option:: --einsums:profiler-filename

    The name of the file for the profiler output.

    .. versionadded:: 1.0.0

.. option:: --einsums:buffer-size

    The amount of memory Einsums is allowed to use. It takes a string containing a number and units.
    The units are either bytes or words. See :cpp:func:`memory_string` for more information.

    .. versionadded:: 1.0.0

.. option:: --einsums:gpu-buffer-size

    The amount of memory Einsums is allowed to use on the GPU. It takes a string containing a number and units.
    The units are either bytes or words. See :cpp:func:`memory_string` for more information.

    .. versionadded:: 1.0.0


==================
Advanced Arguments
==================

.. option:: --einsums:no-install-signal-handlers

    Tells Einsums not to install its custom signal handlers.

    .. versionadded:: 1.0.0

.. option:: --einsums:no-attach-debugger

    Tells Einsums not to allow users the ability to attach a debugger when an error is detected.

    .. versionadded:: 1.0.0

.. option:: --einsums:no-diagnostics-on-terminate

    When present, Einsums won't print extra diagnostics on termination.

    .. versionadded:: 1.0.0

.. option:: --einsums:log-format

    A format string used for the logger output.

    .. versionadded:: 1.0.0

.. option:: --einsums:profiler-append

    If present, the profiling information will be appended to the profiling file. Otherwise, the profiling
    file will be overwritten.

    .. versionadded:: 1.0.0