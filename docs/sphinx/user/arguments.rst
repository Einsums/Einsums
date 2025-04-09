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

.. option:: --einsums:log-destination [cerr | cout]

    Set whether the logger will log to standard output or standard error.

.. option:: --einsums:no-profiler-report

    Tells Einsums not to output the profiling information.

.. option:: --einsums:profiler-filename

    The name of the file for the profiler output.


==================
Advanced Arguments
==================

.. option:: --einsums:no-install-signal-handlers

    Tells Einsums not to install its custom signal handlers.

.. option:: --einsums:no-attach-debugger

    Tells Einsums not to allow users the ability to attach a debugger when an error is detected.

.. option:: --einsums:no-diagnostics-on-terminate

    When present, Einsums won't print extra diagnostics on termination.

.. option:: --einsums:log-format

    A format string used for the logger output.

.. option:: --einsums:profiler-append

    If present, the profiling information will be appended to the profiling file. Otherwise, the profiling
    file will be overwritten.