
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Profiling:

=========
Profiling
=========

This module contains symbols for profiling Einsums.

See the :ref:`API reference <modules_Profiling_api>` of this module for more
details.

--------------
Public Symbols
--------------

.. cpp:function:: void report()

    Prints the timer report to standard out.

.. cpp:function:: void report(std::string const &fname)

    Print the timer report to the file with the given name. The file will be created if it does not exist.

.. cpp:function:: void report(std::ostream &os)

    Print the timer report to the given output stream.

.. cpp:function:: void report(std::FILE *fp)

    Print the timer report to the given output file pointer.