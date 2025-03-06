..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Profile:

=========
Profiling
=========

This module contains symbols for profiling Einsums.

See the :ref:`API reference <modules_Einsums_Profile_api>` of this module for more
details.

--------------
Public Symbols
--------------

.. cpp:function:: void einsums::profile::report(std::string const &fname, bool append)

    Print the timer report to the file with the given name. The file will be created if it does not exist.
    If :code:`append` is true, then the report will be appended to the end of the file. If not, then the
    file will be cleared, and then filled with the timer report.

    :param fname: The name of the file.
    :param append: Whether to append or overwrite the timer information.
