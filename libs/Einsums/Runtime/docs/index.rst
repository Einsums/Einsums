..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Runtime:

===============
Einsums Runtime
===============

This module contains several runtime utilities.

See the :ref:`API reference <modules_Einsums_Runtime_api>` of this module for more
details.

----------
Public API
----------

.. cpp:function:: int initialize()

    Initializes the state of the Einsums library. It must be called before certain things are available.

.. cpp:function:: void finalize(const char *file_name)
.. cpp:function:: void finalize(const std::string &file_name)
.. cpp:function:: void finalize(FILE *file_pointer)

    Tear down the state of the Einsums library and print timing information to the specified file.
    In the prototypes with strings, the string specifies the file name and path, relative to the
    current directory. In the prototype with the file pointer, this will print the information to
    the file pointer. It must be a writable or appendable file pointer.

.. cpp:function:: void finalize(std::ostream &out)

    Tear down the state of the Einsums library and put timing information in the given output
    stream. This may be any output stream, including file and string streams.

.. cpp:function:: void finalize(bool timer_report = false)

    Tear down the state of the Einsums library. If passed :code:`true`, then the timing report
    will be printed to standard output.