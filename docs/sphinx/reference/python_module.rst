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

.. py:module:: einsums

This module contains various facilities for interfacing with the Einsums library.

.. py:function:: set_finalize_arg(out: str | bool) -> None

    Sets the argument to pass to einsums.core.finalize. By default, the argument is False, which tells it not to print the
    timing file. If the argument is True, the timing information will be printed to standard output. If
    it is a file name, that will be used as the timing file, and the timing will be printed there on exit. Calling this
    function with a string, then again with `True` or `False` will override the behavior. For instance,
  
    >>> einsums.set_finalize_arg("timings.txt")
    >>> exit()

    will create a file called :code:`timings.txt` and put the timing information there. However, the following will have different behavior.

    >>> einsums.set_finalize_arg("timings.txt")
    >>> einsums.set_finalize_arg(True) # Overrides the previous.
    >>> exit()

    This will not create the file and will instead print timing information to standard output.

    :param out: The output object. If `out` is a string, then this specifies the output file name. If it is `True`, then
        Einsums will output timing information to standard output. If it is false, then no timing information will be emitted.

.. py:module:: einsums.core

This module contains the C++ wrappings. It is documented in :ref:`einsums.core`.

.. py:module:: einsums.utils

This module contains several helpful classes and functions. It is documented in :ref:`einsums.utils`.

.. py:module:: einsums.gpu_except

This module contains all of the possible exceptions that can be thrown by HIP, hipBlas, and hipSolver.
There are hundreds of these, so it is pretty big. It is its own module so as to avoid cluttering the 
namespaces of the other modules.

.. toctree::
    :maxdepth: 3

    einsums.core
    einsums.utils
    einsums.gpu_except
    
