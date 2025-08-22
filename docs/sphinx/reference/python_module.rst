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

This module contains various facilities for interfacing with the Einsums library.

.. versionadded:: 1.0.0

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
    
