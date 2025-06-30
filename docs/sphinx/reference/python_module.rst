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
    
