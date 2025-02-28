..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Config:

======
Config
======

This module contains several utilities for telling the compiler how to handle things.

See the :ref:`API reference <modules_Einsums_Config_api>` of this module for more
details.

Public Symbols
--------------

Most symbols in this module are private. However, there is one that may be of use to users.

.. c:macro:: EINSUMS_COMPUTE_CODE

    This macro is only defined when Einsums is built with GPU capabilities. It is an easy way
    to determine when GPU support is available.