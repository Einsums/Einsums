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

.. cpp:function:: int initialize(std::function<int(int, char **)> f, int argc, char const *const *argv, InitParams const &params = InitParams());
.. cpp:function:: int initialize(std::function<int()> f, int argc, char const *const *argv, InitParams const &params = InitParams());
.. cpp:function:: int initialize(std::nullptr_t f, int argc, char const *const *argv, InitParams const &params = InitParams());

    Initializes the Einsums framework. It must be called early on otherwise other things may not work. It then executes the
    function that is passed as if it were :code:`main`, passing any command arguments not consumed by Einsums.
    The initialization parameters are passed on to the initialization function.

    :param f: The function to call. If it is the null pointer, then no function will be called, and the initialization routine will exit once finished.
    :param argc: The number of arguments as passed to :code:`main`.
    :param argv: The vector of arguments as passed to :code:`main`.
    :param params: Extra parameters to pass on to the initialization routine.

.. cpp:function:: void finalize()

    Mark that the Einsums library may be torn down.