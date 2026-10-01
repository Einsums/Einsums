..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Print:

=====
Print
=====

This module contains overloads for :code:`fmt::println` that work with Tensors, as well as a few special symbols for
other tasks.

See the :ref:`API reference <modules_Einsums_Print_api>` of this module for more
details.

--------------
Public Symbols
--------------

There are a few symbols that may be useful to users.

.. cpp:class:: template<std::integral IntType> print::ordinal

    This is a wrapper for a value that allows formatting an integer as an ordinal, such as 1st, 2nd, etc.
    Here is an example.

    .. code::
        
        fmt::format("Error with the {} argument", print::ordinal{arg_num});
    
    This might give a string like :code:`Error with the 3rd argument`, if the value passed was 3.

    This class puts the correct ordinal abbreviation after the number based on its value. It can also
    handle negative numbers. To make things easier for users, there are also a few basic operations that
    are defined to allow these to act like normal numbers, such as in-place arithmetic. Ordinals can also
    be specified with the :code:`_th` suffix.

    .. code::
        
        fmt::format("Error with the {} argument", 3_th);

    This will give  :code:`Error with the 3rd argument`.

    .. versionadded:: 1.0.0

    .. versionchanged:: 2.0.0

        Added the literal suffix operator.

