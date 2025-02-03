..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_StringUtil:

==========
StringUtil
==========

This module contains functions for modifying strings.

See the :ref:`API reference <modules_Einsums_StringUtil_api>` of this module for more
details.

Public API
----------

Some functions may be useful to users.

.. cpp:function:: void rtrim(std::string &s)

    Trims whitespace from the end of the string.

    :param s: The string to trim.

.. cpp:function:: void ltrim(std::string &s)

    Trims whitespace from the beginning of the string.

    :param s: The string to trim.

.. cpp:function:: void trim(std::string &s)

    Trims whitespace from the beginning and end of the string.

    :param s: The string to trim.

.. cpp:function:: std::string rtrim_copy(std::string s)
.. cpp:function:: std::string ltrim_copy(std::string s)
.. cpp:function:: std::string trim_copy(std::string s)

    Same as :cpp:func:`ltrim`, :cpp:func:`rtrim`, and :cpp:func:`trim`, but makes a copy
    of the input string, then modifies and returns the copy.