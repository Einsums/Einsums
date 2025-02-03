..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Iterator:

========
Iterator
========

This module defines symbols for iterating over things at compile time.

See the :ref:`API reference <modules_Einsums_Iterator_api>` of this module for more
details.

A few of the symbols may be useful for those using Einsums.

.. cpp:function:: template<auto n, typename F> constexpr void for_sequence(F f)

    This function expands as if it were a loop, but it does its evaluation at compile-time if possible.
    Functions should handle :code:`std::integral_constant`s as their inputs. The types of these constants
    will be the same as the type of :code:`n`. Here is an example of how one might use this.

    .. code::
        
        auto index_lists = std::make_tuple(Indices{index::i, index::j, index::k}, Indices{index::i, index::k, index::j});
        Tensor<double, 3> A = create_random_tensor("A", 10, 10, 10), B = create_random_tensor("B", 10, 10, 10);
        double C = 0;

        for_sequence<2>([&](auto n) { einsum(1.0, Indices{}, &C, 1.0,
            std::get<(size_t) n>(index_lists), A, Indices{index::i, index::j, index::k}, B); });

    This code will go through and compute the einsum with several different layouts of indices. Note that we need to cast the
    argument to the lambda to :code:`size_t` so that it can be consumed by :code:`std::get`.