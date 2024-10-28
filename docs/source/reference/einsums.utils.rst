..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.utils :

*******************
Einsums Utilities Module
*******************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

.. py:currentmodule:: einsums.utils

This module contains extra utilities written in Python for the Einsums library.

.. py:class:: TensorIndices(other[, reverse : bool = False])

    This is an iterator that goes through all possible combinations of indices for a tensor.

    :param other: The tensor to use to generate index combinations, or a :py:class:`TensorIndices` object to copy.
    :param reverse: If true, walk through the indices in reverse. Otherwise, walk forward.

    .. py:method:: __iter__()

        Return the iterator for this iterator (itself).

        :return: The object itself.

    .. py:method:: __reversed__()

        Return the iterator that goes in reverse.

        :return: The iterator that goes from the current index back to the beginning.

    .. py:method:: __next__()

        Get the next index and update the internal state.

        :return: The next index.
        :raises StopIteration: Raises this if there are no more indices.

.. py:function:: create_tensor(*args [, dtype = float]) -> einsums.core.RuntimeTensor

    Create a new tensor with the given data type. The arguments will be passed to the proper
    constructor.

    :param args: Arguments to pass to the constructor.
    :param dtype: The data type of the tensor.

    :return: A tensor that holds the requested type.

.. py:function:: create_random_tensor(name: str, dims: list[int], dtype = float, random_func = random.random)

    Creates a tensor with the given name, dimensions, and data type, then fills that tensor with random data.
    An optional function for specifying the random numbers can be given as well. By default, this is the 
    :py:func:`random.random()` function, which gives a uniformly distributed random number on the range
    :math:`[0, 1)` for real values, and complex values :math:`a + b i` where :math:`a` and :math:`b` are
    each uniformly distributed on the range :math:`[0, 1)`.
    
    If the random function returns real values and
    a complex tensor is requested, then the real and imaginary parts will each be distributed according to
    that random distribution. If the random function returns complex values and a real tensor is requested,
    then the values of the tensor will be taken from the real part of the random values.

    :param name: The name of the tensor.
    :param dims: The dimensions of the tensor. The rank will be determined from these dimensions.
    :param dtype: The type of data to store.
    :param random_func: The function that will give the random values.
    :return: A tensor with the specified name, dimensions, and data type that has been filled with random data.