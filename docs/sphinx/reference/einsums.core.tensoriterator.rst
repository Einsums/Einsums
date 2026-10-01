..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.tensoriterator :

************************
Runtime Tensor Iterators
************************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

.. py:currentmodule:: einsums.core

This tensor iterator allows Python programs to iterate over the values of a tensor. In the same way
that the :py:class:`RuntimeTensorX` is defined, these iterators are defined here as :py:class:`PyTensorIteratorX`,
where :code:`X` is either :code:`F` for single-precision real values, :code:`D` is for double-precision real values,
:code:`C` is for single-precision complex values, and :code:`Z` is for double-precision complex values.

.. py:class:: PyTensorIteratorX

    Iterator for tensors and tensor views.

    .. versionadded:: 1.0.0

    .. py:method:: __next__()

        Get the next value in the tensor.

        :return: The next value in the tensor.
        :raises StopIteration: Raises this when there are no more values to give.

        .. versionadded:: 1.0.0

    .. py:method:: __iter__()

        Get the iterator for this object.

        :return: Returns a copy of the object.

        .. versionadded:: 1.0.0

    .. py:method:: __reversed__()

        Get the reverse iterators starting from the current position.

        :return: The reversed iterator.

        .. versionadded:: 1.0.0

    .. py:method:: reversed() -> bool

        Gives whether the iterator goes forward or backwards.

        :return: :code:`True` if the iterator is going in reverse, :code:`False` if it is going forward.

        .. versionadded:: 1.0.0