..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.runtimetensorviews:

***********************************
Einsums Python Runtime Tensor Views
***********************************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

.. py:currentmodule:: einsums.core

These tensor views are how the Python and C++ sides of Einsums are able to interact. There
are four tensor views, :code:`RuntimeTensorViewF`, :code:`RuntimeTensorViewD`, :code:`RuntimeTensorViewC`, and
:code:`RuntimeTensorViewZ`, corresponding to single-precision real values, double-precision real values,
single-precision complex values, and double-precision complex values respecively. In order to increase
brevity, these will all be referred to as :py:class:`RuntimeTensorViewX`, where `X` should be replaced by the 
respective letter. There is also the base class, :py:class:`RuntimeTensorView`, though this does not have
much code to itself. These views share data with the respective :py:class:`RuntimeTensorX` such that a change in one
is reflected in the other.

.. py:class:: RuntimeTensorView

    Very basic class. Should not be instantiated. It is the superclass for all :py:class:`RuntimeTensorViewX` types.

.. py:class:: RuntimeTensorViewX

    These are the tensor view classes that allow the Einsums Python code to interface with the C++ library.
    It has a runtime-computed rank, since compile-time computed ranks are not available in Python. This
    wraps the :cpp:class:`einsums::RuntimeTensorView` class with a trampoline class provided by :cpp:class:`einsums::python::PyTensorView`.
    The data contained in these views is shared with a tensor so that changes in either are reflected in both.

    .. py:method:: __init__()

        Construct a new empty tensor view.

    .. py:method:: __init__(tensor [, dims])
        :noindex:

        Create a view of a tensor. The tensor can be a :py:class:`einsums.core.RuntimeTensorX` of the same
        underyling type, a :py:class:`einsums.core.RuntimeTensorViewX` of the same underlying type, or a buffer
        object of the same underlying type. The :code:`dims` argument indicates the dimensions of the view.

        :param tensor: The tensor to view.
        :param dims: The dimensions of the view.

    .. py:method:: zero() -> None

        Zero out the data in the tensor. Wraps :cpp:func:`einsums::RuntimeTensorView::zero()`.

    .. py:method:: set_all(value) -> None

        Set all values in the tensor to the value passed to the function. Wraps :cpp:func:`einsums::RuntimeTensorView::set_all`

        :param value: The value to fill the tensor with.

    .. py:method:: __getitem__(index)

        Get the value at an index using Python's bracket syntax.

        :param index: The index to pass. Can be a single value, a tuple, a slice, or pretty much anything that normally works.
        :return: Return value depends on the index passed. It may be a single value or it may be a :code:`einsums.core.RuntimeTensorView` object.

    .. py:method:: __setitem__(key, value)

        Similar to :py:meth:`__getitem__`, it can take pretty much anything that will normally work for the key. 
        For the value, a single value is always accepted. If the key creates a view, this will fill the view with
        the single value. If the key is a single value, it will only set that value. Otherwise, if the
        value is a buffer object, including a tensor or tensor view, the key must refer to a view with the same
        dimensions as that buffer object. It will then copy that object into the view.

        :param key: Which item or items to set.
        :param value: The value or buffer of values to set that key to.

    .. py:method:: __imul__(other)
    .. py:method:: __itruediv__(other)
    .. py:method:: __iadd__(other)
    .. py:method:: __isub__(other)

        In-place arithmetic operations. These can accept either a single value or a buffer
        object. If `other` is a single value, it will operate every single element with that
        value. If it is a buffer, then it must have the same dimensions as this tensor, and it
        will then perform the element-wise operation between the elements of the tensor and the buffer.

        :param other: The object to operate with.

    .. py:method:: __mul__(other)
    .. py:method:: __truediv__(other)
    .. py:method:: __add__(other)
    .. py:method:: __sub__(other)
    .. py:method:: __rmul__(other)
    .. py:method:: __rtruediv__(other)
    .. py:method:: __radd__(other)
    .. py:method:: __rsub__(other)

        Out-of-place arithmetic operators. These can accept either a single value or a buffer object.
        If :code:`other` is a single value, it will operate every single element with that
        value. If it is a buffer, then it must have the same dimensions as this tensor, and it
        will then perform the element-wise operation between the elements of the tensor and the buffer.
        These will create a new tensor before operating and will return that new tensor.

        :param other: The object to operate with.

    .. py:method:: assign(buffer)

        Copy the buffer into this tensor. The tensor will resize and reshape to fit the buffer.

        :param buffer: The buffer object to assign from.

    .. py:method:: dim(axis: int) -> int

        Get the dimension along the given axis.

        :param axis: The axis whose dimension should be found.

    .. py:method:: dims() -> list[int]

        Get the dimensions of the tensor.

    .. py:method:: stride(axis: int) -> int

        Get the stride in elements along the given axis.

        :param axis: The axis whos stride should be found.

    .. py:method:: strides() -> list[int]

        Get the strides of the tensor, in elements.
    
    .. py:method:: get_name() -> str

        Get the name of the tensor.

    .. py:method:: set_name(name: str)

        Set the name of the tensor.

        :param name: The new name of the tensor.

    .. py:property:: name

        Python property wrapping :py:meth:`get_name` and :py:meth:`set_name`.

    .. py:method:: size() -> int
    .. py:method:: __len__() -> int

        Get the number of elements in the tensor. :code:`size` and :code:`__len__` are synonyms of each other.

        :return: The number of elements in the tensor.

    .. py:method:: __iter__() -> einsums.core.PyTensorIteratorX

        Get an iterator that iterates over the elements in the tensor.

        :return: An iterator that will iterate over the elements.

    .. py:method:: __reversed__() -> einsums.core.PyTensorIteratorX

        Get an iterator that iterates over the elements in the tensor in reverse.

        :return: An iterator that will iterate over the elements in reverse.

    .. py:method:: rank() -> int

        Get the rank of the tensor, or the number of dimensions.

        :return: The rank of the tensor.

    .. py:method:: __copy__()
    .. py:method:: __deepcopy__()
    .. py:method:: copy()
    .. py:method:: deepcopy()

        Create a copy of the tensor. These are all synonyms of each other.

        :return: A copy of the tensor.

    .. py:method:: __str__() -> str

        Return a string representation of the tensor.

