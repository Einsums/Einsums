..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.testing:

********************************
Einsums Python Testing Utilities
********************************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs


In order to make testing Einsums smoother, several utilities have been added to push Einsums to its limits.

.. py:currentmodule:: einsums.core

.. py:class:: BadBuffer

    This class is a minimal implementation of a Python buffer object. However, it allows users
    to set its properties arbitrarily. This allows us to test code paths that are normally not
    allowed if the data in an object is validated properly, thus improving code coverage.

    .. py:method:: __init__([copy])

        Create a new buffer object. Optionally, copy the data from another regular buffer object.

        :param copy: The buffer to copy from.

    .. py:method:: get_ptr()

        Get the underlying pointer to the data.

    .. py:method:: clear_ptr()

        Set the pointer in the buffer to the null pointer. This helps to test code when invalid
        addresses are passed in.

    .. py:method:: get_ndim() -> int

        Get the number of dimensions.

    .. py:method:: set_ndim(ndim: int)

        Set the number of dimensions. This will resize the strides and shape to fit, as well as the data
        to match the new strides and shape.
    
        :param ndim: The new number of dimensions.

    .. py:method:: set_ndim_noresize(ndim: int)

        Set the number of dimensions, but don't resize anything else. This leaves the buffer in an invalid
        state. This is intentional, as this is a tool for testing invalid code paths.

        :param ndim: The new number of dimensions.

    .. py:method:: get_itemsize() -> int

        Get the number of bytes in an item.

    .. py:method:: set_itemszie(itemsize: int)

        Set the item size. This will leave the buffer in an invalid state.

        :param itemsize: The new size of the items.

    .. py:method:: get_format() -> str

        Gets the format string.

    .. py:method:: set_format(fmt: str)

        Set the format string. The recommendation from the developers is to use the string :code:`'X'` as an invalid
        scalar type, and :code:`'ZX'` as an invalid complex type. We also recommend :code:`'y'` and :code:`'Y'` for a signed and 
        unsigned invalid integer type respectively. Code should not check for these types explicitly. Instead, they
        should be handled in general by any error handling code. This will leave the buffer in an invalid state.

        :param str: The new format string.

    .. py:method:: get_dims() -> list[int]

        Get the dimensions or shape of the buffer.

    .. py:method:: set_dims(dims: list[int])

        Set the dimensions of the buffer. This does not update any of the other data, leaving
        the buffer in an invalid state.

    .. py:method:: set_dim(axis: int, dim: int)

        Sets the dimension on a given axis. This does not update any of the other data, leaving
        the buffer in an invalid state.

        :param axis: The axis to update.
        :param dim: The new dimension.

        :raises IndexError: When :code:`axis` is outside of the size of the dimension array.

    .. py:method:: get_strides() -> list[int]

        Get the strides of the buffer in bytes.

    .. py:method:: set_strides(strides: list[int])

        Set the strides of the buffer in bytes. This does not update any of the other data, leaving
        the buffer in an invalid state.

    .. py:method:: set_stride(axis: int, stride: int)

        Sets the stride in bytes on a given axis. This does not update any of the other data, leaving
        the buffer in an invalid state.

        :param axis: The axis to update.
        :param stride: The new stride.

        :raises IndexError: When :code:`axis` is outside of the size of the stride array.

    .. py:method:: change_dims_size(new_size: int)

        Resizes the dimension array. This does not initialize any values added on to the end.
        This will leave the buffer in an invalid state. This also doesn't change any other data.

        :param new_size: The new size for the dimension array.

    .. py:method:: change_strides_size(new_size: int)

        Resizes the stride array. This does not initialize any values added on to the end.
        This will leave the buffer in an invalid state. This also doesn't change any other data.

        :param new_size: The new size for the stride array.

.. py:function:: throw_hip(status: int [, throw_success: bool = False])

    Throws a HIP status exception. If :code:`status == 0`, it will not throw :py:class:`einsums.gpu_except.Success` unless
    :code:`throw_sucess == True`.

    :param status: The status value to use for the exception.
    :param throw_success: Whether to throw an exception when passed the success condition.

.. py:function:: throw_hipblas(status: int [, throw_success: bool = False])

    Throws a hipBlas status exception. If :code:`status == 0`, it will not throw :py:class:`einsums.gpu_except.blasSuccess` unless
    :code:`throw_sucess == True`.

    :param status: The status value to use for the exception.
    :param throw_success: Whether to throw an exception when passed the success condition.

.. py:function:: throw_hipsolver(status: int [, throw_success: bool = False])

    Throws a hipSolver status exception. If :code:`status == 0`, it will not throw :py:class:`einsums.gpu_except.solverSuccess` unless
    :code:`throw_sucess == True`.

    :param status: The status value to use for the exception.
    :param throw_success: Whether to throw an exception when passed the success condition.
