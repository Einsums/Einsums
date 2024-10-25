..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.runtimetensor:

******************************
Einsums Python Runtime Tensors
******************************

.. py:currentmodule:: einsums.core

These tensors are how the Python and C++ sides of Einsums are able to interact. There
are four tensors, :code:`RuntimeTensorF`, :code:`RuntimeTensorD`, :code:`RuntimeTensorC`, and
:code:`RuntimeTensorZ`, corresponding to single-precision real values, double-precision real values,
single-precision complex values, and double-precision complex values respecively. In order to increase
brevity, these will all be referred to as :py:class:`RuntimeTensorX`, where `X` should be replaced by the 
respective letter. There is also the base class, :py:class:`RuntimeTensor`, though this does not have
much code to itself.

.. py:class:: RuntimeTensor

    Very basic class. Should not be instantiated. It is the superclass for all :py:class:`RuntimeTensorX` and :py:class:`RuntimeTensorViewX` types.

.. py:class:: RuntimeTensorX

    These are the tensor classes that allow the Einsums Python code to interface with the C++ library.
    It has a runtime-computed rank, since compile-time computed ranks are not available in Python. This
    wraps the :cpp:class:`einsums::RuntimeTensor` class with a trampoline class provided by :cpp:class:`einsums::python::PyTensor`.

    .. py:method:: __init__()

        Construct a new empty tensor.

    .. py:method:: __init__(name: str, dims: list[int])
        :noindex:

        Construct a new tensor with a name and dimensions.

        :param name: The name of the tensor.
        :param dims: The dimensions of the tensor. The rank is determined from the length of this argument.

    .. py:method:: __init__(dims: list[int])
        :noindex:

        Construct a new tensor with the given dimensions. Its name will be initialized to a default value.
        Wraps :cpp:func:`einsums::python::PyTensor(const std::vector<size_t> &)`

        :param dims: The dimensions of the tensor. The rank is determined from the length of this argument.

    .. py:method:: __init__(buffer_object)
        :noindex:

        Construct a new tensor from the given buffer object. Its dimensions will be determined from this object,
        and its data will be copied from this object.

        :param buffer_object: The object to copy from. Can only be an object implementing the Python buffer protocol.

    .. py:method:: zero() -> None

        Zero out the data in the tensor. Wraps :cpp:func:`einsums::RuntimeTensor::zero()`.

    .. py:method:: set_all(value) -> None

        Set all values in the tensor to the value passed to the function. Wraps :cpp:func:`einsums::RuntimeTensor::set_all`

        :param value: The value to fill the tensor with.

    .. py:method:: __getitem__(index)

        Get the value at an index using Python's bracket syntax.

        :param index: The index to pass. Can be a single value, a tuple, a slice, or pretty much anything that normally works.
        :return: Return value depends on the index passed. It may be a single value or it may be a :py:class:`einsums.core.RuntimeTensorView` object.

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
        object. If :code:`other` is a single value, it will operate every single element with that
        value. If it is a buffer, then it must have the same dimensions as this tensor, and it
        will then perform the element-wise operation between the elements of the tensor and the buffer.

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

    .. py:method:: to_rank_1_view() -> einsums.core.RuntimeTensorViewX

        Return a view of the tensor where all the elements are in a list. Here is an example.

        .. code:: Python

            >>> A = einsums.utils.create_random_tensor("A", [3, 3])
            >>> print(A)
                Name: A
                    Type: In Core Runtime Tensor
                    Data Type: double
                    Dims{3 3 }
                    Strides{3 1 }

                    (0,  0-2):        0.03651354     0.25669908     0.11172557 

                    (1,  0-2):        0.56452605     0.26229278     0.13112895 

                    (2,  0-2):        0.45176621     0.25069921     0.54104020
            >>> print(A.to_rank_1_view())
                Name: (unnamed view)
                    Type: In Core Runtime Tensor View
                    Data Type: double
                    Dims{9 }
                    Strides{1 }

                    (0):     0.03651354 
                    (1):     0.25669908 
                    (2):     0.11172557 
                    (3):     0.56452605 
                    (4):     0.26229278 
                    (5):     0.13112895 
                    (6):     0.45176621 
                    (7):     0.25069921 
                    (8):     0.54104020
    
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

