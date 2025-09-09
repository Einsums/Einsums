..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.gpu_view :

************************
Einsums Python GPU Views
************************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

.. py:currentmodule:: einsums.core

The idea behind the :py:class:`GPUView` is to make transferring data between the host and the GPU much simpler
so that even the most inexperienced programmers can start to benefit from GPU acceleration in their code. These
are designed to be as accessible as possible. This means that they can accept any Python buffer object, not just
the ones defined by Einsums. In fact, much of the testing for these is done on Numpy arrays.

.. py:class:: GPUViewMode

    Base class for the enumerated values for the GPU views. It is created by Pybind11.

    .. versionadded:: 1.0.0

    .. py:property:: name
        
        Gets the name of the enumerated value.

        .. versionadded:: 1.0.0
    
    .. py:property:: value

        Gets the underlying value of the enumerated symbol.

        .. versionadded:: 1.0.0

.. py:data:: COPY
    :type: GPUViewMode

    When creating a view, this indicates that the view should allocate memory on the device and
    copy the data back and forth.

    .. versionadded:: 1.0.0

.. py:data:: MAP
    :type: GPUViewMode

    When creating a view, this indicates that the view should map the host memory into the virtual
    address space on the device. Data will be synchronized automatically, but data validity should 
    not be assumed until all kernels affecting the data have finished.

    .. versionadded:: 1.0.0

.. py:data:: DEVICE_TENSOR
    :type: GPUViewMode

    This mode is used by C++ extensions to create a Python-compatible wrapper around Einsums device
    tensors. Views can not be created with this mode from Python.

    .. versionadded:: 1.0.0

.. py:class:: GPUView

    A view that wraps data to allow for it to be transferred between the host and a device. As an example
    of the use of these views, see below.

    >>> import einsums as ein
    >>> import numpy as np
    >>> plan = ein.core.compile_plan("ij", "ik", "kj")
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> C = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    >>> A_view = ein.core.GPUView(A, ein.core.COPY) # Copy the data into the GPU.
    >>> B_view = ein.core.GPUView(B, ein.core.COPY)
    >>> C_view = ein.core.GPUView(C, ein.core.COPY)
    >>> # At this point, the data is all synchronized, since view creation performs synchronization.
    >>> plan.execute(0, C_view, 1, A_view, B_view)
    >>> # After this call to execute, C has become desynchronized.
    >>> print(C)
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> C_view.update_D2H() # Bring C back into synchronization.
    >>> print(C)
    [[ 30.  36.  42.]
     [ 66.  81.  96.]
     [102. 126. 150.]]

    .. versionadded:: 1.0.0

    .. py:method:: __init__(buffer, mode: GPUViewMode)

        Creates a new :py:class:`GPUView` around the given buffer object with the given mode.

        :param buffer: The buffer object to wrap. Can be anything that implements the Python buffer protocol.
        :param mode: The mode to use when creating this. Can either be :py:data:`MAP` or :py:data:`COPY`.

        .. versionadded:: 1.0.0

    .. py:method:: dims() -> list[int]

        Get the dimensions of the view.

        .. versionadded:: 1.0.0

    .. py:method:: strides() -> list[int]

        Get the strides of the view in bytes.

        .. versionadded:: 1.0.0

    .. py:method:: dim(axis: int) -> int

        Get the dimension of the view along the given axis.

        :param axis: The axis to query.
        :return: The width of the view along that axis.
        :raises IndexError: If the axis is outside of the range of dimensions.

        .. versionadded:: 1.0.0
    
    .. py:method:: stride(axis: int) -> int
        
        Get the stride of the view in bytes along the given axis.

        :param axis: The axis to query.
        :return: The stride of the view in bytes along that axis.
        :raises IndexError: If the axis is outside of the range of dimensions.

        .. versionadded:: 1.0.0

    .. py:method:: fmt_spec() -> str

        Get the format specifier for the view.

        .. versionadded:: 1.0.0

    .. py:method:: update_H2D()

        Synchronize the view's data by moving the host data to the device.

        .. versionadded:: 1.0.0

    .. py:method:: update_D2H()

        Synchronize the view's data by moving the device data to the host.

        .. versionadded:: 1.0.0
    
    .. py:method:: size() -> int
    .. py:method:: __len__() -> int

        Get the number of elements in the view. These two are synonyms.

        .. versionadded:: 1.0.0

    .. py:method:: rank() -> int

        Get the rank of the view, or the number of dimensions.

        .. versionadded:: 1.0.0

    .. py:method:: itemsize() -> int

        Get the number of bytes in each element.

        .. versionadded:: 1.0.0