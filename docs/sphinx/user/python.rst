..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

*****************
Einsums in Python
*****************

.. sectionauthor:: Connor Briggs

As mentioned in the introduction, Einsums is compatible with Python.
Here, we will describe the basics for setting up a computation in Python, as well as some
useful functions.

Running Einsums
-----------------

In order to do a calculation using Einsums, we need to import it.

.. code:: Python

    >>> import einsums as ein

From here, we can now define some data. The core Einsums functions will work with our own
tensor classes, their children, and anything that implements the Python buffer protocol,
including :code:`numpy.ndarray`. To start, we need to compile a plan. This is where Einsums
will decide on any optimizations that can be done, such as restructuring the tensor to use in
a call to BLAS. Here is an example for a matrix multiplication.

>>> import einsums as ein
>>> import numpy as np
>>> plan = ein.core.compile_plan("ij", "ik", "kj")
>>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> C = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
>>> plan.execute(0, C, 1, A, B) # Compute C = 0 * C + 1 * A @ B
>>> print(C)
[[ 30.  36.  42.]
 [ 66.  81.  96.]
 [102. 126. 150.]]

And that's it. Now you know the basics of running Einsums.

GPU Acceleration
----------------

One of the big features of Einsums is its GPU acceleration using AMD's HIP language. In order to make
this feature accessible to even the most novice of users, we have provided an interface that should be
very easy to use. This is done using the :code:`einsums.core.GPUView` class. Essentially, objects of this
class wrap a buffer object and handles data transfers from the CPU to the GPU and back. We will start with
the previous example for the definitions, and wrap each of these in a :code:`GPUView` before execution.

.. code:: Python
    
    >>> A_view = ein.core.GPUView(A, ein.core.COPY) # Copy the data into the GPU.
    >>> B_view = ein.core.GPUView(B, ein.core.COPY)
    >>> C_view = ein.core.GPUView(C, ein.core.COPY)
    >>> plan.execute(0, C_view, 1, A_view, B_view)
    >>> C_view.update_D2H() # Copy the data from the GPU back to the host.

As we can see, there is very little difference between the CPU and GPU. One very important thing to note
is that in order to reduce the amount of memory operations, memory is NOT AUTOMATICALLY SYNCHRONIZED after
calls to :code:`execute` when the buffers are wrapped using the :code:`einsums.core.COPY` mode. Thus,
in order to maintain data validity, you should call :code:`einsums.core.GPUView.update_H2D()` anytime you
modify a buffer on the host side before a call to :code:`einsums.core.execute`, and you should call
:code:`einsums.core.GPUView.update_D2H()` anytime you modify a buffer on the GPU before you access it on the
host. If we modify the example above with print statements, we can see this in more detail.

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

This does not need to be done if the data has been wrapped using the :code:`einsums.core.MAP` mode.
However, this mode tends to be very slow, since data is constantly being synchronized whenever the GPU
encounters a cache miss, which will happen very often for large tensors.

>>> import einsums as ein
>>> import numpy as np
>>> plan = ein.core.compile_plan("ij", "ik", "kj")
>>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> C = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
>>> A_view = ein.core.GPUView(A, ein.core.MAP) # Map the data into the GPU's virtual memory
>>> B_view = ein.core.GPUView(B, ein.core.MAP)
>>> C_view = ein.core.GPUView(C, ein.core.MAP)
>>> # At this point, the data is all synchronized, since view creation performs synchronization.
>>> plan.execute(0, C_view, 1, A_view, B_view)
>>> # Since C is mapped into virtual memory, C will already be synchronized.
>>> print(C)
[[ 30.  36.  42.]
 [ 66.  81.  96.]
 [102. 126. 150.]]
>>> C_view.update_D2H() # Does nothing when wrapped with ein.core.MAP.
>>> print(C)
[[ 30.  36.  42.]
 [ 66.  81.  96.]
 [102. 126. 150.]]

Creating Tensors
----------------

As we have seen, Einsums is compatible with any buffer object, including Numpy arrays. However, the C++ side
of Einsums is not compatible with these Python objects. To aid in this transition, there are two sets of tensors
that have been made available: :code:`einsums.core.RuntimeTensorX` and :code:`einsums.core.RuntimeTensorViewX`,
where :code:`X` stands for :code:`F` for 32-bit single-precision floating point values such as :code:`numpy.single`,
:code:`D` for 64-bit double-precision floating point values such as Python's :code:`float` or :code:`numpy.double`,
:code:`C` for 64-bit single-precision complex values such as :code:`numpy.complex64`, or
:code:`Z` for 128-bit double-precision complex values such as Python's :code:`complex` or :code:`numpy.complex128`.
Extended precision is not available, since it is not available for Windows or for AMD graphics cards. Half-precision
is also not available due to lack of support in the C++ standard. For more documentation on the methods defined within
these tensors, see the relevant documents. There are also types called :code:`einsums.core.RuntimeTensor` and :code:`einsums.core.RuntimeTensorView`.
These are the base classes for all of these other tensors, but they have no code of their own. They are provided for things like
:code:`isinstance(A, einsums.core.RuntimeTensor)` to check if something is a runtime tensor without specifying its type.
It should be noted that :code:`einsums.core.RuntimeTensorView` is a child of :code:`einsums.core.RuntimeTensor`, so
all tensor views are also instance of :code:`einsums.core.RuntimeTensor`. However, tensor views are not instances of the
runtime tensors associated with their type. The following example will show all of this behavior.

>>> import einsums as ein
>>> plan = ein.core.compile_plan("ij", "ik", "kj")
>>> A = ein.utils.create_random_tensor("A", [3, 3])
>>> B = ein.utils.create_random_tensor("B", [3, 3])
>>> C = ein.utils.create_tensor("C", [3, 3], dtype=float)
>>> plan.execute(0, C, 1, A, B)
>>> print(C) # Since A and B are random, this is just an example.
Name: C
    Type: In Core Runtime Tensor
    Data Type: double
    Dims{3 3 }
    Strides{3 1 }
<BLANKLINE>
    (0,  0-2):        0.52218486     0.20413352     0.18708155 
<BLANKLINE>
    (1,  0-2):        0.97491459     0.48250664     0.56360688 
<BLANKLINE>
    (2,  0-2):        0.66677923     0.38629482     0.38812904
>>> # Checking instances.
>>> A_view = A[0:2, 0:2]
>>> print(type(A))
<class 'einsums.core.RuntimeTensorD'>
>>> print(type(A_view))
<class 'einsums.core.RuntimeTensorViewD'>
>>> print(isinstance(A, ein.core.RuntimeTensorD)) # A is a RuntimeTensorD.
True
>>> print(isinstance(A, ein.core.RuntimeTensor)) # A is a RuntimeTensorD, so also a RuntimeTensor.
True
>>> print(isinstance(A, ein.core.RuntimeTensorF)) # A is a RuntimeTensorD, not a RuntimeTensorF.
False
>>> print(isinstance(A, ein.core.RuntimeTensorView)) # A is not a view.
False
>>> print(isinstance(A_view, ein.core.RuntimeTensorView)) # A_view is a RuntimeTensorViewD, so also a RuntimeTensorView.
True
>>> print(isinstance(A_view, ein.core.RuntimeTensorViewD)) # A_view is a RuntimeTensorViewD.
True
>>> print(isinstance(A_view, ein.core.RuntimeTensor)) # RuntimeTensorView is a subclass of RuntimeTensor
True
>>> print(isinstance(A_view, ein.core.RuntimeTensorD)) # A is a view, not a tensor.
False

