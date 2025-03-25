..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_GPUMemory:

=================
Einsums GPUMemory
=================

Contains allocators for GPU-related memory.

See the :ref:`API reference <modules_Einsums_GPUMemory_api>` of this module for more
details.

GPU Allocators
--------------

There are two allocators provided: :cpp:class:`GPUAllocator` and :cpp:class:`MappedAllocator`. The mapped allocator
is easiest to explain. It is a normal C++ allocator that allocates memory that is mapped into the GPU's virtual
address space. Memory operations are synchronized as they happen. This is useful, but can lead to severe performance
hits. The GPU allocator, however, is not a traditional C++ allocator. It can be interacted with in much the same way,
but due to how it works, it is likely not possible to use in a C++ container, though it may be acceptable in some cases.
Instead, this allocator gives a wrapped GPU pointer. These mapped pointers can be treated as normal pointers, with the
exception that references created from them don't make sense, and it is not possible to construct objects of non-arithmetic
types. Complex numbers are fine, though. In addition, `std::memcpy` is overloaded for these pointers to wrap the appropriate
HIP memcpy call. 