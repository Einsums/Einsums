..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Concepts:

========
Concepts
========

This module contains a bunch of concepts to help with compile-time decision making. These are considered
to be internal utilities. However, some concepts define requirements for tensors. Those requirements
will be listed here.

See the :ref:`API reference <modules_Einsums_Concepts_api>` of this module for more
details.


Named Requirements
------------------

Tensor
^^^^^^

Represents a general tensor that is compatible with Einsums calls. It must define the following.

.. code:: C++

    // For some type TensorType,
    TensorType tensor;
    // The following needs to be well defined.
    tensor.full_view_of_underlying(); // Returns a bool
    tensor.name(); // Returns a std::string
    tensor.dim(int);
    tensor.dims();

It can be tested with :cpp:concept:`TensorConcept`.

TypedTensor
^^^^^^^^^^^

Represents a tensor whose stored type is known at compile time.

.. code:: C++

    // For some TensorType, the following must be defined.
    typename TensorType::ValueType;

It can be tested with :cpp:concept:`TypedTensorConcept`.

RankTensor
^^^^^^^^^^

Represents a tensor whose rank is known at compile time.

.. code:: C++

    // For some TensorType, the following must be valid.
    TensorType::Rank;

It can be tested with :cpp:concept:`RankTensorConcept`.

BasicLockable and Lockable
^^^^^^^^^^^^^^^^^^^^^^^^^^

These represent objects, not necessarily tensors, that follow C++'s *Lockable* and *BasicLockable*
named requirements.

For *BasicLockable*, the following must be valid.

.. code:: C++

    // For some type T,
    T var;
    // The following must be valid.
    var.lock();
    var.unlock();

For *Lockable*, the type must be *BasicLockable*, as well as defining the following.

.. code:: C++

    var.try_lock(); // Returns a bool.

TensorView
^^^^^^^^^^


Represents a tensor that sees another tensor's data. It must define the following.

.. code:: C++

    // For some TensorType, the following must be valid.
    typename TensorType::underlying_type;

BasicTensor
^^^^^^^^^^^

Represents a tensor that holds its data in a way that is useable by BLAS or LAPACK. It must define the following.

.. code:: C++

    // For some TensorType,
    TensorType tensor;
    // The following must be valid.
    tensor.data(); // Gives a pointer to the data.
    tensor.stride(int); // Gets the stride along an axis, in elements.
    tensor.strides(); // Gets the list of strides.

CollectedTensor
^^^^^^^^^^^^^^^

Represents a tensor that uses several other tensors for its representation. It must define the following.

.. code:: C++

    // For some TensorType
    typename TensorType::StoredType;

FunctionTensor
^^^^^^^^^^^^^^

Represents a tensor that can be indexed using function call syntax. The tensor must be a *RankTensor*.

.. code:: C++

    // For some TensorType,
    TensorType tensor;
    // The following must be valid.
    tensor(int, int, ...); // The number of indices must match the rank of the tensor.

