..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _function.tensor.creation:

Tensor creation functions
=========================

.. sectionauthor:: Justin M. Turney

Functions for creating standard tensors
---------------------------------------

In Einsums, there are two-basic types of tensors: :cpp:class:`einsums::Tensor` and :cpp:class:`einsums::DiskTensor`.
Tensor is an in-memory variant of a tensor and DiskTensor is an on-disk variant of a tensor. DiskTensor
can and will use Tensors.

.. doxygenfunction:: einsums::create_tensor(const std::string, Args...)

.. doxygenfunction:: einsums::create_disk_tensor(h5::fd_t&, const std::string, Args...)

.. doxygenfunction:: einsums::create_disk_tensor_like(h5::fd_t&, const Tensor<T, Rank>&)

Functions for creating pre-filled tensors
-----------------------------------------

.. doxygenfunction:: einsums::create_incremented_tensor(const std::string&, MultiIndex...)

.. doxygenfunction:: einsums::create_random_tensor(const std::string&, MultiIndex...)

.. doxygenfunction:: einsums::create_identity_tensor(const std::string&, MultiIndex...)

.. doxygenfunction:: einsums::create_ones_tensor(const std::string&, MultiIndex...)

.. doxygenfunction:: einsums::create_tensor_like(const TensorType<DataType, Rank>&)

.. doxygenfunction:: einsums::create_tensor_like(const std::string, const TensorType<DataType, Rank>&)

Additional functions for creating tensors
-----------------------------------------

.. doxygenfunction:: einsums::arange(T start, T stop, T step)
.. doxygenfunction:: einsums::arange(T stop)
