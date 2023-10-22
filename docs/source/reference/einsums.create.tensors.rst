****************
Creating Tensors
****************

.. sectionauthor:: Justin M. Turney

Functions for creating standard tensors
---------------------------------------

.. doxygenfunction:: einsums::create_tensor(const std::string, Args...)
   :project: Einsums

.. doxygenfunction:: einsums::create_disk_tensor(h5::fd_t&, const std::string, Args...)
   :project: Einsums

.. doxygenfunction:: einsums::create_disk_tensor_like(h5::fd_t&, const Tensor<T, Rank>&)
   :project: Einsums

Functions for create pre-filled tensors
---------------------------------------

.. doxygenfunction:: einsums::create_incremented_tensor(const std::string&, MultiIndex...)
   :project: Einsums

.. doxygenfunction:: einsums::create_random_tensor(const std::string&, MultiIndex...)
   :project: Einsums

.. doxygenfunction:: einsums::create_identity_tensor(const std::string&, MultiIndex...)
   :project: Einsums

.. doxygenfunction:: einsums::create_ones_tensor(const std::string&, MultiIndex...)
   :project: Einsums

.. doxygenfunction:: einsums::create_tensor_like(const TensorType<DataType, Rank>&)
   :project: Einsums

.. doxygenfunction:: einsums::create_tensor_like(const std::string, const TensorType<DataType, Rank>&)
   :project: Einsums

