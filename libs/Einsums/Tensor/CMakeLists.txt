#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorHeaders
        Einsums/Tensor/BlockTensor.hpp
        Einsums/Tensor/DiskTensor.hpp
        Einsums/Tensor/H5.hpp
        Einsums/Tensor/Tensor.hpp
        Einsums/Tensor/TensorForward.hpp
        Einsums/Tensor/ArithmeticTensor.hpp
        Einsums/Tensor/FunctionTensor.hpp
        Einsums/Tensor/RuntimeTensor.hpp
)

if(EINSUMS_WITH_GPU_SUPPORT)
      list(APPEND TensorHeaders Einsums/Tensor/DeviceTensor.hpp)
      
      set(TensorGPUModules Einsums_GPUStreams hip::host hip::device)
endif()

set(TensorSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Tensor
  SOURCES ${TensorSources}
  HEADERS ${TensorHeaders}
  DEPENDENCIES Einsums::h5cpp
  MODULE_DEPENDENCIES
        Einsums_Concepts
        Einsums_Config
        Einsums_Errors
        Einsums_Iterator
        Einsums_TensorBase
        Einsums_TypeSupport
        Einsums_Profile
        Einsums_Print
        ${TensorGPUModules}
  CMAKE_SUBDIRS examples tests
)

