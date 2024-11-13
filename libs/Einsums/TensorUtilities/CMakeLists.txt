#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorUtilitiesHeaders
        Einsums/TensorUtilities/CreateIncrementedTensor.hpp
        Einsums/TensorUtilities/CreateRandomDefinite.hpp
        Einsums/TensorUtilities/CreateRandomSemidefinite.hpp
        Einsums/TensorUtilities/CreateRandomTensor.hpp
        Einsums/TensorUtilities/CreateTensorLike.hpp
        Einsums/TensorUtilities/Diagonal.hpp
)

set(TensorUtilitiesSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorUtilities
  GLOBAL_HEADER_GEN ON
  SOURCES ${TensorUtilitiesSources}
  HEADERS ${TensorUtilitiesHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        Einsums_TensorBase Einsums_Tensor Einsums_LinearAlgebra Einsums_Concepts
  CMAKE_SUBDIRS examples tests
)
