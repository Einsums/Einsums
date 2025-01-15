#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LinearAlgebraHeaders
    Einsums/LinearAlgebra.hpp Einsums/LinearAlgebra/Base.hpp Einsums/LinearAlgebra/BlockTensor.hpp
    Einsums/LinearAlgebra/TiledTensor.hpp Einsums/LinearAlgebra/Unoptimized.hpp
)

set(LinearAlgebraSources LinearAlgebra.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND LinearAlgebraHeaders Einsums/LinearAlgebra/GPULinearAlgebra.hpp)
  list(APPEND LinearAlgebraSources GPULinearAlgebra.hip)
  set(LinAlgGPUModules Einsums_GPUStreams roc::hipblas roc::hipsolver hip::host hip::device)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums LinearAlgebra
  SOURCES ${LinearAlgebraSources}
  HEADERS ${LinearAlgebraHeaders}
  DEPENDENCIES ${LinAlgGPUModules}
  MODULE_DEPENDENCIES
    Einsums_Assertion
    Einsums_Config
    Einsums_BLAS
    Einsums_Concepts
    Einsums_Errors
    Einsums_Print
    Einsums_Tensor
    Einsums_TensorUtilities
    Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
