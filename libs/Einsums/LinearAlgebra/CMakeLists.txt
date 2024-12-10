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

include(Einsums_AddModule)
einsums_add_module(
  Einsums LinearAlgebra
  GLOBAL_HEADER_GEN OFF
  SOURCES ${LinearAlgebraSources}
  HEADERS ${LinearAlgebraHeaders}
  DEPENDENCIES
        MODULE_DEPENDENCIES
        Einsums_Assertion
        Einsums_Config
        Einsums_BLAS
        Einsums_Concepts
        Einsums_Errors
        Einsums_Print
        Einsums_Tensor
        Einsums_TensorUtilities
  CMAKE_SUBDIRS examples tests
)
