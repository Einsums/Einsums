#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorAlgebraHeaders
        Einsums/TensorAlgebra.hpp
        Einsums/TensorAlgebra/Sort.hpp
        Einsums/TensorAlgebra/TensorAlgebra.hpp
        Einsums/TensorAlgebra/Backends/BaseAlgebra.hpp
        Einsums/TensorAlgebra/Backends/BlockAlgebra.hpp
        Einsums/TensorAlgebra/Backends/BlockTileAlgebra.hpp
        Einsums/TensorAlgebra/Backends/Dispatch.hpp
        Einsums/TensorAlgebra/Backends/ElementTransform.hpp
        Einsums/TensorAlgebra/Backends/GenericAlgorithm.hpp
        Einsums/TensorAlgebra/Backends/KhatriRao.hpp
        Einsums/TensorAlgebra/Backends/TileAlgebra.hpp
        Einsums/TensorAlgebra/Backends/Unfold.hpp
        Einsums/TensorAlgebra/Detail/Index.hpp
        Einsums/TensorAlgebra/Detail/Utilities.hpp
)

if (NOT WIN32)
  set(TensorAlgebraSources Sort.cpp)
  set(EXTRA_TENSOR_ALGEBRA_DEPENDENCIES Einsums_HPTT)
endif ()

if (EINSUMS_WITH_GPU_SUPPORT)
    list(APPEND TensorAlgebraHeaders Einsums/TensorAlgebra/Backends/GPUTensorAlgebra.hpp Einsums/TensorAlgebra/Backends/DeviceSort.hpp)
endif ()

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorAlgebra
  SOURCES ${TensorAlgebraSources}
  HEADERS ${TensorAlgebraHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        Einsums_Config
        Einsums_LinearAlgebra
        Einsums_Concepts
        Einsums_Errors
        Einsums_Print
        Einsums_Profile
        Einsums_Tensor
        Einsums_TensorUtilities
        ${EXTRA_TENSOR_ALGEBRA_DEPENDENCIES}
  CMAKE_SUBDIRS examples tests
)
