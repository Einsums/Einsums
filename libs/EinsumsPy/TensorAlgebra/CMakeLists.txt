#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorAlgebraHeaders EinsumsPy/TensorAlgebra/PyTensorAlgebra.hpp)

set(TensorAlgebraSources Export.cpp PyTensorAlgebra.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND TensorAlgebraSources PyTensorAlgebra.hip)
  set(TensorAlgebraGPUDeps hip::host hip::device roc::hipblas roc::hipsolver)
  set(TensorAlgebraGPUModules ${EINSUMS_PYTHON_LIB_NAME}_GPU)
endif()

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} TensorAlgebra
  SOURCES ${TensorAlgebraSources}
  HEADERS ${TensorAlgebraHeaders}
  DEPENDENCIES Einsums ${TensorAlgebraGPUDeps}
  MODULE_DEPENDENCIES ${EINSUMS_PYTHON_LIB_NAME}_Tensor ${EINSUMS_PYTHON_LIB_NAME}_LinearAlgebra
                      ${TensorAlgebraGPUModules}
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME EinsumsPy
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python_headers(${EINSUMS_PYTHON_LIB_NAME}_TensorAlgebra)
