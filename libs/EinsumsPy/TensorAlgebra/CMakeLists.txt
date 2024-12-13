#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorAlgebraHeaders EinsumsPy/TensorAlgebra/Export.hpp EinsumsPy/TensorAlgebra/PyTensorAlgebra.hpp)

set(TensorAlgebraSources Export.cpp PyTensorAlgebra.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND TensorAlgebraSources PyTensorAlgebra.hip)
  set(TensorAlgebraGPUDeps hip::host hip::device roc::hipblas roc::hipsolver)
  set(TensorAlgebraGPUModules ${EINSUMS_PYTHON_LIB_NAME}_GPU)
endif()

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} TensorAlgebra
  GLOBAL_HEADER_GEN ON
  SOURCES ${TensorAlgebraSources}
  HEADERS ${TensorAlgebraHeaders}
  DEPENDENCIES Einsums ${TensorAlgebraGPUDeps}
  MODULE_DEPENDENCIES
    ${EINSUMS_PYTHON_LIB_NAME}_Tensor
    ${TensorAlgebraGPUModules}
  CMAKE_SUBDIRS examples tests
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python(${EINSUMS_PYTHON_LIB_NAME}_TensorAlgebra ${PYTHON_LIB_TYPE})
