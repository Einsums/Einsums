#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorHeaders EinsumsPy/Tensor/Export.hpp EinsumsPy/Tensor/PyTensor.hpp)

set(TensorSources Export.cpp)

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} Tensor
  SOURCES ${TensorSources}
  HEADERS ${TensorHeaders}
  DEPENDENCIES
    Einsums
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME EinsumsPy
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python(${EINSUMS_PYTHON_LIB_NAME}_Tensor ${PYTHON_LIB_TYPE})
