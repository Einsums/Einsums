#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TestUtilsHeaders EinsumsPy/TestUtils/Export.hpp EinsumsPy/TestUtils/BadBuffer.hpp)

set(TestUtilsSources Export.cpp BadBuffer.cpp)

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} TestUtils
  SOURCES ${TestUtilsSources}
  HEADERS ${TestUtilsHeaders}
  DEPENDENCIES Einsums
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME EinsumsPy
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python(${EINSUMS_PYTHON_LIB_NAME}_TestUtils ${PYTHON_LIB_TYPE})
