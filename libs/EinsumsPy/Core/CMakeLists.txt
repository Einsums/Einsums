#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CoreHeaders EinsumsPy/Core/Export.hpp)

set(CoreSources Export.cpp)

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} Core
  SOURCES ${CoreSources}
  HEADERS ${CoreHeaders}
  DEPENDENCIES Einsums
  MODULE_DEPENDENCIES
        CMAKE_SUBDIRS examples tests BASE_LIBNAME EinsumsPy
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python(${EINSUMS_PYTHON_LIB_NAME}_Core ${PYTHON_LIB_TYPE})
