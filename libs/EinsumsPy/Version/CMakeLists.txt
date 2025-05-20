#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(VersionHeaders )

set(VersionSources Export.cpp)

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} Version
  SOURCES ${VersionSources}
  HEADERS ${VersionHeaders}
  DEPENDENCIES Einsums
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME EinsumsPy
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python_headers(${EINSUMS_PYTHON_LIB_NAME}_Version)
