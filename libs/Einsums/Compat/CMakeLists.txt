#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CompatHeaders Einsums/Compat/FmtAlternatives.hpp Einsums/Compat/PosixAlternatives.hpp)

set(CompatSources PosixAlternatives.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Compat
  SOURCES ${CompatSources}
  HEADERS ${CompatHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)


