#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PrintHeaders Einsums/Print.hpp)

set(PrintSources Print.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Print
  GLOBAL_HEADER_GEN OFF
  SOURCES ${PrintSources}
  HEADERS ${PrintHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    Einsums_Config
    Einsums_Concepts
  CMAKE_SUBDIRS examples tests
)
