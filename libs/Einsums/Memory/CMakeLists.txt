#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(MemoryHeaders)

set(MemorySources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Memory
  GLOBAL_HEADER_GEN ON
  SOURCES ${MemorySources}
  HEADERS ${MemoryHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
