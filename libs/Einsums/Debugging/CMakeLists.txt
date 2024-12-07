#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(DebuggingHeaders)

set(DebuggingSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Debugging
  GLOBAL_HEADER_GEN ON
  SOURCES ${DebuggingSources}
  HEADERS ${DebuggingHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
