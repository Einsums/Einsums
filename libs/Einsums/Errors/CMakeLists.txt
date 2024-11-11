#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ErrorsHeaders)

set(ErrorsSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Errors
  GLOBAL_HEADER_GEN ON
  SOURCES ${ErrorsSources}
  HEADERS ${ErrorsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
