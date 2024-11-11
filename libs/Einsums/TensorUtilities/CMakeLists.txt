#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorUtilitiesHeaders)

set(TensorUtilitiesSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorUtilities
  GLOBAL_HEADER_GEN ON
  SOURCES ${TensorUtilitiesSources}
  HEADERS ${TensorUtilitiesHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
