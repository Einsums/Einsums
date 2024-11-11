#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LinearAlgebraHeaders)

set(LinearAlgebraSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums LinearAlgebra
  GLOBAL_HEADER_GEN ON
  SOURCES ${LinearAlgebraSources}
  HEADERS ${LinearAlgebraHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
