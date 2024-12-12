#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BLASHeaders
    Einsums/BLAS.hpp
)

set(BLASSources
    BLAS.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums BLAS
  SOURCES ${BLASSources}
  HEADERS ${BLASHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        Einsums_Config
        Einsums_Concepts
        Einsums_BLASBase
        Einsums_BLASVendor
  CMAKE_SUBDIRS examples tests
)
