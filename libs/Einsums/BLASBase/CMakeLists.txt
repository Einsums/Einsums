#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (EINSUMS_LINALG_INT_INTERFACE STREQUAL ilp64)
    einsums_add_config_define_namespace(DEFINE EINSUMS_BLAS_INTERFACE_ILP64 NAMESPACE BLASBASE)
elseif(EINSUMS_LINALG_INT_INTERFACE STREQUAL lp64)
    einsums_add_config_define_namespace(DEFINE EINSUMS_BLAS_INTERFACE_LP64 NAMESPACE BLASBASE)
else()
    einsums_error("EINSUMS_LINALG_INT_INTERFACE is not defined")
endif()

set(BLASBaseHeaders
    Einsums/BLAS/Types.hpp
)

set(BLASBaseSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums BLASBase
  GLOBAL_HEADER_GEN ON
  SOURCES ${BLASBaseSources}
  HEADERS ${BLASBaseHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
