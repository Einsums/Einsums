#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(hipBLASHeaders Einsums/hipBLAS.hpp)

set(hipBLASSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums hipBLAS
  SOURCES ${hipBLASSources}
  HEADERS ${hipBLASHeaders}
  DEPENDENCIES hipblas hipsolver hip::host hip::device
  MODULE_DEPENDENCIES Einsums_Config Einsums_Concepts Einsums_BLASBase Einsums_hipBLASVendor
                      Einsums_Runtime
  CMAKE_SUBDIRS examples tests
)
endif()
