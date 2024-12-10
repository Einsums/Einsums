#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(GPUStreamsHeaders Einsums/GPUStreams/GPUStreams.hpp)

set(GPUStreamsSources GPUStreams.hip)

include(Einsums_AddModule)
einsums_add_module(
  Einsums GPUStreams
  GLOBAL_HEADER_GEN ON
  SOURCES ${GPUStreamsSources}
  HEADERS ${GPUStreamsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    Einsums_Config
    Einsums_Errors
    hip::host
    hip::device
    roc::hipblas
    roc::hipsolver
  CMAKE_SUBDIRS examples tests
)
