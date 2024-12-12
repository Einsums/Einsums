#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ErrorsHeaders
    Einsums/Errors/Error.hpp
    Einsums/Errors/ThrowException.hpp
)

set(ErrorsSources ErrorNames.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND ErrorsHeaders Einsums/Errors/hipblas_status_doc.hpp)
  list(APPEND ErrorsSources GPUErrors.hip)
  set(ErrorsGPUModules hip::host hip::device roc::hipblas roc::hipsolver)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Errors
  GLOBAL_HEADER_GEN ON
  SOURCES ${ErrorsSources}
  HEADERS ${ErrorsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_TypeSupport
    ${ErrorsGPUModules}
  CMAKE_SUBDIRS examples tests
)
