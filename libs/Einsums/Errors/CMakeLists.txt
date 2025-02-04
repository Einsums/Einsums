#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ErrorsHeaders Einsums/Errors/Error.hpp Einsums/Errors/ThrowException.hpp)

set(ErrorsSources ErrorNames.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND ErrorsSources GPUErrors.hip)
  set(ErrorsGPUModules hip::host hip::device roc::hipblas roc::hipsolver)

  if(EINSUMS_WITH_CUDA)
    foreach(f IN LISTS ErrorsSources)
      if(f MATCHES ".*\.hip")
        set_source_files_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
      endif()
    endforeach()
  endif()
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Errors
  SOURCES ${ErrorsSources}
  HEADERS ${ErrorsHeaders}
  DEPENDENCIES ${ErrorsGPUModules}
  MODULE_DEPENDENCIES Einsums_Config Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
