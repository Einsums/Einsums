#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(GPUMemoryHeaders Einsums/GPUMemory/GPUAllocator.hpp Einsums/GPUMemory/InitModule.hpp Einsums/GPUMemory/ModuleVars.hpp Einsums/GPUMemory/GPUPointer.hpp)

set(GPUMemorySources InitModule.cpp ModuleVars.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums GPUMemory
  SOURCES ${GPUMemorySources}
  HEADERS ${GPUMemoryHeaders}
  DEPENDENCIES 
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)

  if(EINSUMS_WITH_CUDA)
    foreach(f IN LISTS GPUMemorySources)
      set_source_file_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
    endforeach()
  endif()
endif()
        