#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  set(GPUStreamsHeaders Einsums/GPUStreams/GPUStreams.hpp Einsums/GPUStreams/InitModule.hpp)

  set(GPUStreamsSources GPUStreams.hip)

  if(EINSUMS_WITH_CUDA)
    foreach(f IN LISTS GPUStreamsSources)
      if(f MATCHES ".*\.hip")
        set_source_files_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
      endif()
    endforeach()
  endif()

  include(Einsums_AddModule)
  einsums_add_module(
    Einsums GPUStreams
    SOURCES ${GPUStreamsSources}
    HEADERS ${GPUStreamsHeaders}
    DEPENDENCIES hip::host hip::device roc::hipblas roc::hipsolver
    MODULE_DEPENDENCIES Einsums_Config Einsums_Errors Einsums_Runtime Einsums_BufferAllocator
    CMAKE_SUBDIRS examples tests
  )
endif()
