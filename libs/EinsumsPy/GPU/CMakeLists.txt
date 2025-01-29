#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  set(GPUHeaders EinsumsPy/GPU/PyGPUView.hpp)

  set(GPUSources Export.hip PyGPUView.hip)

  if(EINSUMS_WITH_CUDA)
  foreach(f IN LISTS GPUSources)
    if(f MATCHES ".*\.hip")
      set_source_files_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
    endif()
  endforeach()
endif()

  include(Einsums_AddModule)
  einsums_add_module(
    ${EINSUMS_PYTHON_LIB_NAME} GPU
    SOURCES ${GPUSources}
    HEADERS ${GPUHeaders}
    DEPENDENCIES Einsums
    MODULE_DEPENDENCIES
    CMAKE_SUBDIRS examples tests
    BASE_LIBNAME EinsumsPy
  )

  include(Einsums_ExtendWithPython)
  einsums_extend_with_python_headers(${EINSUMS_PYTHON_LIB_NAME}_GPU)
endif()
