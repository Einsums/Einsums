#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(hipBLASHeaders Einsums/hipBLAS/hipBLAS.hpp)

set(hipBLASSources )

include(Einsums_AddModule)
einsums_add_module(
  Einsums hipBLAS
  SOURCES ${hipBLASSources}
  HEADERS ${hipBLASHeaders}
  DEPENDENCIES 
  MODULE_DEPENDENCIES Einsums_GPUStreams Einsums_Errors
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)



        if(EINSUMS_WITH_CUDA)
          foreach(f IN LISTS hipBLASSources)
            set_source_file_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
          endforeach()
        endif()
        endif()
        