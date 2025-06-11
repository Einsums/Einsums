#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LinearAlgebraHeaders)

set(LinearAlgebraSources Export.cpp)

include(Einsums_AddModule)
einsums_add_module(
  ${EINSUMS_PYTHON_LIB_NAME} LinearAlgebra
  SOURCES ${LinearAlgebraSources}
  HEADERS ${LinearAlgebraHeaders}
  DEPENDENCIES Einsums
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME EinsumsPy
)

include(Einsums_ExtendWithPython)
einsums_extend_with_python_headers(${EINSUMS_PYTHON_LIB_NAME}_LinearAlgebra)

        if(EINSUMS_WITH_CUDA)
          foreach(f IN LISTS LinearAlgebraSources)
            set_source_file_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
          endforeach()
        endif()
        endif()
        