#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorImplHeaders Einsums/TensorImpl/TensorImpl.hpp Einsums/TensorImpl/TensorImplOperations.hpp)

set(TensorImplSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorImpl
  SOURCES ${TensorImplSources}
  HEADERS ${TensorImplHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    Einsums_Logging
    Einsums_TypeSupport
    Einsums_Config
    Einsums_BLAS
    Einsums_Concepts
    Einsums_BufferAllocator
    Einsums_TensorBase
    Einsums_Errors
    Einsums_BlockManager
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
