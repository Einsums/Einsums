#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorBaseHeaders
    Einsums/TensorBase/Common.hpp
    Einsums/TensorBase/IndexUtilities.hpp
    Einsums/TensorBase/TensorBase.hpp
    Einsums/TensorBase/HashFunctions.hpp
)

set(TensorBaseSources IndexUtilities.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorBase
  GLOBAL_HEADER_GEN OFF
  SOURCES ${TensorBaseSources}
  HEADERS ${TensorBaseHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        Einsums_Errors
        Einsums_StringUtil
  CMAKE_SUBDIRS examples tests
)
