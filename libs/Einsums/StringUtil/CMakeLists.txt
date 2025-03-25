#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(StringUtilHeaders Einsums/StringUtil/BadLexicalCast.hpp Einsums/StringUtil/FromString.hpp
                      Einsums/StringUtil/Trim.hpp Einsums/StringUtil/MemoryString.hpp
)

set(StringUtilSources MemoryString.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums StringUtil
  SOURCES ${StringUtilSources}
  HEADERS ${StringUtilHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Errors
  CMAKE_SUBDIRS examples tests
)
