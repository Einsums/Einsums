#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(StringUtilHeaders Einsums/StringUtil/Trim.hpp)

set(StringUtilSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums StringUtil
  SOURCES ${StringUtilSources}
  HEADERS ${StringUtilHeaders}
  DEPENDENCIES
        MODULE_DEPENDENCIES Einsums_Config
  CMAKE_SUBDIRS examples tests
)
