#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(DesignPatternsHeaders Einsums/DesignPatterns/Singleton.hpp Einsums/DesignPatterns/Lockable.hpp)

set(DesignPatternsSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums DesignPatterns
  SOURCES ${DesignPatternsSources}
  HEADERS ${DesignPatternsHeaders}
  DEPENDENCIES 
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)


