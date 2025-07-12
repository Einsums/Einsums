#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(IteratorHeaders Einsums/Iterator/Enumerate.hpp Einsums/Iterator/Zip.hpp)

set(IteratorSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Iterator
  SOURCES ${IteratorSources}
  HEADERS ${IteratorHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
