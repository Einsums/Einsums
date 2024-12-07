#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TypeSupportHeaders
    Einsums/TypeSupport/AreAllConvertible.hpp
    Einsums/TypeSupport/Arguments.hpp
    Einsums/TypeSupport/CountOfType.hpp
    Einsums/TypeSupport/TypeName.hpp
)

set(TypeSupportSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TypeSupport
  GLOBAL_HEADER_GEN OFF
  SOURCES ${TypeSupportSources}
  HEADERS ${TypeSupportHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
