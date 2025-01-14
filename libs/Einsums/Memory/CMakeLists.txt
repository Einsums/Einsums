#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(MemoryHeaders Einsums/Memory/Allocator.hpp Einsums/Memory/Manager.hpp
        Einsums/Memory/StdAllocator.hpp
)

set(MemorySources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Memory
  SOURCES ${MemorySources}
  HEADERS ${MemoryHeaders}
        DEPENDENCIES einsums_dependencies_allocator
        MODULE_DEPENDENCIES Einsums_Config Einsums_Logging
  CMAKE_SUBDIRS examples tests
)
