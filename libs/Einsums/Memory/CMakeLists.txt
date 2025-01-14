#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(MemoryHeaders Einsums/Memory/Allocator.hpp Einsums/Memory/Manager.hpp
        Einsums/Memory/StdAllocator.hpp
)

set(MemorySources)

if (TARGET einsums_dependencies_allocator)
    set(extra_dependencies einsums_dependencies_allocator)
endif ()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Memory
  SOURCES ${MemorySources}
  HEADERS ${MemoryHeaders}
        DEPENDENCIES ${extra_dependencies}
        MODULE_DEPENDENCIES Einsums_Config Einsums_Logging
  CMAKE_SUBDIRS examples tests
)
