#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ReflectionHeaders
        Einsums/Reflection/Default.hpp
        Einsums/Reflection/Detail/RemoveNamespace.hpp
)

set(ReflectionSources Temporary.cpp)

include(Einsums_AddModule)
einsums_add_module(
        EinsumsExperimental Reflection
        SOURCES ${ReflectionSources}
        HEADERS ${ReflectionHeaders}
        DEPENDENCIES Einsums
        MODULE_DEPENDENCIES Einsums_TypeSupport
        CMAKE_SUBDIRS examples tests
        BASE_LIBNAME EinsumsExperimental
)


