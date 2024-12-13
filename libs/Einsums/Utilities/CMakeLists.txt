#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(UtilitiesHeaders
        Einsums/Utilities/Tuple.hpp
        Einsums/Utilities/Random.hpp
)

set(UtilitiesSources Random.cpp)

include(Einsums_AddModule)
einsums_add_module(
        Einsums Utilities
        SOURCES ${UtilitiesSources}
        HEADERS ${UtilitiesHeaders}
        DEPENDENCIES
        MODULE_DEPENDENCIES
                Einsums_Config
        CMAKE_SUBDIRS examples tests
)
