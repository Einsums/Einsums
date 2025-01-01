#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LoggingHeaders Einsums/Logging.hpp)

set(LoggingSources Logging.cpp)

include(Einsums_AddModule)
einsums_add_module(
        Einsums Logging
        SOURCES ${LoggingSources}
        HEADERS ${LoggingHeaders}
        DEPENDENCIES
        MODULE_DEPENDENCIES Einsums_Assertion Einsums_Config Einsums_StringUtil
        CMAKE_SUBDIRS examples tests
        BASE_LIBNAME Einsums
)


