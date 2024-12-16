#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(RuntimeHeaders Einsums/Runtime.hpp Einsums/Runtime/InitializeFinalize.hpp)

set(RuntimeSources InitializeFinalize.cpp)

include(Einsums_AddModule)
einsums_add_module(
        Einsums Runtime
        SOURCES ${RuntimeSources}
        HEADERS ${RuntimeHeaders}
        DEPENDENCIES Einsums::h5cpp
        MODULE_DEPENDENCIES Einsums_BLAS Einsums_Config Einsums_Profile
        CMAKE_SUBDIRS examples tests
        #  BASE_LIBNAME Einsums
)


