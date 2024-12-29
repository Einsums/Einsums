#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(RuntimeHeaders
        Einsums/Runtime.hpp
        Einsums/Runtime/InitRuntime.hpp
        #    Einsums/Runtime/InitializeFinalize.hpp
        Einsums/Runtime/Runtime.hpp
        Einsums/Runtime/RuntimeForward.hpp
        Einsums/Runtime/StartupFunction.hpp
        Einsums/Runtime/ShutdownFunction.hpp
)

set(RuntimeSources
        #        InitializeFinalize.cpp
        InitRuntime.cpp
        Runtime.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  set(RuntimeGPUStreamsInit Einsums_GPUStreams)
endif()

include(Einsums_AddModule)
einsums_add_module(
        Einsums Runtime
        SOURCES ${RuntimeSources}
        HEADERS ${RuntimeHeaders}
        DEPENDENCIES Einsums::h5cpp
        MODULE_DEPENDENCIES
        Einsums_Assertion
        Einsums_BLAS
        Einsums_Config
        Einsums_Debugging
        Einsums_Profile
        Einsums_Utilities
        Einsums_RuntimeConfiguration
        Einsums_Errors
        ${RuntimeGPUStreamsInit}
        CMAKE_SUBDIRS examples tests
        # BASE_LIBNAME Einsums
)
