#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if (NOT TARGET FFT::FFT)
    return()
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(FFTHeaders Einsums/FFT.hpp)

set(FFTSources FFT.cpp)

include(Einsums_AddModule)
einsums_add_module(
        Einsums FFT
        SOURCES ${FFTSources}
        HEADERS ${FFTHeaders}
        DEPENDENCIES
        MODULE_DEPENDENCIES
        Einsums_Config
        Einsums_Errors
        Einsums_LinearAlgebra
        Einsums_Tensor
        Einsums_TensorUtilities
        CMAKE_SUBDIRS examples tests
)
