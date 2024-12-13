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

if (EINSUMS_FFT_LIBRARY MATCHES fftw3)
  list(APPEND FFTSources Backends/FFTW3/FFT.cpp Backends/FFTW3/FFT.hpp)
  einsums_add_config_cond_define_namespace(DEFINE EINSUMS_HAVE_FFT_LIBRARY_FFTW3 NAMESPACE FFT)
elseif (EINSUMS_FFT_LIBRARY MATCHES mkl)
  list(APPEND FFTSources Backends/MKL/FFT.cpp Backends/MKL/FFT.hpp)
  einsums_add_config_cond_define_namespace(DEFINE EINSUMS_HAVE_FFT_LIBRARY_MKL NAMESPACE FFT)
endif ()

include(Einsums_AddModule)
einsums_add_module(
        Einsums FFT
        SOURCES ${FFTSources}
        HEADERS ${FFTHeaders}
        DEPENDENCIES FFT::FFT
        MODULE_DEPENDENCIES Einsums_Config Einsums_Errors Einsums_LinearAlgebra Einsums_Tensor
        Einsums_TensorUtilities
        CMAKE_SUBDIRS examples tests
)
