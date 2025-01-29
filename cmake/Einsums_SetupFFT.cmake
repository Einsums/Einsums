# Set default FFT library to mkl, if mkl was found, otherwise FFTW3
set(EINSUMS_FFT_LIBRARY_DEFAULT "fftw3")
if(${EINSUMS_LINALG_VENDOR} MATCHES "[mM][kK][lL]")
  set(EINSUMS_FFT_LIBRARY_DEFAULT "mkl")
endif()

einsums_option(
  EINSUMS_FFT_LIBRARY STRING "FFT library" ${EINSUMS_FFT_LIBRARY_DEFAULT} STRINGS "fftw3;mkl;off"
)

# If MKL is found then we must use it.
if(NOT EINSUMS_FFT_LIBRARY MATCHES off AND EINSUMS_LINALG_VENDOR STREQUAL MKL)
  if(NOT EINSUMS_FFT_LIBRARY MATCHES mkl)
    # Do this at least until we can ensure we link to FFTW3 for FFT and not MKL.
    message(FATAL_ERROR "MKL was detected. You must use MKL's FFT library")
  endif()
endif()

function(fft_mkl)
  if(EINSUMS_LINALG_VENDOR STREQUAL MKL)
    add_library(FFT::FFT ALIAS tgt::lapack)
  else()
    message(FATAL_ERROR "MKL FFT library requested but MKL was not found.")
  endif()
endfunction()

function(fft_fftw3)
  # Attempt to find FFTW for real
  find_package(FFTW COMPONENTS FLOAT_LIB DOUBLE_LIB)

  if(FFTW_FLOAT_LIB_FOUND AND FFTW_DOUBLE_LIB_FOUND)
    add_library(FFT::FFT INTERFACE IMPORTED)
    target_link_libraries(FFT::FFT INTERFACE FFTW::Float FFTW::Double)
  endif()
endfunction()

if(EINSUMS_FFT_LIBRARY MATCHES mkl)
  fft_mkl()
elseif(EINSUMS_FFT_LIBRARY MATCHES fftw3)
  fft_fftw3()
elseif(EINSUMS_FFT_LIBRARY MATCHES off)
  einsums_info("FFT support will not be included.")
else()
  einsums_error("EINSUMS_FFT_LIBRARY(${EINSUMS_FFT_LIBRARY}) does not match mkl or fftw3.")
endif()

# Make sure an FFT library was found
if(NOT TARGET FFT::FFT)
  einsums_info(STATUS "No FFT library was found.")
endif()
