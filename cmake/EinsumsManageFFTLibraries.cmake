include_guard()

# If MKL is found then we must use it.
if (NOT EINSUMS_FFT_LIBRARY MATCHES off AND TARGET MKL::MKL)
    if (NOT EINSUMS_FFT_LIBRARY MATCHES mkl)
        # Do this atleast until we can ensure we link to FFTW3 for FFT and not MKL.
        message(FATAL_ERROR "MKL was detected. You must use MKL's FFT library")
    endif()
endif()

function(fft_mkl)
    if (TARGET MKL::MKL)
        add_library(FFT::FFT ALIAS MKL::MKL)
    else()
        message(FATAL_ERROR "MKL FFT library requested but MKL was not found.")
    endif()
endfunction()

function(fft_fftw3)
    # Attempt to find FFTW for real
    find_package(FFTW
        COMPONENTS
            FLOAT_LIB
            DOUBLE_LIB
    )

    if (FFTW_FLOAT_LIB_FOUND AND FFTW_DOUBLE_LIB_FOUND)
        add_library(FFT::FFT INTERFACE IMPORTED)
        target_link_libraries(FFT::FFT
            INTERFACE
                FFTW::Float
                FFTW::Double
        )
    endif()
endfunction()

# Unable to figure out how to build single and double precision at once.
function(build_fftw3)
    # Machinery for running the external project
    set(EXTERNAL_FFTW_VERSION 3.3.10)
    set(EINSUMS_BUILD_OWN_FFTW_URL
        "http://www.fftw.org/fftw-${EXTERNAL_FFTW_VERSION}.tar.gz" CACHE STRING
        "URL from where to download fftw (use an absolute path when offline, adjust EINSUMS_BUILD_OWN_FFTW_MD5 if downloading other version than ${EXTERNAL_FFTW_VERSION})")
    set(EINSUMS_BUILD_OWN_FFTW_MD5 8ccbf6a5ea78a16dbc3e1306e234cc5c CACHE STRING
        "Expected MD5 hash for the file at GMX_BUILD_OWN_FFTW_URL")
    mark_as_advanced(EINSUMS_BUILD_OWN_FFTW_URL EINSUMS_BUILD_OWN_FFTW_MD5)

    set(FFTW_ARCH ENABLE_SSE=ON ENABLE_SSE2=ON)
    if (HAS_AVX)
        list(APPEND FFTW_ARCH ENABLE_AVX=ON)
    endif (HAS_AVX)

    if (HAS_AVX2)
        list(APPEND FFTW_ARCH ENABLE_AVX2=ON)
    endif (HAS_AVX2)

    list(TRANSFORM FFTW_ARCH PREPEND -D)

    include(ExternalProject)
    ExternalProject_Add(
        build-fftw
        URL ${EINSUMS_BUILD_OWN_FFTW_URL}
        URL_MD5 ${EINSUMS_BUILD_OWN_FFTW_MD5}
        CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_POSITION_INDEPENDENT_CODE=ON  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DENABLE_OPENMP=ON -DENABLE_FLOAT=ON ${FFTW_ARCH}
    )
    ExternalProject_get_property(build-fftw INSTALL_DIR)
    set(FFTW_SINGLE_LIBRARIES ${INSTALL_DIR}/lib/libfftw3f${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(FFTW_SINGLE_INCLUDE_DIRS ${INSTALL_DIR}/include)

    add_library(fftw STATIC IMPORTED)
    set_target_properties(fftw
        PROPERTIES
            IMPORTED_LOCATION ${FFTW_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIRS}
    )
    add_dependencies(fftw build-fftw)
    add_library(FFT::FFT ALIAS fftw)

    set(EINSUMS_FFT_LIBRARY FFTW3 PARENT_SCOPE)
endfunction()

if (EINSUMS_FFT_LIBRARY MATCHES mkl)
    fft_mkl()
elseif(EINSUMS_FFT_LIBRARY MATCHES fftw3)

    # Check for fftw3
    fft_fftw3()

    # if (NOT TARGET FFT::FFT)
        # build_fftw3()
    # endif()
elseif(EINSUMS_FFT_LIBRARY MATCHES off)
    message(STATUS "FFT support will not be included.")
else()
    message(FATAL_ERROR "EINSUMS_FFT_LIBRARY(${EINSUMS_FFT_LIBRARY}) does not match mkl or fftw3.")
endif()

# Make sure an FFT library was found
if (NOT TARGET FFT::FFT)
    message(STATUS "No FFT library was found or being built.")
endif()