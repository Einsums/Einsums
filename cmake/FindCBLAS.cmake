include(FindPackageHandleStandardArgs)
include(CheckIncludeFile)
include(CMakePushCheckState)

# Check for mkl_cblas.h only if we are using MKL.
if (TARGET MKL::MKL)
    cmake_push_check_state()
    # MKL include cblas by default. If we find the mkl_cblas.h
    # header then assume we have cblas.
    set(CMAKE_REQUIRED_LIBRARIES MKL::MKL)
    check_include_file(mkl_cblas.h HAVE_MKL_CBLAS_H)
    if (HAVE_MKL_CBLAS_H)
        set(CBLAS_FOUND TRUE)
    endif()
    cmake_pop_check_state()
else()
    check_include_file(cblas.h HAVE_CBLAS_H)
    if (HAVE_CBLAS_H)
        find_path(CBLAS_INCLUDE_DIRS
            NAMES cblas.h
            HINTS ${CBLAS_DIR}
            PATHS /usr /usr/local
            PATH_SUFFIXES include
        )

        find_library(CBLAS_LIBRARIES
            NAMES cblas
            # formerly NAMES included blas, but this was "finding" CBLAS pkg when BLAS installation (headers and lib) and CBLAS headers were present
            HINTS ${CBLAS_DIR}
            PATHS /usr /usr/local
            PATH_SUFFIXES lib64 lib
        )

        find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDE_DIRS)

        if (CBLAS_FOUND)
            message(STATUS "Found components for CBLAS.")

            add_library(cblas INTERFACE)
            target_include_directories(cblas
                INTERFACE
                    ${CBLAS_INCLUDE_DIRS}
            )
            target_link_libraries(cblas
                INTERFACE
                    ${CBLAS_LIBRARIES}
            )
        endif()
    else()
        set(HAVE_CBLAS_H OFF)
    endif()
endif()
