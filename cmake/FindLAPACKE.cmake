include(FindPackageHandleStandardArgs)
include(CheckIncludeFile)
include(CMakePushCheckState)

# Only check for mkl_lapacke.h if we are using MKL
if (TARGET MKL::MKL)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES MKL::MKL)
    check_include_file(mkl_lapacke.h HAVE_MKL_LAPACKE_H)
    if (HAVE_MKL_LAPACKE_H)
        set(LAPACKE_FOUND TRUE)
    endif()
    cmake_pop_check_state()
else()
    check_include_file(lapacke.h HAVE_LAPACKE_H)
    if (HAVE_LAPACKE_H)
        find_path(LAPACKE_INCLUDE_DIRS
            NAMES lapacke.h
            HINTS ${LAPACKE_DIR}
            PATHS /usr /usr/local
            PATH_SUFFIXES include
        )

        find_library(LAPACKE_LIBRARIES lapacke
            HINTS ${LAPACKE_DIR}
            PATHS /usr /usr/local
            PATH_SUFFIXES lib64 lib
        )

        find_package_handle_standard_args(LAPACKE DEFAULT_MSG LAPACKE_LIBRARIES LAPACKE_INCLUDE_DIRS)

        if (LAPACKE_FOUND)
            set(HAVE_LAPACKE_H TRUE)

            message(STATUS "Found components for LAPACKE.")

            add_library(lapacke INTERFACE)
            target_include_directories(lapacke
                INTERFACE
                    ${LAPACKE_INCLUDE_DIRS}
            )
            target_link_libraries(lapacke
                INTERFACE
                    ${LAPACKE_LIBRARIES}
            )
        endif()
    endif()
endif()
