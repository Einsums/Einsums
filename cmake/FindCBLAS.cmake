include(FindPackageHandleStandardArgs)

# MKL include cblas by default. If we find the mkl_cblas.h
# header then assume we have cblas.
check_include_file(mkl_cblas.h HAVE_MKL_CBLAS_H)
if (HAVE_MKL_CBLAS_H)
    set(CBLAS_FOUND TRUE)
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
            NAMES cblas blas
            HINTS ${CBLAS_DIR}
            PATHS /usr /usr/local
            PATH_SUFFIXES lib64 lib
        )

        find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDE_DIRS)

        if (CBLAS_FOUND)
            set(HAVE_CBLAS_H TRUE)

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
            target_compile_definitions(cblas
                INTERFACE
                    HAVE_CBLAS
            )
        endif()
    endif()
endif()