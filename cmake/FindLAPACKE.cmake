include(FindPackageHandleStandardArgs)

check_include_file(mkl_lapacke.h HAVE_MKL_LAPACKE_H)
if (HAVE_MKL_LAPACKE_H)
    set(LAPACKE_FOUND TRUE)
else()
# if (NOT LAPACKE_FOUND AND NOT HAVE_MKL_LAPACKE_H)

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
        target_compile_definitions(lapacke
            INTERFACE
                HAVE_LAPACKE
        )
    endif()

endif()