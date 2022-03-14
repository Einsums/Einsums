cmake_minimum_required(VERSION 3.20)

project(EinsumsInCpp)

add_library(project-options INTERFACE)
target_compile_features(project-options 
    INTERFACE 
        cxx_std_17
)
set_target_properties(project-options 
    PROPERTIES 
        CXX_EXTENSIONS OFF
)

option(EINSUMS_CONTINUOUSLY_TEST_EINSUM "Every call to TensorAlgebra::einsum is tested" OFF)
if (EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    target_compile_definitions(project-options
        INTERFACE
            EINSUMS_CONTINUOUSLY_TEST_EINSUM
    )
endif()

option(EINSUMS_TEST_EINSUM_ABORT "Abort execution if an error is found" ON)
if (EINSUMS_TEST_EINSUM_ABORT)
    target_compile_definitions(project-options
        INTERFACE
            EINSUMS_TEST_EINSUM_ABORT
    )
endif()


option(EINSUMS_WARN_NOT_IMPLEMENTED "If a specific case is not implemented in einsum then warn the user" OFF)
if (EINSUMS_WARN_NOT_IMPLEMENTED)
    target_compile_definitions(project-options
        INTERFACE
            EINSUMS_WARN_NOT_IMPLEMENTED
    )
endif()

option(EINSUMS_SHOW_WARNING "Does the code report warnings to the user or not" ON)
if (EINSUMS_SHOW_WARNING)
    target_compile_definitions(project-options
        INTERFACE
            EINSUMS_SHOW_WARNING
    )
endif()

option(EINSUMS_USE_HPTT "Use the HPTT package for tensor transpositions" ON)

# Attempt to find MKL. This assumes the user set MKL_DIR.
# set(BLA_VENDOR Intel10_64_dyn)
# find_package(MKL CONFIG)
# message(STATUS "${MKL_IMPORTED_TARGETS}")
if (TARGET MKL::MKL)
    add_library(BLAS::BLAS ALIAS MKL::MKL)
    add_library(LAPACK::LAPACK ALIAS MKL::MKL)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()
find_package(OpenMP)

include(cmake/BackwardConfig.cmake)

add_subdirectory(external)
add_subdirectory(src)

add_subdirectory(tests)
add_subdirectory(timing)
