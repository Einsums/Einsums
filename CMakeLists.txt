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

option(EINSUMS_VERBOSE "Print what algorithm einsum is using" OFF)
if (EINSUMS_VERBOSE)
    target_compile_definitions(project-options
        INTERFACE
            EINSUMS_VERBOSE    
    )
endif()

option(EINSUMS_WARN_NOT_IMPLEMENTED "If a specific case is not implemented in einsum then warn the user" OFF)
if (EINSUMS_WARN_NOT_IMPLEMENTED)
    target_compile_definitions(project-options
        INTERFACE
            EINSUMS_WARN_NOT_IMPLEMENTED
    )
endif()

# Attempt to find MKL
set(BLA_VENDOR Intel10_64_dyn)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_subdirectory(external)
