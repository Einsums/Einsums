cmake_minimum_required(VERSION 3.20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FeatureSummary)
include(EinsumsBranding)
include(EinsumsAPI)

project(einsums VERSION ${EINSUMS_VERSION})

add_library(einsums-project-options INTERFACE)
target_compile_features(einsums-project-options
    INTERFACE
        cxx_std_17
)
set_target_properties(einsums-project-options
    PROPERTIES
        CXX_EXTENSIONS OFF
)

option(EINSUMS_CONTINUOUSLY_TEST_EINSUM "Every call to TensorAlgebra::einsum is tested" OFF)
if (EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    target_compile_definitions(einsums-project-options
        INTERFACE
            EINSUMS_CONTINUOUSLY_TEST_EINSUM
    )
endif()

option(EINSUMS_TEST_EINSUM_ABORT "Abort execution if an error is found" ON)
if (EINSUMS_TEST_EINSUM_ABORT)
    target_compile_definitions(einsums-project-options
        INTERFACE
            EINSUMS_TEST_EINSUM_ABORT
    )
endif()

option(EINSUMS_WARN_NOT_IMPLEMENTED "If a specific case is not implemented in einsum then warn the user" OFF)
if (EINSUMS_WARN_NOT_IMPLEMENTED)
    target_compile_definitions(einsums-project-options
        INTERFACE
            EINSUMS_WARN_NOT_IMPLEMENTED
    )
endif()

option(EINSUMS_SHOW_WARNING "Does the code report warnings to the user or not" ON)
if (EINSUMS_SHOW_WARNING)
    target_compile_definitions(einsums-project-options
        INTERFACE
            EINSUMS_SHOW_WARNING
    )
endif()

option(EINSUMS_RUNTIME_INDICES_CHECK "Check the sizes of corresponding indices at runtime" ON)
if (EINSUMS_RUNTIME_INDICES_CHECK)
    target_compile_definitions(einsums-project-options
        INTERFACE
            EINSUMS_RUNTIME_INDICES_CHECK
    )
endif()

option(EINSUMS_TEST_NANS "Test the result of every einsum for nans" OFF)
if (EINSUMS_TEST_NANS)
    target_compile_definitions(einsums-project-options
        INTERFACE
            EINSUMS_TEST_NANS
    ) 
endif()

option(EINSUMS_USE_HPTT "Use the HPTT package for tensor transpositions" ON)

add_feature_info(EINSUMS_CONTINUOUSLY_TEST_EINSUM ${EINSUMS_CONTINUOUSLY_TEST_EINSUM} "Test every call to einsum. EXPENSIVE!")
add_feature_info(EINSUMS_TEST_EINSUM_ABORT ${EINSUMS_TEST_EINSUM_ABORT} "Abort execution if an error found")
add_feature_info(EINSUMS_WARN_NOT_IMPLEMENTED ${EINSUMS_WARN_NOT_IMPLEMENTED} "Report if a specific case is not implemented")
add_feature_info(EINSUMS_SHOW_WARNING ${EINSUMS_SHOW_WARNING} "Does the code report warnings to the user or not")
add_feature_info(EINSUMS_TEST_NANS ${EINSUMS_TEST_NANS} "Test the result of every einsum for nans, infs, and large values")
add_feature_info(EINSUMS_RUNTIME_INDICES_CHECK ${EINSUMS_RUNTIME_INDICES_CHECK} "Check the sizes of corresponding indices at runtime")
add_feature_info(EINSUMS_USE_HPTT ${EINSUMS_USE_HPTT} "Build with internal HPTT version")

include(cmake/DetectHostArch.cmake)
detect_host_arch()
message(STATUS "${CMAKE_CXX_COMPILER_ID} flags: ${HOST_FLAGS}")

if (UNIX)
    if (CMAKE_CXX_COMPILER_ID MATCHES Intel OR CMAKE_CXX_COMPILER_ID MATCHES IntelLLVM)
        target_compile_options(einsums-project-options INTERFACE -fp-model=precise)
    endif()
endif ()

target_compile_options(einsums-project-options INTERFACE -fPIC)

# NOTE: Ideally this would be included but clangd assumes the possible -xHost is specifying the programming language.
# target_compile_options(einsums-project-options
#     INTERFACE
#         ${HOST_FLAGS}
# )

include(cmake/FindITTNOTIFY.cmake)
# include(cmake/FindIntelSYCL.cmake)

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

# Prefer MKL lapacke over vendor lapacke.  Additionally MKL lapacke is bundled in with lapack.
check_include_file(mkl_lapacke.h HAVE_MKL_LAPACKE_HEADER)
if (NOT HAVE_MKL_LAPACKE_HEADER)
    include(cmake/FindLAPACKE.cmake)
    if (LAPACKE_FOUND)
        check_include_file(lapacke.h HAVE_LAPACKE_HEADER)
    endif()
endif()

find_package(OpenMP)

include(cmake/BackwardConfig.cmake)

add_subdirectory(external)
add_subdirectory(src)

add_subdirectory(tests)
add_subdirectory(timing)

feature_summary(INCLUDE_QUIET_PACKAGES WHAT
    PACKAGES_FOUND PACKAGES_NOT_FOUND
    ENABLED_FEATURES DISABLED_FEATURES
)