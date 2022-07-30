cmake_minimum_required(VERSION 3.20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FeatureSummary)
include(EinsumsBranding)
include(EinsumsAPI)

project(einsums
    VERSION ${EINSUMS_VERSION}
    LANGUAGES C CXX
)

# Force C++ standard globally, do not fall back, do not use compiler extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force position independent code globally.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EINSUMS_CONTINUOUSLY_TEST_EINSUM "Every call to TensorAlgebra::einsum is tested" OFF)
option(EINSUMS_TEST_EINSUM_ABORT "Abort execution if an error is found" ON)
option(EINSUMS_WARN_NOT_IMPLEMENTED "If a specific case is not implemented in einsum then warn the user" OFF)
option(EINSUMS_SHOW_WARNING "Does the code report warnings to the user or not" ON)
option(EINSUMS_RUNTIME_INDICES_CHECK "Check the sizes of corresponding indices at runtime" ON)
option(EINSUMS_TEST_NANS "Test the result of every einsum for nans" OFF)
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

check_include_file(mkl_cblas.h HAVE_MKL_CBLAS_HEADER)
if (NOT HAVE_MKL_CBLAS_HEADER)
    check_include_file(cblas.h HAVE_CBLAS_HEADER)
    if (NOT HAVE_CBLAS_HEADER)
        add_subdirectory(src/backends/cblas/cblas)
    endif()
endif()

# Prefer MKL lapacke over vendor lapacke.  Additionally MKL lapacke is bundled in with lapack.
check_include_file(mkl_lapacke.h HAVE_MKL_LAPACKE_HEADER)
if (NOT HAVE_MKL_LAPACKE_HEADER)
    include(cmake/FindLAPACKE.cmake)
    if (LAPACKE_FOUND)
        check_include_file(lapacke.h HAVE_LAPACKE_HEADER)
    else()
        # Nothing was found. We will build our internal copy of lapacke
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