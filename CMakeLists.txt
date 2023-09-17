cmake_minimum_required(VERSION 3.24)  # 3.24 for FetchContent_Declare(FIND_PACKAGE_ARGS
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(EinsumsBranding)

project(
  Einsums
  VERSION ${EINSUMS_VERSION}
  LANGUAGES C CXX
  DESCRIPTION "C++ library for compile-time contraction pattern analysis optimizing tensor operations"
  HOMEPAGE_URL https://github.com/jturney/einsums
  )

include(FeatureSummary)
include(EinsumsAPI)

# Force C++ standard globally, do not fall back, do not use compiler extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(RANGES_CXX_STD 17)

# Force position independent code globally.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(EinsumsBuildTypeASAN)
include(EinsumsBuildTypeMSAN)
include(EinsumsBuildTypeUBSAN)

set(ein Einsums)  # Namespace

# ##############################################################################
# User input options                                                           #
# ##############################################################################
include(einsums_option)

# borrowed from pybind11 since einsums_option doesn't work
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting tests build type to Release as none was specified")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

einsums_option(EINSUMS_CONTINUOUSLY_TEST_EINSUM BOOL
               "Every call to TensorAlgebra::einsum is tested" OFF)
einsums_option(EINSUMS_TEST_EINSUM_ABORT BOOL
               "Abort execution if an error is found" ON)
einsums_option(
  EINSUMS_WARN_NOT_IMPLEMENTED BOOL
  "If a specific case is not implemented in einsum then warn the user" OFF)
einsums_option(EINSUMS_SHOW_WARNING BOOL
               "Does the code report warnings to the user or not" ON)
einsums_option(EINSUMS_RUNTIME_INDICES_CHECK BOOL
               "Check the sizes of corresponding indices at runtime" ON)
einsums_option(EINSUMS_TEST_NANS BOOL
               "Test the result of every einsum for nans" OFF)
einsums_option(EINSUMS_USE_HPTT BOOL
               "Use the HPTT package for tensor transpositions" ON)
einsums_option(EINSUMS_ENABLE_TESTING BOOL "Build testing (requires Catch2)"
               ${PROJECT_IS_TOP_LEVEL})
einsums_option(EINSUMS_INSTALL_CMAKEDIR STRING
               "Directory to which Einsums CMake config files installed."
               share/cmake/Einsums)

add_feature_info(
  EINSUMS_CONTINUOUSLY_TEST_EINSUM ${EINSUMS_CONTINUOUSLY_TEST_EINSUM}
  "Test every call to einsum. EXPENSIVE!")
add_feature_info(EINSUMS_TEST_EINSUM_ABORT ${EINSUMS_TEST_EINSUM_ABORT}
                 "Abort execution if an error found")
add_feature_info(EINSUMS_WARN_NOT_IMPLEMENTED ${EINSUMS_WARN_NOT_IMPLEMENTED}
                 "Report if a specific case is not implemented")
add_feature_info(EINSUMS_SHOW_WARNING ${EINSUMS_SHOW_WARNING}
                 "Does the code report warnings to the user or not")
add_feature_info(
  EINSUMS_TEST_NANS ${EINSUMS_TEST_NANS}
  "Test the result of every einsum for nans, infs, and large values")
add_feature_info(EINSUMS_RUNTIME_INDICES_CHECK ${EINSUMS_RUNTIME_INDICES_CHECK}
                 "Check the sizes of corresponding indices at runtime")
add_feature_info(EINSUMS_USE_HPTT ${EINSUMS_USE_HPTT}
                 "Build with internal HPTT version")

include(autocmake_safeguards)

include(cmake/DetectHostArch.cmake)
detect_host_arch()
message(STATUS "${CMAKE_CXX_COMPILER_ID} flags: ${HOST_FLAGS}")

include(cmake/FindITTNOTIFY.cmake)
# include(cmake/FindIntelSYCL.cmake)

# We don't do anything in parallel yet. find_package(MPI)
if(TARGET MPI::MPI_CXX)
  set(HDF5_PREFER_PARALLEL True)
endif()

# Prefer to use MKL
find_package(MKL CONFIG)
if(TARGET MKL::MKL)
  add_library(BLAS::BLAS ALIAS MKL::MKL)
  add_library(LAPACK::LAPACK ALIAS MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

# Set default FFT library to mkl, if mkl was found, otherwise FFTW3
set(EINSUMS_FFT_LIBRARY_DEFAULT "fftw3")
if(TARGET MKL::MKL)
  set(EINSUMS_FFT_LIBRARY_DEFAULT "mkl")
endif()

einsums_option(EINSUMS_FFT_LIBRARY STRING "FFT library"
               ${EINSUMS_FFT_LIBRARY_DEFAULT} STRINGS "fftw3;mkl;off")

include(EinsumsManageFFTLibraries)

find_package(OpenMP REQUIRED MODULE COMPONENTS CXX)
if(OpenMP_FOUND)
  message(
    STATUS
      "OpenMP Version ${OpenMP_CXX_VERSION_MAJOR}.${OpenMP_CXX_VERSION_MINOR}")
endif()

include(cmake/BackwardConfig.cmake)

add_subdirectory(external)
add_subdirectory(src)

if (EINSUMS_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(timing)
# add_subdirectory(experiments)

feature_summary(INCLUDE_QUIET_PACKAGES WHAT PACKAGES_FOUND PACKAGES_NOT_FOUND
                                            ENABLED_FEATURES DISABLED_FEATURES)
