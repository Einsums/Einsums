include(FetchContent)

# <<< Catch2 >>>
if(EINSUMS_ENABLE_TESTING)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
    FIND_PACKAGE_ARGS 3 NAMES Catch2 GLOBAL)

  FetchContent_MakeAvailable(catch2)
endif()

# <<< range-v3 >>>
FetchContent_Declare(
  rangev3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  GIT_TAG 0.12.0
  FIND_PACKAGE_ARGS 0.12.0 NAMES range-v3 GLOBAL)
  # note that upstream uses `write_basic_package_version_file(... COMPATIBILITY ExactVersion)`
  #   so no version range of compatibilty can be expressed here.

FetchContent_MakeAvailable(rangev3)

# <<< fmt >>>
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.0
  FIND_PACKAGE_ARGS 10.1 GLOBAL)

FetchContent_MakeAvailable(fmt)

# <<<  CBLAS/LAPACKE  >>>
# * note nonexistant SOURCE_SUBDIR here so it fetches over the source but doesn't try to build it.
#   Still built by local src/CMakeLists.txt . Could have done the same w/FetchContent_Populate instead
#   of FetchContent_MakeAvailable but only the latter integrates w/find_package.

FetchContent_Declare(
  netlib
  URL https://github.com/Reference-LAPACK/lapack/archive/v3.11.0.tar.gz
  URL_HASH SHA256=4b9ba79bfd4921ca820e83979db76ab3363155709444a787979e81c22285ffa9
  SOURCE_SUBDIR fake
  FIND_PACKAGE_ARGS MODULE
  )

FetchContent_MakeAvailable(netlib)
set(netlib_SOURCE_DIR "${netlib_SOURCE_DIR}" PARENT_SCOPE)
set(CBLAS_FOUND "${CBLAS_FOUND}" PARENT_SCOPE)
set(LAPACKE_FOUND "${LAPACKE_FOUND}" PARENT_SCOPE)

# <<< h5cpp >>>
# This version is modified to support higher rank tensors.
add_subdirectory(h5cpp-1.10.4-6-EAT)

# <<< hptt >> Not using hptt is just asking for slow sorts.
if(EINSUMS_USE_HPTT)
  string(TOLOWER CMAKE_SYSTEM_PROCESSOR cm_sys_processor)
  if(cm_sys_processor STREQUAL "arm64")
    set(ENABLE_ARM ON)
    set(ENABLE_AVX OFF)
  endif()
  add_subdirectory(hptt)
endif()
