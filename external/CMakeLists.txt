include(FetchContent)

# <<<  general notes  >>>

# * URL preferred over GIT_REPOSITORY
# * commit or permanent tag preferred over branch name for traceability
# * to get the URL_HASH, run `curl -sL <URL> | openssl sha256`
# * to use local source, put path in URL: `URL "/psi/gits/h5cpp"`


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


# <<<  HDF5  >>>

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)


# <<< h5cpp >>>

# * v1.10.4-6-EAT6 on EAT branch is upstream v1.10.4-6 tag +2, so last upstream tag
#   plus extended array dimensions to support higher rank tensors plus deleter stuff.
# * find_package() is disabled since we need patched source
# * upstream CMakeLists.txt isn't useable and project is header-only, so to keep code
#   changes and build changes separate, we won't let FetchContent build (`SOURCE_SUBDIR
#   fake`) and will create the interface Einsums_h5cpp target after download.

FetchContent_Declare(
  h5cpp
  URL https://github.com/Einsums/h5cpp/archive/v1.10.4-6-EAT6.tar.gz
  URL_HASH SHA256=bc6370d2a2b11d9208c53d7815c2fbcfefc09f509f3a9f4d9cbd76a8c442feac
  SOURCE_SUBDIR fake
  OVERRIDE_FIND_PACKAGE
  )
FetchContent_MakeAvailable(h5cpp)

add_library(Einsums_h5cpp INTERFACE)
target_include_directories(
  Einsums_h5cpp
  SYSTEM  # supress warnings
  INTERFACE
    $<BUILD_INTERFACE:${h5cpp_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_link_libraries(
  Einsums_h5cpp
  INTERFACE
    HDF5::HDF5
    ZLIB::ZLIB
  )

# <<< hptt >> Not using hptt is just asking for slow sorts.
if(EINSUMS_USE_HPTT)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|ARM64)")
    set(ENABLE_ARM ON)
    set(ENABLE_AVX OFF)
  endif()
  add_subdirectory(hptt)
endif()
