include(FetchContent)
include(CMakePrintHelpers)

#  <<<  general notes  >>>

# * URL preferred over GIT_REPOSITORY
# * commit or permanent tag preferred over branch name for traceability
# * to get the URL_HASH, run `curl -sL <URL> | openssl sha256`
# * to use local source, put path in URL: `URL "/psi/gits/h5cpp"`


#  <<<  range-v3  >>>

find_package(range-v3 CONFIG REQUIRED GLOBAL)


#  <<<  fmt  >>>

find_package(fmt 9.1...<10.0 CONFIG REQUIRED GLOBAL)


#  <<<  CBLAS/LAPACKE  >>>

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


#  <<<  Catch2  >>>

# * CMAKE_MODULE_PATH additions in PARENT_SCOPE make `include(Catch)` available in tests/

if (EINSUMS_ENABLE_TESTING)
    FetchContent_Declare(
      Catch2
      URL https://github.com/catchorg/Catch2/archive/v3.1.0.tar.gz
      URL_HASH SHA256=c252b2d9537e18046d8b82535069d2567f77043f8e644acf9a9fffc22ea6e6f7
      FIND_PACKAGE_ARGS 3.1 CONFIG GLOBAL
      )
    FetchContent_MakeAvailable(Catch2)

    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${catch2_SOURCE_DIR}/extras" PARENT_SCOPE)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};$CACHE{Catch2_DIR}" PARENT_SCOPE)
endif()


#  <<<  h5cpp  >>>

# * 3031c1b = EATplus branch is v1.10.4-6-EAT+2, so upstream plus extended array dims
#   plus deleter stuff plus cmake fetchcontent-able
# * find_package() is disabled since we need patched source
# * h5cpp is header-only, so define Einsums_h5cpp as a practical dependency-including target

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

FetchContent_Declare(
  h5cpp
  URL https://github.com/loriab/h5cpp/archive/3031c1b.tar.gz
  URL_HASH SHA256=6c5676369a8097a6baec579dbe9610c95fe2437c70e8b628345c50566173ce84
  OVERRIDE_FIND_PACKAGE
  )
FetchContent_MakeAvailable(h5cpp)

add_library(Einsums_h5cpp INTERFACE)
target_link_libraries(
  Einsums_h5cpp
  INTERFACE
    h5cpp::h5cpp
    HDF5::HDF5
    ZLIB::ZLIB
  )


