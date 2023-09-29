# <<<  CBLAS/LAPACKE  >>>
# * note nonexistant SOURCE_SUBDIR here so it fetches over the source but
#   doesn't try to build it. Still built by local src/CMakeLists.txt . Could
#   have done the same w/FetchContent_Populate instead of
#   FetchContent_MakeAvailable but only the latter integrates w/find_package.

FetchContent_Declare(
  netlib
  URL https://github.com/Reference-LAPACK/lapack/archive/v3.11.0.tar.gz
  URL_HASH SHA256=4b9ba79bfd4921ca820e83979db76ab3363155709444a787979e81c22285ffa9
  SOURCE_SUBDIR fake
  FIND_PACKAGE_ARGS MODULE
  )

