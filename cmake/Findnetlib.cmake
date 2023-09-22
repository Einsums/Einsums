include(FindPackageHandleStandardArgs)

find_package(CBLAS MODULE GLOBAL)
find_package(LAPACKE MODULE GLOBAL)

set(CBLAS_FOUND 1)
set(LAPACKE_FOUND 1)
find_package_handle_standard_args(netlib DEFAULT_MSG CBLAS_FOUND LAPACKE_FOUND)
