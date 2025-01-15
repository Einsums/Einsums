include(FetchContent)

fetchcontent_declare(
  hipblas_common
  GIT_REPOSITORY git@github.com:ROCm/hipBLAS-common.git
  FIND_PACKAGE_ARGS
)

fetchcontent_declare(
  hipblas
  GIT_REPOSITORY https://github.com/ROCm/hipBLAS.git
  FIND_PACKAGE_ARGS
)

fetchcontent_makeavailable(hipblas_common hipblas)
