include(FetchContent)


FetchContent_Declare(
    hipblas_common
    GIT_REPOSITORY git@github.com:ROCm/hipBLAS-common.git
    FIND_PACKAGE_ARGS
)

FetchContent_Declare(
    hipblas
    GIT_REPOSITORY https://github.com/ROCm/hipBLAS.git
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(hipblas_common hipblas)