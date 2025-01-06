include(FetchContent)


FetchContent_Declare(
    hipsolver
    GIT_REPO https://github.com/ROCm/hipSOLVER.git
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(hipsolver)