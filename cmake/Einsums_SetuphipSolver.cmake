include(FetchContent)

fetchcontent_declare(hipsolver GIT_REPO https://github.com/ROCm/hipSOLVER.git FIND_PACKAGE_ARGS)

fetchcontent_makeavailable(hipsolver)
