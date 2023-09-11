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

FetchContent_MakeAvailable(rangev3)

# <<< fmt >>>
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.0
  FIND_PACKAGE_ARGS 10.1 GLOBAL)

FetchContent_MakeAvailable(fmt)

# <<< h5cpp >>>
# This version is modified to support higher rank tensors.
add_subdirectory(h5cpp-1.10.4-6-EAT)

# <<< hptt >> Not using hptt is just asking for slow sorts.
if(EINSUMS_USE_HPTT)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL
                                                "ARM64")
    set(ENABLE_ARM ON)
    set(ENABLE_AVX OFF)
  endif()
  add_subdirectory(hptt)
endif()
