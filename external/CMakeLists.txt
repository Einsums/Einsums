include(FetchContent)

# <<< Catch2 >>>
if(EINSUMS_ENABLE_TESTING)
  find_package(Catch2 3 GLOBAL QUIET)

  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.4.0)

    FetchContent_MakeAvailable(catch2)
  endif()
endif()

# <<< range-v3 >>>
find_package(range-v3 0.11 GLOBAL QUIET)
if(NOT range-v3_FOUND)
  FetchContent_Declare(
    rangev3
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
    GIT_TAG 0.11.0)

  FetchContent_MakeAvailable(rangev3)
endif()

# <<< fmt >>>
find_package(fmt 9.1 GLOBAL QUIET)
if(NOT fmt_FOUND)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.0)

  FetchContent_MakeAvailable(fmt)
endif()

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
