include(FetchContent)

# <<< Catch2 >>>
if(EINSUMS_ENABLE_TESTING)
  find_package(Catch2 3 QUIET)

  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.4.0)

    FetchContent_MakeAvailable(catch2)
  endif()
endif()

# <<< range-v3 >>>
find_package(range-v3 0.11 QUIET)
if(NOT range-v3_FOUND)
  FetchContent_Declare(
    rangev3
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
    GIT_TAG 0.11.0)

  FetchContent_MakeAvailable(rangev3)
endif()

add_subdirectory(fmt-9.1.0)
add_subdirectory(h5cpp-1.10.4-6-EAT)

if(EINSUMS_USE_HPTT)
  add_subdirectory(hptt)
endif()
