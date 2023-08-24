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

add_subdirectory(range-v3-0.11.0)
add_subdirectory(fmt-9.1.0)
add_subdirectory(h5cpp-1.10.4-6-EAT)

if(EINSUMS_USE_HPTT)
  add_subdirectory(hptt)
endif()
