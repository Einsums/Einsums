include(FetchContent)

FetchContent_Declare(
  librett
  GIT_REPOSITORY https://github.com/victor-anisimov/Librett.git
  FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(librett)

if(NOT TARGET librett::librett)
  message(FATAL_ERROR "Did not find librett!")
endif()
