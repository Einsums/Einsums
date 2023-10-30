# Downloaded from
#   https://github.com/coderefinery/autocmake/blob/master/modules/safeguards.cmake
# * changed text of in-source message
# * added additional build types that we support

#.rst:
#
# Provides safeguards against in-source builds and bad build types.
#
# Variables used::
#
#   PROJECT_SOURCE_DIR
#   PROJECT_BINARY_DIR
#   CMAKE_BUILD_TYPE

if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please run CMake from top directory and specify a build directory (e.g., cmake -S. -Bbuild).")
endif ()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
string(TOUPPER "${CMAKE_BUILD_TYPE}" cmake_build_type_toupper)

if (NOT cmake_build_type_tolower STREQUAL "debug" AND
        NOT cmake_build_type_tolower STREQUAL "release" AND
        NOT cmake_build_type_tolower STREQUAL "minsizerel" AND
        NOT cmake_build_type_tolower STREQUAL "relwithdebinfo" AND
        NOT cmake_build_type_tolower STREQUAL "asan" AND
        NOT cmake_build_type_tolower STREQUAL "msan" AND
        NOT cmake_build_type_tolower STREQUAL "ubsan")
    message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo, MinSizeRel, ASAN, MSAN, or UBSAN (case-insensitive).")
endif ()
