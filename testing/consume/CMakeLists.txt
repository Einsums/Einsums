cmake_minimum_required(VERSION 3.24)
project(sample)

find_package(Einsums 1 REQUIRED)

include(CMakePrintHelpers)
cmake_print_properties(
  TARGETS Einsums::Einsums
  PROPERTIES IMPORTED_LINK_DEPENDENT_LIBRARIES
             INTERFACE_LOCATION
             INTERFACE_LINK_LIBRARIES
             INTERFACE_INCLUDE_DIRECTORIES
             INTERFACE_COMPILE_DEFINITIONS
             INTERFACE_COMPILE_OPTIONS
             INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
)

add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld Einsums::Einsums)
