cmake_minimum_required(VERSION 3.24)

project(profile_test C CXX Fortran)

find_package(Einsums 1 REQUIRED)

add_executable(profile_for_loop profile_for_loop.c)
target_link_libraries(profile_for_loop PRIVATE m)

add_executable(profile_fort_loop profile_fort_loop.f03)

add_executable(profile_einsums profile_einsums.cpp)
target_link_libraries(profile_einsums PRIVATE Einsums::Einsums)

add_executable(profile_blas profile_blas.cpp)
target_link_libraries(profile_blas PRIVATE Einsums::Einsums)