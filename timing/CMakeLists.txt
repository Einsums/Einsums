add_executable(test-timing main.cpp)
target_link_libraries(test-timing einsums)

add_executable(test-onemkl onemkl.cpp)
target_link_libraries(test-onemkl einsums)