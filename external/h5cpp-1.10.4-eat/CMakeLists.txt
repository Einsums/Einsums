# This modified version of h5cpp can be found here
# https://github.com/jturney/h5cpp/releases/tag/v1.10.4-eat

add_library(h5cpp INTERFACE)
target_include_directories(h5cpp INTERFACE .)

find_package(HDF5 REQUIRED COMPONENTS CXX)

target_link_libraries(h5cpp INTERFACE HDF5::HDF5)
add_library(h5cpp::h5cpp ALIAS h5cpp)