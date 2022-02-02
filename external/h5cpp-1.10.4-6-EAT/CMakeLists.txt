# This modified version of h5cpp can be found here
# https://github.com/jturney/h5cpp/releases/tag/v1.10.4-EAT

add_library(h5cpp INTERFACE)
target_include_directories(h5cpp SYSTEM INTERFACE .)

if (WIN32)
	find_package(HDF5 REQUIRED COMPONENTS CXX NAMES hdf5)
	find_package(ZLIB REQUIRED NAMES zlib)

	if (TARGET hdf5::hdf5-static)
		add_library(HDF5::HDF5 ALIAS hdf5::hdf5-static)
	endif()
	if (TARGET hdf5::zlib-static)
		add_library(ZLIB::ZLIB ALIAS hdf5::zlib-static)
	endif()

	target_include_directories(h5cpp SYSTEM INTERFACE ../zlib)
else(WIN32)
	find_package(HDF5 REQUIRED COMPONENTS CXX)
	find_package(ZLIB REQUIRED)
endif(WIN32)

target_link_libraries(h5cpp INTERFACE HDF5::HDF5 ZLIB::ZLIB)
add_library(h5cpp::h5cpp ALIAS h5cpp)