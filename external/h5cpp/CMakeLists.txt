# * v1.10.4-6-EAT6 on EAT branch is upstream v1.10.4-6 tag +2, so last upstream tag
#   plus extended array dimensions to support higher rank tensors plus deleter stuff.
# * find_package() is disabled since we need patched source
# * upstream CMakeLists.txt isn't useable and project is header-only, so to keep code
#   changes and build changes separate, we won't let FetchContent build (`SOURCE_SUBDIR
#   fake`) and will create the interface Einsums_h5cpp target after download.
# * MakeAvailable called here so that install (of vendored headers into einsums namespace)
#   can be localized into this file.

FetchContent_Declare(
  h5cpp
  URL https://github.com/loriab/h5cpp/archive/EAT_align.tar.gz
  URL_HASH SHA256=12fc71709d01ee7e9d04d3c3de75696f3695b68ee63fd0f32734058e52dd5a7c
  SOURCE_SUBDIR fake
  SYSTEM
  OVERRIDE_FIND_PACKAGE
  )

FetchContent_MakeAvailable(h5cpp)

add_library(Einsums_h5cpp INTERFACE)
add_library("${ein}::h5cpp" ALIAS Einsums_h5cpp)

if (EINSUMS_H5CPP_USE_OMP_ALIGNED_ALLOC)
    target_compile_definitions(
      Einsums_h5cpp
      INTERFACE
        $<BUILD_INTERFACE:H5CPP_USE_OMP_ALIGNED_ALLOC>
      )
endif()
set_target_properties(
  Einsums_h5cpp
  PROPERTIES
    EXPORT_NAME h5cpp
  )
target_include_directories(
  Einsums_h5cpp
  # SYSTEM suppresses "error: non-constant-expression cannot be narrowed" for some compilers
  SYSTEM
  INTERFACE
    $<BUILD_INTERFACE:${h5cpp_SOURCE_DIR}>
    # TODO return to this when build headers adjusted   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/einsums>
  )
target_link_libraries(
  Einsums_h5cpp
  INTERFACE
    tgt::hdf5
    ZLIB::ZLIB
  )

install(
  DIRECTORY
    ${h5cpp_SOURCE_DIR}/h5cpp
  COMPONENT ${ein}_Development
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsums
  )

