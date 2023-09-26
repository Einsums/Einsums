include(FetchContent)

# <<<  general notes  >>>

# * URL preferred over GIT_REPOSITORY
# * commit or permanent tag preferred over branch name for traceability
# * to get the URL_HASH, run `curl -sL <URL> | openssl sha256`
# * to use local source, put path in URL: `URL "/psi/gits/h5cpp"`

if(EINSUMS_ENABLE_TESTING)
    add_subdirectory(catch2)
endif()
add_subdirectory(fmt)
add_subdirectory(rangev3)
add_subdirectory(h5cpp)

add_subdirectory(netlib)

FetchContent_MakeAvailable(netlib)
set(netlib_SOURCE_DIR
    "${netlib_SOURCE_DIR}"
    PARENT_SCOPE)
set(CBLAS_FOUND
    "${CBLAS_FOUND}"
    PARENT_SCOPE)
set(LAPACKE_FOUND
    "${LAPACKE_FOUND}"
    PARENT_SCOPE)


# <<<  HDF5  >>>

# * promote Mm version's scope so that it can be baked into EinsumsConfig

find_package(ZLIB REQUIRED)
find_package(TargetHDF5 REQUIRED)
set(TargetHDF5_VERSION_Mm ${TargetHDF5_VERSION_Mm} PARENT_SCOPE)

if(EINSUMS_USE_HPTT)
    # EXCLUDE_FROM_ALL to suppress install
    add_subdirectory(hptt EXCLUDE_FROM_ALL)
endif()

FetchContent_MakeAvailable(h5cpp)

add_library(Einsums_h5cpp INTERFACE)
add_library("${ein}::h5cpp" ALIAS Einsums_h5cpp)
set_target_properties(
  Einsums_h5cpp
  PROPERTIES
    EXPORT_NAME h5cpp
  )
target_include_directories(
  Einsums_h5cpp
  SYSTEM  # supress warnings
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

if(EINSUMS_ENABLE_TESTING)
    FetchContent_MakeAvailable(catch2)
endif()
FetchContent_MakeAvailable(rangev3)
FetchContent_MakeAvailable(fmt)

