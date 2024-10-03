if(__VERSION)
    set(PYBIND11_PYTHON_VERSION ${__VERSION})
    find_package(Python3 "${__VERSION}" EXACT COMPONENTS Interpreter Development REQUIRED)
    set(__core_name "core-${__VERSION}")
else()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    set(__core_name "core")
endif()
find_package(pybind11 CONFIG REQUIRED)
message("Found Python version ${Python3_VERSION}")

add_einsums_library(${__core_name} MODULE
    SOURCES
    PyTensorAlgebra.cpp
    PyMain.cpp
    DEPENDS pybind11::module pybind11::lto ${Python3_LIBRARIES} tgt::lapack range-v3::range-v3 fmt::fmt Einsums::h5cpp
    PUBLIC_DEPENDS einsums)

set_target_properties(${__core_name} PROPERTIES
    BUILD_RPATH                     "${_LIB_RPATH};${CMAKE_BUILD_RPATH}"
    INSTALL_RPATH                   "${_LIB_RPATH};${CMAKE_INSTALL_RPATH};${_LIB_RPATH}/../"
    OUTPUT_NAME                     "core"
)

extend_einsums_target(${__core_name}
    DEPENDS pybind11::windows_extras
    CONDITION MSVC)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${__core_name})
endif()

extend_einsums_target(${__core_name}
    SOURCES
    PyGpuExceptions.hip
    PyGPUView.hip
    PyTensorAlgebra.hip
    CONDITION EINSUMS_BUILD_HIP
)

pybind11_extension(${__core_name})

target_compile_features(${__core_name} PUBLIC cxx_std_20)

foreach(define IN LISTS defines)
    extend_einsums_target(
        ${__core_name}
        DEFINES ${define}
        CONDITION ${define}
    )
endforeach()

# All need to be LIBRARY_BASE_PATH because Windows thinks this is a runtime artifact, not a library.
install(
    TARGETS ${__core_name} 
    EXPORT primary_set
    RUNTIME DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
    LIBRARY DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
    ARCHIVE DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
)