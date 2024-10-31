if(__VERSION)
    find_package(Python "${__VERSION}" EXACT COMPONENTS Interpreter Development REQUIRED)
    set(__core_name "core-${__VERSION}")
    set(__except_name "gpu_except-${__VERSION}")
    set(__pybind_dir "pybind11-${__VERSION}")
else()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    set(__core_name "core")
    set(__except_name "gpu_except")
    set(__pybind_dir "pybind11")
endif()


find_package(pybind11 REQUIRED)
message("Found Python version ${Python_VERSION}")

if(NOT APPLE)
    set(__libtype "SHARED")
else()
    set(__libtype "MODULE")
endif()

add_einsums_pymod(${__core_name} ${__libtype} MODULENAME core
    SOURCES
        PyMain.cpp
        PyTensorAlgebra.cpp
        PyTensors.cpp
    PUBLIC_DEPENDS einsums
)

if(EINSUMS_BUILD_HIP)
add_einsums_pymod(${__except_name} ${__libtype} MODULENAME gpu_except
    SOURCES
        PyGpuExceptions.hip
    PUBLIC_DEPENDS einsums
)
endif()

extend_einsums_target(${__core_name}
    SOURCES
        PyGPUView.hip
        PyTensorAlgebra.hip
    CONDITION
        EINSUMS_BUILD_HIP
)

extend_einsums_target(${__core_name}
    SOURCES
        PyTesting.cpp
    CONDITION
        EINSUMS_ENABLE_TESTING
)

extend_einsums_target(${__core_name}
    SOURCES
        PyTesting.hip
    CONDITION
        EINSUMS_ENABLE_TESTING AND EINSUMS_BUILD_HIP
)

# All need to be LIBRARY_BASE_PATH because Windows thinks this is a runtime artifact, not a library.
install(
    TARGETS ${__core_name} 
    EXPORT primary_set
    RUNTIME DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
    LIBRARY DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
    ARCHIVE DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
)

if(EINSUMS_BUILD_HIP)
    # All need to be LIBRARY_BASE_PATH because Windows thinks this is a runtime artifact, not a library.
    install(
        TARGETS ${__except_name} 
        EXPORT primary_set
        RUNTIME DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
        LIBRARY DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
        ARCHIVE DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
    )
endif()