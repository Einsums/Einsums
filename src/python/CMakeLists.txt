if(__VERSION)
    set(PYBIND11_PYTHON_VERSION ${__VERSION} CACHE STRING "Python version for Pybind11")
    find_package(Python "${__VERSION}" EXACT COMPONENTS Interpreter Development REQUIRED)
    set(__core_name "core-${__VERSION}")
    set(__except_name "gpu_except-${__VERSION}")
else()
    if(NOT EINSUMS_PYBIND11_SUBDIR)
        find_package(Python COMPONENTS Interpreter Development REQUIRED)
    endif()
    set(__core_name "core")
    set(__except_name "gpu_except")
endif()

if(NOT EINSUMS_PYBIND11_SUBDIR)
    find_package(pybind11 CONFIG REQUIRED)
    message("Found Python version ${Python_VERSION}")
endif()

add_einsums_pymod(${__core_name} SHARED MODULENAME core
    SOURCES
        PyMain.cpp
        PyTensorAlgebra.cpp
        PyTensors.cpp
    PUBLIC_DEPENDS einsums
)

if(EINSUMS_BUILD_HIP)
add_einsums_pymod(${__except_name} SHARED MODULENAME gpu_except
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

foreach(define IN LISTS defines)
    extend_einsums_target(
        ${__core_name}
        DEFINES ${define}
        CONDITION ${define}
    )
endforeach()

set_target_properties(einsums PROPERTIES ${ein}_VERSION ${Einsums_VERSION})
set(export_properties "Einsums_VERSION")
set_property(
        TARGET einsums
        APPEND
        PROPERTY EXPORT_PROPERTIES "${export_properties}"
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
    foreach(define IN LISTS defines)
    extend_einsums_target(
        ${__except_name}
        DEFINES ${define}
        CONDITION ${define}
    )
    endforeach()

    # All need to be LIBRARY_BASE_PATH because Windows thinks this is a runtime artifact, not a library.
    install(
        TARGETS ${__except_name} 
        EXPORT primary_set
        RUNTIME DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
        LIBRARY DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
        ARCHIVE DESTINATION ${EINSUMS_LIBRARY_BASE_PATH}/einsums
    )
endif()