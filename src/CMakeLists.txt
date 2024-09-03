find_package(range-v3)
find_package(fmt 11)

add_einsums_library(
        einsums
        SOURCES Blas.cpp
        Decomposition.cpp
        ElementOperations.cpp
        Exception.cpp
        InitializeFinalize.cpp
        LinearAlgebra.cpp
        Memory.cpp
        Print.cpp
        Section.cpp
        State.cpp
        TensorAlgebra.cpp
        Timer.cpp
        polynomial/Laguerre.cpp
        PUBLIC_INCLUDES include
        DEPENDS tgt::lapack
        PUBLIC_DEPENDS range-v3::range-v3 fmt::fmt Einsums::h5cpp
)
target_compile_features(einsums PUBLIC cxx_std_20)

pybind11_add_module(
    einsums_py
    PyMain.cpp
    PyTensorAlgebra.cpp
)

extend_einsums_target(
        einsums
        SOURCES _GPUUtils.hip backends/linear_algebra/hipblas/hipblas.hip backends/linear_algebra/hipblas/hipsolver.hip GPULinearAlgebra.hip GPUTimer.hip
        CONDITION EINSUMS_BUILD_HIP PUBLIC_DEPENDS roc::hipblas roc::hipsolver hip::host hip::device
)

# HIP's amd_comgr needs ncurses. Ncurses needs tinfo. Neither target adds the appropriate dependencies or paths. If this gets fixed, this can be removed.
if(EINSUMS_BUILD_HIP)
    target_compile_options(einsums PUBLIC ${CURSES_CFLAGS})
    target_include_directories(einsums PUBLIC ${CURSES_INCLUDE_DIR})
    list(GET CURSES_LIBRARIES 0 __first)
    cmake_path(GET __first PARENT_PATH CURSES_LIB_DIR)
    message(STATUS "Curses directory: ${CURSES_LIB_DIR}")
    target_link_directories(einsums BEFORE PUBLIC ${CURSES_LIB_DIR})
    target_link_libraries(einsums PUBLIC ${CURSES_LIBRARIES})
    if(EINSUMS_USE_HPTT)
        message("Librett populated: ${librett_POPULATED}")
        message("Librett source: ${librett_SOURCE_DIR}")
        #target_include_directories(einsums PUBLIC ${LIBRETT_INCLUDE_DIRS})
        target_include_directories(einsums PRIVATE ${librett_SOURCE_DIR}/src)
    endif()
endif()

if(MSVC)
    target_compile_definitions(einsums PUBLIC _USE_MATH_DEFINES)
    target_compile_options(einsums PUBLIC "/EHsc")
endif()

extend_einsums_target(
        einsums
        SOURCES backends/onemkl/onemkl.cpp DEPENDS Intel::SYCL
        CONDITION TARGET Intel::SYCL
)

extend_einsums_target(
        einsums
        SOURCES OpenMP.c
        CONDITION NOT TARGET OpenMP::OpenMP_CXX
)

extend_einsums_target(einsums PUBLIC_DEPENDS OpenMP::OpenMP_CXX CONDITION TARGET OpenMP::OpenMP_CXX)

extend_einsums_target(
        einsums
        CONDITION TARGET FFT::FFT
        SOURCES FFT.cpp
)

extend_einsums_target(
        einsums
        CONDITION (TARGET FFT::FFT AND EINSUMS_FFT_LIBRARY MATCHES fftw3)
        DEFINES EINSUMS_HAVE_FFT_LIBRARY_FFTW3 DEPENDS FFT::FFT
        SOURCES backends/fft/fftw3/fft.cpp
        FEATURE_INFO "FFTW3 backend"
)

extend_einsums_target(
        einsums PUBLIC_DEFINES EINSUMS_BLAS_INTERFACE_ILP64
        CONDITION EINSUMS_LINALG_INT_INTERFACE STREQUAL ilp64
        FEATURE_INFO "BLAS ILP64 (64-bit integers)"
)

extend_einsums_target(
        einsums PUBLIC_DEFINES EINSUMS_BLAS_INTERFACE_LP64
        CONDITION EINSUMS_LINALG_INT_INTERFACE STREQUAL lp64
        FEATURE_INFO "LP64 (32-bit integers)"
)

extend_einsums_target(
        einsums
        CONDITION (EINSUMS_LINALG_VENDOR STREQUAL MKL
        AND TARGET FFT::FFT
        AND EINSUMS_FFT_LIBRARY STREQUAL mkl)
        DEFINES EINSUMS_HAVE_FFT_LIBRARY_MKL
        SOURCES backends/fft/mkl/fft.cpp backends/fft/fftw3/fft.cpp
        FEATURE_INFO "MKL FFT+FFTW3 backends"
)

extend_einsums_target(
        einsums
        SOURCES backends/linear_algebra/hipblas/hipblas.hip backends/linear_algebra/hipblas/hipsolver.hip
        CONDITION EINSUMS_BUILD_HIP PUBLIC_DEPENDS ${hipblas_LIBRARIES}
        FEATURE_INFO "HIP GPU linear algebra backend"
)

extend_einsums_target(
        einsums
        SOURCES backends/linear_algebra/vendor/Vendor.cpp
        DEFINES EINSUMS_LINEAR_ALGEBRA_NAMESPACE=vendor
        CONDITION NOT EINSUMS_LINALG_VENDOR STREQUAL HIP
        FEATURE_INFO "Generic vendor backend"
)

extend_einsums_target(
        einsums
        SOURCES Sort.cpp PUBLIC_DEFINES EINSUMS_USE_HPTT DEPENDS hptt::hptt
        CONDITION EINSUMS_USE_HPTT
        FEATURE_INFO "HPTT backend"
)

extend_einsums_target(
        einsums
        SOURCES DeviceSort.hip
        DEPENDS librett::librett
        CONDITION (EINSUMS_USE_HPTT AND EINSUMS_BUILD_HIP)
        FEATURE_INFO "Librett backend"
)

extend_einsums_target(
        einsums DEPENDS ittnotify
        DEFINES HAVE_ITTNOTIFY
        CONDITION TARGET ittnotify
        FEATURE_INFO "VTune Instrumentation and Tracing Technology API"
)

extend_einsums_target(
        einsums PUBLIC_OPTIONS -ffp-model=precise -Rno-debug-disables-optimization CONDITION (UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES Intel OR CMAKE_CXX_COMPILIER_ID MATCHES IntelLLVM)
)
)

list(
        APPEND
        defines
    EINSUMS_CONTINUOUSLY_TEST_EINSUM
    EINSUMS_TEST_EINSUM_ABORT
    EINSUMS_WARN_NOT_IMPLEMENTED
    EINSUMS_SHOW_WARNING
    EINSUMS_TEST_NANS
    EINSUMS_RUNTIME_INDICES_CHECK
)
foreach(define IN LISTS defines)
    extend_einsums_target(
            einsums
            DEFINES ${define}
            CONDITION ${define}
    )
endforeach()

# <<<  Install & Export  >>>

include(CMakePackageConfigHelpers)

set_target_properties(einsums PROPERTIES ${ein}_VERSION ${Einsums_VERSION})
set(export_properties "Einsums_VERSION")
set_property(
        TARGET einsums
        APPEND
        PROPERTY EXPORT_PROPERTIES "${export_properties}"
)

install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        TYPE INCLUDE
        COMPONENT ${ein}_Development
        FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

set(tgts einsums Einsums_h5cpp)
if (TARGET hptt::hptt)
    list(APPEND tgts hptt)
endif()

install(
        TARGETS ${tgts}
        EXPORT primary_set
        RUNTIME COMPONENT ${ein}_Runtime
        LIBRARY COMPONENT ${ein}_Runtime NAMELINK_COMPONENT ${ein}_Development
        ARCHIVE COMPONENT ${ein}_Development
        PUBLIC_HEADER COMPONENT ${ein}_Development
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
        EXPORT primary_set
        FILE "${ein}Targets-CXX.cmake"
        NAMESPACE "${ein}::"
        DESTINATION ${EINSUMS_INSTALL_CMAKEDIR}
        COMPONENT ${ein}_Development
)

configure_package_config_file(
        ../cmake/${ein}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${ein}Config.cmake"
        INSTALL_DESTINATION ${EINSUMS_INSTALL_CMAKEDIR}
        NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${ein}ConfigVersion.cmake
        VERSION ${Einsums_VERSION}
        COMPATIBILITY SameMajorVersion
)
install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/${ein}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${ein}ConfigVersion.cmake
        DESTINATION ${EINSUMS_INSTALL_CMAKEDIR}
        COMPONENT ${ein}_Development
)

install(
        FILES ../cmake/FindFFTW.cmake ../cmake/FindTargetHDF5.cmake
        DESTINATION ${EINSUMS_INSTALL_CMAKEDIR}
        COMPONENT ${ein}_Development
)

# include(JoinPaths) join_paths(libdir_for_pc_file "\${exec_prefix}" "${CMAKE_INSTALL_LIBDIR}") join_paths(includedir_for_pc_file "\${prefix}" "${CMAKE_INSTALL_INCLUDEDIR}")
#
# configure_file(cmake/einsums.pc.cmake.in einsums.pc @ONLY) install( FILES ${CMAKE_CURRENT_BINARY_DIR}/einsums.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig/ COMPONENT
# ${ein}_Development )
