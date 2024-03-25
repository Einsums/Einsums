add_executable(test-all
    Common/main.cpp
    Disk.cpp
    Decomposition.cpp
    LinearAlgebra.cpp
    Other.cpp
    Polynomial.cpp
    Tensor.cpp
    TensorAlgebra.cpp
    TensorAlgebraTasks.cpp
    BlockTensor.cpp
    SymmTensor.cpp
)

if(EINSUMS_COVERAGE)
    target_compile_options(test-all PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping>)
    target_compile_options(test-all PUBLIC $<$<COMPILE_LANGUAGE:HIP>:-fprofile-instr-generate -fcoverage-mapping>)
    target_compile_options(test-all PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU>:--coverage>)
endif()

target_compile_options(einsums BEFORE PUBLIC $<$<CONFIG:Debug>:-gdwarf-2>)

extend_einsums_target(test-all
    SOURCES
        FFT.cpp
    CONDITION
        TARGET FFT::FFT
)

extend_einsums_target(test-all
    SOURCES
        Parallel.cpp
    CONDITION
        TARGET MPI::MPI_CXX
)

extend_einsums_target(test-all
    SOURCES
        DeviceTensor_extras.hip
        DeviceTensor.hip
        DeviceTensorView.hip
        GPULinearAlgebra.hip
        GPU.hip
        GPUTensorAlgebraTasks.hip
    CONDITION
        EINSUMS_BUILD_HIP
    PUBLIC_DEPENDS
        roc::hipblas
        roc::hipsolver
        hip::host
        hip::device
)

set_source_files_properties(TensorAlgebra_Jobs.cpp PROPERTIES COMPILE_OPTIONS "-Werror")

# Enable einsum auto test capabilities
target_compile_definitions(test-all 
    PRIVATE 
        EINSUMS_USE_CATCH2 
        EINSUMS_CONTINUOUSLY_TEST_EINSUM
        TEST_PATH="${CMAKE_CURRENT_LIST_DIR}"    
)
find_package(Catch2)

target_link_libraries(test-all PUBLIC Catch2::Catch2 einsums)

include(CTest)
include(Catch)
catch_discover_tests(test-all)
