add_executable(test-all
    Common/main.cpp
    Disk.cpp
    Decomposition.cpp
    LinearAlgebra.cpp
    Other.cpp
    Polynomial.cpp
    Tensor.cpp
    TensorAlgebra.cpp
)

if(EINSUMS_COVERAGE)
    target_compile_options(test-all PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping -Og -g3 -ggdb>)
    target_compile_options(test-all PUBLIC $<$<COMPILE_LANGUAGE:HIP>:-fprofile-instr-generate -fcoverage-mapping -Og -g3 -ggdb>)
    target_compile_options(test-all PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU>:--coverage -Og -g3 -ggdb>)
endif()

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
        TensorAlgebra_Jobs.cpp
    CONDITION
        EINSUMS_JOB_QUEUES
)

extend_einsums_target(test-all
    SOURCES
        GPU.hip
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
