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

set_source_files_properties(TensorAlgebra_Jobs.cpp PROPERTIES COMPILE_OPTIONS "-Werror")

# Enable einsum auto test capabilities
target_compile_definitions(test-all 
    PRIVATE 
        EINSUMS_USE_CATCH2 
        EINSUMS_CONTINUOUSLY_TEST_EINSUM
        TEST_PATH="${CMAKE_CURRENT_LIST_DIR}"    
)
find_package(Catch2)
target_link_libraries(test-all Catch2::Catch2 einsums)

include(CTest)
include(Catch)
catch_discover_tests(test-all)
