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
        TARGET Einsums_FFT
)

extend_einsums_target(test-all
    SOURCES
        Parallel.cpp
    CONDITION
        TARGET MPI::MPI_CXX
)

# Enable einsum auto test capabilities
target_compile_definitions(test-all 
    PRIVATE 
        EINSUMS_USE_CATCH2 
        EINSUMS_CONTINUOUSLY_TEST_EINSUM
        TEST_PATH="${CMAKE_CURRENT_LIST_DIR}"    
)
target_link_libraries(test-all Catch2::Catch2 einsums)

# from https://github.com/catchorg/Catch2/blob/devel/docs/migrate-v2-to-v3.md :
# * Catch2::Catch2WithMain target preferred. Catch2::Catch2 if custom main.
# * present setup seems closest to former line:
#   `target_compile_definitions(Catch2::Catch2 INTERFACE CATCH_AMALGAMATED_CUSTOM_MAIN)`
# * they recc replacing the catch_all.hpp with specific ones for test-building speed

include(CTest)
include(Catch)
catch_discover_tests(test-all)
