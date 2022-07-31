add_executable(test-all
    Common/main.cpp
    Disk.cpp
    Decomposition.cpp
    LinearAlgebra.cpp
    Tensor.cpp
    TensorAlgebra.cpp
)

# Enable einsum auto test capabilities
target_compile_definitions(test-all PRIVATE EINSUMS_USE_CATCH2 EINSUMS_CONTINUOUSLY_TEST_EINSUM)
target_link_libraries(test-all Catch2::Catch2 einsums)
