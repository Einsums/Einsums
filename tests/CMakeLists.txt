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
        (HAVE_MKL_LAPACKE_H AND HAVE_MKL_CBLAS_H)
)

# Enable einsum auto test capabilities
target_compile_definitions(test-all PRIVATE EINSUMS_USE_CATCH2 EINSUMS_CONTINUOUSLY_TEST_EINSUM)
target_link_libraries(test-all Catch2::Catch2 einsums)
