add_executable(test-all
    Common/main.cpp
    Disk.cpp
    LinearAlgebra.cpp
    Tensor.cpp
    TensorAlgebra.cpp
)

if (UNIX AND NOT APPLE)
    target_sources(test-all PUBLIC Decomposition.cpp)
endif()

target_link_libraries(test-all Catch2::Catch2 einsums)
