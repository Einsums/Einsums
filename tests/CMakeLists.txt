add_executable(test-all Common/main.cpp Disk.cpp Tensor.cpp TensorAlgebra.cpp)
target_link_libraries(test-all Catch2::Catch2 EinsumsInCpp)
# catch_discover_tests(test-tensoralgebra
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )
