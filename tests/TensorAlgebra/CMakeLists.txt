add_executable(test-tensoralgebra ../Common/main.cpp TensorAlgebra.cpp)
target_link_libraries(test-tensoralgebra Catch2::Catch2 EinsumsInCpp)
# catch_discover_tests(test-tensoralgebra
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )
