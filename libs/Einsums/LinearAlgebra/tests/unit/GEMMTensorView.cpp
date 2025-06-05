//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("GEMM TensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto I_original = create_tensor<TestType>("I", 3, 3);

    for (int i = 0, ij = 1; i < 3; i++)
        for (int j = 0; j < 3; j++, ij++)
            I_original(i, j) = ij;

    Tensor     I_copy = I_original;
    TensorView I_view{I_copy, Dim{2, 2}, Offset{1, 1}};

    SECTION("Result into 2x2 matrix") {
        auto result = create_tensor<TestType>("result", 2, 2);

        gemm<false, false>(1.0, I_view, I_view, 0.0, &result);

        REQUIRE(result(0, 0) == TestType{73.0});
        REQUIRE(result(0, 1) == TestType{84.0});
        REQUIRE(result(1, 0) == TestType{112.0});
        REQUIRE(result(1, 1) == TestType{129.0});
    }

    SECTION("Result into 2x2 view of matrix") {
        auto result = create_tensor<TestType>("result", 5, 5);
        result.zero();
        TensorView result_view{result, Dim{2, 2}, Offset{3, 2}};

        gemm<false, false>(1.0, I_view, I_view, 0.0, &result_view);

        // Check view
        REQUIRE(result_view(0, 0) == TestType{73.0});
        REQUIRE(result_view(0, 1) == TestType{84.0});
        REQUIRE(result_view(1, 0) == TestType{112.0});
        REQUIRE(result_view(1, 1) == TestType{129.0});

        // Check full
        REQUIRE(result(3, 2) == TestType{73.0});
        REQUIRE(result(3, 3) == TestType{84.0});
        REQUIRE(result(4, 2) == TestType{112.0});
        REQUIRE(result(4, 3) == TestType{129.0});
    }

    SECTION("Transpose") {
        auto result = create_tensor<TestType>("result", 2, 2);

        gemm<false, true>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE(result(0, 0) == TestType{61.0});
        REQUIRE(result(0, 1) == TestType{94.0});
        REQUIRE(result(1, 0) == TestType{94.0});
        REQUIRE(result(1, 1) == TestType{145.0});

        gemm<true, false>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE(result(0, 0) == TestType{89.0});
        REQUIRE(result(0, 1) == TestType{102.0});
        REQUIRE(result(1, 0) == TestType{102.0});
        REQUIRE(result(1, 1) == TestType{117.0});

        gemm<true, true>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE(result(0, 0) == TestType{73.0});
        REQUIRE(result(0, 1) == TestType{112.0});
        REQUIRE(result(1, 0) == TestType{84.0});
        REQUIRE(result(1, 1) == TestType{129.0});
    }
}

TEST_CASE("GEMMSubset TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> 3x3") {
        size_t const d1_size = 7, d2_size = 3, d3_size = 3;
        size_t const d1 = 4;

        Tensor original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = ij;
            }
        }

        // Obtain a 3x3 view of original[4,:,:]
        TensorView view = original(d1, All, All);
        Tensor     result{"result", d2_size, d3_size};

        // false, false
        {
            gemm<false, false>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 36.0);
            REQUIRE(result(0, 2) == 42.0);
            REQUIRE(result(1, 0) == 66.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 96.0);
            REQUIRE(result(2, 0) == 102.0);
            REQUIRE(result(2, 1) == 126.0);
            REQUIRE(result(2, 2) == 150.0);
        }
        // false, true
        {
            gemm<false, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 14.0);
            REQUIRE(result(0, 1) == 32.0);
            REQUIRE(result(0, 2) == 50.0);
            REQUIRE(result(1, 0) == 32.0);
            REQUIRE(result(1, 1) == 77.0);
            REQUIRE(result(1, 2) == 122.0);
            REQUIRE(result(2, 0) == 50.0);
            REQUIRE(result(2, 1) == 122.0);
            REQUIRE(result(2, 2) == 194.0);
        }
        // true, false
        {
            gemm<true, false>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 66.0);
            REQUIRE(result(0, 1) == 78.0);
            REQUIRE(result(0, 2) == 90.0);
            REQUIRE(result(1, 0) == 78.0);
            REQUIRE(result(1, 1) == 93.0);
            REQUIRE(result(1, 2) == 108.0);
            REQUIRE(result(2, 0) == 90.0);
            REQUIRE(result(2, 1) == 108.0);
            REQUIRE(result(2, 2) == 126.0);
        }
        // true, true
        {
            gemm<true, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 66.0);
            REQUIRE(result(0, 2) == 102.0);
            REQUIRE(result(1, 0) == 36.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 126.0);
            REQUIRE(result(2, 0) == 42.0);
            REQUIRE(result(2, 1) == 96.0);
            REQUIRE(result(2, 2) == 150.0);
        }
    }

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> [2,:,:]") {
        // Description:
        // 1. Allocate tensor [7, 3, 3]
        // 2. Obtain view [4,:,:] (3x3 view) of tensor
        // 3. Perform GEMM and store result into view [2,:,:] (3x3 view) of tensor
        // 4. Test correctness of the GEMM result and of the data
        //    elements that should not have been touched.
        size_t const                d1_size = 7, d2_size = 3, d3_size = 3;
        size_t const                d1 = 4;
        size_t const                e1 = 2;
        std::array<size_t, 6> const untouched_d1{0, 1, 3, 4, 5, 6};

        Tensor original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = static_cast<double>(ij);
            }
        }

        Tensor copy = original;

        // Obtain a 3x3 view of original[4,:,:]
        //   A view does not copy data it is just an offset pointer into the original with necessary striding information.
        TensorView view = original(d1, All, All);

        // Obtain a 3x3 view of original[2,:,:] to store the result
        TensorView result = original(e1, All, All);

        // false, false
        {
            // Call BLAS routine passing necessary offset pointer, dimension, and stride information.
            gemm<false, false>(1.0, view, view, 0.0, &result);

            // Test against the view
            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 36.0);
            REQUIRE(result(0, 2) == 42.0);
            REQUIRE(result(1, 0) == 66.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 96.0);
            REQUIRE(result(2, 0) == 102.0);
            REQUIRE(result(2, 1) == 126.0);
            REQUIRE(result(2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
        // false, true
        {
            gemm<false, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 14.0);
            REQUIRE(result(0, 1) == 32.0);
            REQUIRE(result(0, 2) == 50.0);
            REQUIRE(result(1, 0) == 32.0);
            REQUIRE(result(1, 1) == 77.0);
            REQUIRE(result(1, 2) == 122.0);
            REQUIRE(result(2, 0) == 50.0);
            REQUIRE(result(2, 1) == 122.0);
            REQUIRE(result(2, 2) == 194.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
        // true, false
        {
            gemm<true, false>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 66.0);
            REQUIRE(result(0, 1) == 78.0);
            REQUIRE(result(0, 2) == 90.0);
            REQUIRE(result(1, 0) == 78.0);
            REQUIRE(result(1, 1) == 93.0);
            REQUIRE(result(1, 2) == 108.0);
            REQUIRE(result(2, 0) == 90.0);
            REQUIRE(result(2, 1) == 108.0);
            REQUIRE(result(2, 2) == 126.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
        // true, true
        {
            gemm<true, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 66.0);
            REQUIRE(result(0, 2) == 102.0);
            REQUIRE(result(1, 0) == 36.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 126.0);
            REQUIRE(result(2, 0) == 42.0);
            REQUIRE(result(2, 1) == 96.0);
            REQUIRE(result(2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
    }
}
