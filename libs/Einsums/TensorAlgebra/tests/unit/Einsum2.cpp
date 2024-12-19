//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("einsum TensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> [2,:,:]") {
        // Description: Obtain view [4,:,:] (3x3 view) perform GEMM and store result into
        // view [2,:,:] (3x3 view)
        size_t const                d1_size = 7, d2_size = 3, d3_size = 3;
        size_t const                d1 = 4;
        size_t const                e1 = 2;
        std::array<size_t, 6> const untouched1{0, 1, 3, 4, 5, 6};

        Tensor original = create_random_tensor<TestType>("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = static_cast<TestType>(ij);
            }
        }

        Tensor copy = original;

        // Obtain a 3x3 view of original[4,:,:]
        TensorView view = original(d1, All, All);

        // Obtain a 3x3 view of original[2,:,:] to store the result
        TensorView result = original(e1, All, All);

        // false, false
        {
            // einsum("ik=ij,jk", &result, view, view);
            REQUIRE_NOTHROW(
                einsum(Indices{index::i, index::k}, &result, Indices{index::i, index::j}, view, Indices{index::j, index::k}, view));
            // gemm<false, false>(1.0, view, view, 0.0, &result);

            // Test against the view
            REQUIRE(result(0, 0) == TestType{30.0});
            REQUIRE(result(0, 1) == TestType{36.0});
            REQUIRE(result(0, 2) == TestType{42.0});
            REQUIRE(result(1, 0) == TestType{66.0});
            REQUIRE(result(1, 1) == TestType{81.0});
            REQUIRE(result(1, 2) == TestType{96.0});
            REQUIRE(result(2, 0) == TestType{102.0});
            REQUIRE(result(2, 1) == TestType{126.0});
            REQUIRE(result(2, 2) == TestType{150.0});

            // Test the position in the original
            REQUIRE(original(2, 0, 0) == TestType{30.0});
            REQUIRE(original(2, 0, 1) == TestType{36.0});
            REQUIRE(original(2, 0, 2) == TestType{42.0});
            REQUIRE(original(2, 1, 0) == TestType{66.0});
            REQUIRE(original(2, 1, 1) == TestType{81.0});
            REQUIRE(original(2, 1, 2) == TestType{96.0});
            REQUIRE(original(2, 2, 0) == TestType{102.0});
            REQUIRE(original(2, 2, 1) == TestType{126.0});
            REQUIRE(original(2, 2, 2) == TestType{150.0});

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
    }
}
