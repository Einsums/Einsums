//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorUtilities/CreateIdentity.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Identity", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    auto I = einsums::create_identity_tensor<TestType>("I", 3, 3);

    REQUIRE(I(0, 0) == TestType{1.0});
    REQUIRE(I(0, 1) == TestType{0.0});
    REQUIRE(I(0, 2) == TestType{0.0});
    REQUIRE(I(1, 0) == TestType{0.0});
    REQUIRE(I(1, 1) == TestType{1.0});
    REQUIRE(I(1, 2) == TestType{0.0});
    REQUIRE(I(2, 0) == TestType{0.0});
    REQUIRE(I(2, 1) == TestType{0.0});
    REQUIRE(I(2, 2) == TestType{1.0});
}

TEMPLATE_TEST_CASE("Identity - 3d", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    auto I = einsums::create_identity_tensor<TestType>("I", 3, 3, 3);

    auto   strides  = I.strides();
    size_t elements = I.size();

    for (size_t item = 0; item < elements; item++) {
        std::array<size_t, 3> index;
        sentinel_to_indices(item, strides, index);

        auto [i, j, k] = index;

        if (i == j && j == k) {
            REQUIRE(I(i, j, k) == TestType{1.0});
        } else {
            REQUIRE(I(i, j, k) == TestType{0.0});
        }
    }
}
