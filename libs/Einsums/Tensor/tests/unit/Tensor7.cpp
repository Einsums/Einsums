//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("TensorView Ranges") {
    using namespace einsums;

    SECTION("Subviews") {
        auto                C = einsums::create_random_tensor("C", 3, 3);
        einsums::TensorView viewC(C, einsums::Dim{2, 2}, einsums::Offset{1, 1}, einsums::Stride{3, 1});

        // einsums::println("C strides: %zu %zu\n", C.strides()[0], C.strides()[1]);

        if (C.is_row_major()) {
            REQUIRE(C(1, 1) == viewC(0, 0));
            REQUIRE(C(2, 1) == viewC(1, 0));
            REQUIRE(C(1, 2) == viewC(0, 1));
            REQUIRE(C(2, 2) == viewC(1, 1));
        } else {
            REQUIRE(C(1, 1) == viewC(0, 0));
            REQUIRE(C(1, 2) == viewC(1, 0));
            REQUIRE(C(2, 1) == viewC(0, 1));
            REQUIRE(C(2, 2) == viewC(1, 1));
        }
    }

    SECTION("Subviews 2") {
        auto C = einsums::create_random_tensor("C", 3, 3);
        // std::array<einsums::Range, 2> test;
        einsums::TensorView viewC = C(einsums::Range{1, 3}, einsums::Range{1, 3});

        // einsums::println(C);
        // einsums::println(viewC);

        REQUIRE(C(1, 1) == viewC(0, 0));
        REQUIRE(C(2, 1) == viewC(1, 0));
        REQUIRE(C(1, 2) == viewC(0, 1));
        REQUIRE(C(2, 2) == viewC(1, 1));
    }

    // SECTION("Subviews 3") {
    //     auto C = create_random_tensor("C", 3, 3, 3, 3);
    //     auto viewC = C(0, 0, Range{1, 3}, Range{1, 3});

    //     println(C);
    //     println(viewC);
    // }
}