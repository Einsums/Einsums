//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Subset TensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    SECTION("Subset View 7x7[1,:] -> 1x7") {
        size_t const size = 7;
        size_t const row  = 1;

        Tensor     I_original = create_random_tensor("Original", size, size);
        TensorView I_view     = I_original(row, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(row, i) == I_view(i));
        }
    }

    SECTION("Subset View 7x7x7[4,:,:] -> 7x7") {
        size_t const size = 7;
        size_t const d1   = 4;

        Tensor     I_original = create_random_tensor("Original", size, size, size);
        TensorView I_view     = I_original(d1, All, All);

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                REQUIRE(I_original.subscript(d1, i, j) == I_view.subscript(i, j));
            }
        }
    }

    SECTION("Subset View 7x7x7[4,3,:] -> 7") {
        size_t const size = 7;
        size_t const d1   = 4;
        size_t const d2   = 3;

        Tensor     I_original = create_random_tensor("Original", size, size, size);
        TensorView I_view     = I_original(d1, d2, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(d1, d2, i) == I_view(i));
        }
    }
}

TEMPLATE_TEST_CASE("Block tensor views", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    BlockTensor<TestType, 2> test("Test", 3, 4, 5);

    test[0] = create_random_tensor("1", 3, 3);
    test[2] = create_random_tensor("2", 4, 4);
    test[3] = create_random_tensor("3", 5, 5);

    SECTION("non-const") {
        auto test_view = (TiledTensorView<TestType, 2>)test;

        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                REQUIRE(test(i, j) == test_view(i, j));
            }
        }
    }

    SECTION("const") {
        auto const test_view = (TiledTensorView<TestType, 2> const)test;

        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                REQUIRE(test(i, j) == test_view(i, j));
            }
        }
    }
}