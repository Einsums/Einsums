//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <type_traits>
#include <utility>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Subset TensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    SECTION("Subset View 7x7[1,:] -> 1x7") {
        size_t const size = 7;
        size_t const row  = 1;

        auto I_original = create_random_tensor<TestType>("Original", size, size);
        auto I_view     = I_original(row, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(row, i) == I_view(i));
        }
    }

    SECTION("Subset View 7x7x7[4,:,:] -> 7x7") {
        size_t const size = 7;
        size_t const d1   = 4;

        auto I_original = create_random_tensor<TestType>("Original", size, size, size);
        auto I_view     = I_original(d1, All, All);

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

        auto I_original = create_random_tensor<TestType>("Original", size, size, size);
        auto I_view     = I_original(d1, d2, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(d1, d2, i) == I_view(i));
        }
    }
}

TEMPLATE_TEST_CASE("Block tensor views", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    BlockTensor<TestType, 2> test("Test", 3, 4, 5);

    test[0] = create_random_tensor<TestType>("1", 3, 3);
    test[1] = create_random_tensor<TestType>("2", 4, 4);
    test[2] = create_random_tensor<TestType>("3", 5, 5);

    SECTION("non-const") {
        auto test_view = (TiledTensorView<TestType, 2>)test;

        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                REQUIRE(test(i, j) == test_view.at(i, j));
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

// Views had no move constructor, so std::vector copied them on growth because the move was not noexcept.
TEST_CASE("TensorView move", "[tensor]") {
    using namespace einsums;

    static_assert(std::is_nothrow_move_constructible_v<TensorView<double, 2>>);
    static_assert(std::is_copy_assignable_v<TensorView<double, 2>>);

    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 4, 3);

    SECTION("construction keeps viewing the same data") {
        TensorView<double, 2> view = A(All, Range{1, 3});
        double const         *data = view.data();
        TensorView<double, 2> moved(std::move(view));

        CHECK(moved.data() == data);
        CHECK(moved.dim(1) == 2);
        CHECK(moved(3, 1) == A(3, 2));
    }

    SECTION("assigning a temporary view writes into the viewed data") {
        TensorView<double, 2> view = B(All, All);
        view                       = A(All, All);

        CHECK(view.data() == B.data());
        CHECK(B(3, 2) == A(3, 2));
        CHECK(B.data() != A.data());
    }
}
