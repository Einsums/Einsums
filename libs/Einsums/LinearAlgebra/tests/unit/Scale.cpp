//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Scale Row", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    Tensor I_original = create_random_tensor<TestType>("I", 3, 3);
    Tensor I_copy     = I_original;

    scale_row(1, 2.0, &I_copy);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == TestType{2.0} * I_original(1, 0));
    REQUIRE(I_copy(1, 1) == TestType{2.0} * I_original(1, 1));
    REQUIRE(I_copy(1, 2) == TestType{2.0} * I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == I_original(2, 1));
    REQUIRE(I_copy(2, 2) == I_original(2, 2));
}

TEMPLATE_TEST_CASE("Scale Column", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    Tensor I_original = create_random_tensor<TestType>("I", 3, 3);
    Tensor I_copy     = I_original;

    scale_column(1, 2.0, &I_copy);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == TestType{2.0} * I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == TestType{2.0} * I_original(1, 1));
    REQUIRE(I_copy(1, 2) == I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == TestType{2.0} * I_original(2, 1));
    REQUIRE(I_copy(2, 2) == I_original(2, 2));
}

TEMPLATE_TEST_CASE("Scale Row TensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    Tensor     I_original = create_random_tensor<TestType>("I", 3, 3);
    Tensor     I_copy     = I_original;
    TensorView I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    scale_row(1, 2.0, &I_view);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == I_original(1, 1));
    REQUIRE(I_copy(1, 2) == I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == TestType{2.0} * I_original(2, 1));
    REQUIRE(I_copy(2, 2) == TestType{2.0} * I_original(2, 2));

    REQUIRE(I_view(0, 0) == I_original(1, 1));
    REQUIRE(I_view(0, 1) == I_original(1, 2));
    REQUIRE(I_view(1, 0) == TestType{2.0} * I_original(2, 1));
    REQUIRE(I_view(1, 1) == TestType{2.0} * I_original(2, 2));
}

TEMPLATE_TEST_CASE("Scale Column TensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    Tensor     I_original = create_random_tensor<TestType>("I", 3, 3);
    Tensor     I_copy     = I_original;
    TensorView I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    scale_column(1, 2.0, &I_view);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == I_original(1, 1));
    REQUIRE(I_copy(1, 2) == TestType{2.0} * I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == I_original(2, 1));
    REQUIRE(I_copy(2, 2) == TestType{2.0} * I_original(2, 2));

    REQUIRE(I_view(0, 0) == I_original(1, 1));
    REQUIRE(I_view(0, 1) == TestType{2.0} * I_original(1, 2));
    REQUIRE(I_view(1, 0) == I_original(2, 1));
    REQUIRE(I_view(1, 1) == TestType{2.0} * I_original(2, 2));
}
