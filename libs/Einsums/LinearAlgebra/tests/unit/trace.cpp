//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// CP test cases generate some massive intermediates that the auto einsum tests struggle with.
// Undefine the tests and simply use the manual tests listed in this file.
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>


TEMPLATE_TEST_CASE("Trace", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    Tensor<TestType, 2> tensor{"A", 3, 3};

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            A(i, j) = TestType{3.0 * i + j + 1.0};
        }
    }


    REQUIRE_THAT(linear_algebra::trace(tensor), einsums::testing::CheckWithinRel(TestType{13.0}, 1e-6));
}
