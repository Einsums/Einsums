//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

template <NotComplex T>
void norm_test() {

    auto A = create_tensor<T>("a", 9, 9);

    auto temp = VectorData<T>{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,  0.0,  0.0, 4.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0,  6.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,  8.0,  1.0, 1.0, 1.0, 1.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0};

    A.vector_data() = temp;

    T result = linear_algebra::norm(linear_algebra::Norm::One, A);

    CHECK_THAT(result, Catch::Matchers::WithinRel(33.0, 0.1));
}

TEST_CASE("norm") {
    norm_test<float>();
    norm_test<double>();
}
