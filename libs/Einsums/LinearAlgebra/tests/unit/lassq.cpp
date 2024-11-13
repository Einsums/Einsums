//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

template <typename T>
void lassq_test() {
    auto A       = create_random_tensor<T>("a", 10);
    auto scale   = RemoveComplexT<T>{1.0};
    auto result  = RemoveComplexT<T>{0.0};
    auto result0 = RemoveComplexT<T>{0.0};

    linear_algebra::sum_square(A, &scale, &result);

    for (int i = 0; i < 10; i++) {
        if constexpr (IsComplexV<T>) {
            result0 += A(i).real() * A(i).real() + A(i).imag() * A(i).imag();
        } else {
            result0 += A(i) * A(i);
        }
    }

    CHECK_THAT(result, Catch::Matchers::WithinAbs(result0, 0.00001));
}

TEMPLATE_TEST_CASE("sum_square", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    lassq_test<TestType>();
}
