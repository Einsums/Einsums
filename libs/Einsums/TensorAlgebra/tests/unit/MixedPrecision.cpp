//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

template <typename TC, typename TA, typename TB>
void einsum_mixed_test() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    auto const i_ = 10, j_ = 10, k_ = 10;

    auto A  = create_random_tensor<TA>("A", i_, k_);
    auto B  = create_random_tensor<TB>("B", k_, j_);
    auto C  = create_zero_tensor<TC>("C", i_, j_);
    auto C0 = create_zero_tensor<TC>("C0", i_, j_);

    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t i = 0; i < i_; i++) {
        for (size_t j = 0; j < j_; j++) {
            for (size_t k = 0; k < k_; k++) {
                C0(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    for (size_t i = 0; i < i_; i++) {
        for (size_t j = 0; j < j_; j++) {
            // println("{:20.14f} {:20.14f} {:20.14f}", C(i, j), C0(i, j), std::abs(C(i, j) - C0(i, j)));
            CHECK(std::abs(C(i, j) - C0(i, j)) < RemoveComplexT<TC>{1.0E-4});
            // REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(C0(i, j), RemoveComplexT<TC>{1.0E-16}));
        }
    }
}

TEST_CASE("einsum-mixed") {
    SECTION("d-f-d") {
        einsum_mixed_test<double, float, double>();
    }
    SECTION("d-f-f") {
        einsum_mixed_test<double, float, float>();
    }
    SECTION("f-d-d") {
        einsum_mixed_test<float, double, double>();
    }
    SECTION("cd-cd-d") {
        einsum_mixed_test<std::complex<double>, std::complex<double>, double>();
    }
    SECTION("d-d-d") {
        einsum_mixed_test<double, double, double>();
    }
    // VERY SENSITIVE
    // SECTION("cf-cd-f") {
    //     einsum_mixed_test<std::complex<float>, std::complex<float>, std::complex<float>>();
    // }
}
