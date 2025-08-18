//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// CP test cases generate some massive intermediates that the auto einsum tests struggle with.
// Undefine the tests and simply use the manual tests listed in this file.
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Lyapunov") {
    using namespace einsums;
    using namespace einsums::linear_algebra;
    // Solves for X, where
    // AX + XA^T = Q

    auto A = create_tensor<double>("A", 3, 3);
    auto Q = create_tensor<double>("Q", 3, 3);

    A.vector_data() = {1.25898804, -0.00000000, -0.58802280, -0.00000000, 1.51359048,
                                                  0.00000000, -0.58802280, 0.00000000,  1.71673427};

    Q.vector_data() = {-0.05892104, 0.00000000, 0.00634896, 0.00000000, -0.02508491,
                                                  0.00000000,  0.00634896, 0.00000000, 0.00155829};

    auto X = einsums::linear_algebra::solve_continuous_lyapunov(A, Q);

    auto Qtest = einsums::linear_algebra::gemm<false, false>(1.0, A, X);
    auto Q2    = einsums::linear_algebra::gemm<false, true>(1.0, X, A);
    einsums::linear_algebra::axpy(1.0, Q2, &Qtest);

    for (size_t i = 0; i < 9; i++) {
        CHECK_THAT(Q.data()[i], Catch::Matchers::WithinAbs(Qtest.data()[i], 0.00001));
    }
}

template <typename T>
void truncated_svd_test() {
    using namespace einsums;

    auto a         = create_random_tensor<T>("a", 10, 10);
    auto [b, c, d] = linear_algebra::truncated_svd(a, 5);
}

TEST_CASE("truncated_svd") {
    SECTION("float") {
        truncated_svd_test<float>();
    }
    SECTION("double") {
        truncated_svd_test<double>();
    }
    SECTION("complex float") {
        truncated_svd_test<std::complex<float>>();
    }
    SECTION("complex double") {
        truncated_svd_test<std::complex<double>>();
    }
}
