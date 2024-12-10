//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

template <typename T>
void dot_test() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    using namespace einsums::linear_algebra;

    size_t i_{10}, j_{10}, a_{10}, b_{10};

    SECTION("1") {
        auto         A = create_random_tensor<T>("A", i_);
        auto         B = create_random_tensor<T>("B", i_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i}, A, Indices{i}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("2") {
        auto         A = create_random_tensor<T>("A", i_, j_);
        auto         B = create_random_tensor<T>("B", i_, j_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j}, A, Indices{i, j}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("3") {
        auto         A = create_random_tensor<T>("A", i_, j_, a_);
        auto         B = create_random_tensor<T>("B", i_, j_, a_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a}, A, Indices{i, j, a}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("4") {
        auto         A = create_random_tensor<T>("A", i_, j_, a_, b_);
        auto         B = create_random_tensor<T>("B", i_, j_, a_, b_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a, b}, A, Indices{i, j, a, b}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }
}

TEST_CASE("dot") {
    SECTION("float") {
        dot_test<float>();
    }
    SECTION("double") {
        dot_test<double>();
    }
    SECTION("cfloat") {
        dot_test<std::complex<float>>();
    }
    SECTION("cdouble") {
        dot_test<std::complex<double>>();
    }
}

TEMPLATE_TEST_CASE("Dot TensorView and Tensor", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    auto const i_ = 10, j_ = 10, k_ = 10, l_ = 2;

    auto A  = create_random_tensor<TestType>("A", i_, k_);
    auto B  = create_random_tensor<TestType>("B", k_, j_);
    auto C  = create_tensor<TestType>("C", l_, j_);
    auto C0 = create_tensor<TestType>("C0", l_, j_);
    zero(C0);

    auto A_view = A(Range{0, l_}, All); // (l_, k_)

    einsum(Indices{l, j}, &C, Indices{l, k}, A_view, Indices{k, j}, B);

    for (size_t l = 0; l < l_; l++) {
        for (size_t j = 0; j < j_; j++) {
            for (size_t k = 0; k < k_; k++) {
                C0(l, j) += A(l, k) * B(k, j);
            }
        }
    }

    for (size_t l = 0; l < l_; l++) {
        for (size_t j = 0; j < j_; j++) {
            // println("{:20.14f} {:20.14f} {:20.14f}", C(l, j), C0(l, j), std::abs(C(l, j) - C0(l, j)));
            // REQUIRE_THAT(C(l, j), Catch::Matchers::WithinAbs(C0(l, j), 1e-12));
            CheckWithinRel(C(l, j), C0(l, j), 0.0001);
        }
    }
}
