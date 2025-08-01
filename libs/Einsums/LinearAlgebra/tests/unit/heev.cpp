//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

template <Complex T>
void heev_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A = create_tensor<T>("a", 3, 3);
    auto b = create_tensor<RemoveComplexT<T>>("b", 3);

    A.vector_data() = {{0.199889, 0.0},         {-0.330816, -0.127778},  {-0.0546237, 0.176589}, {-0.330816, 0.127778}, {0.629179, 0.0},
                       {-0.0224813, -0.327171}, {-0.0546237, -0.176589}, {-0.0224813, 0.327171}, {0.170931, 0.0}};

    einsums::linear_algebra::heev(&A, &b);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));
}

template <Complex T>
void heev_strided_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto b = create_tensor<RemoveComplexT<T>>("b", 3);

    auto data = VectorData<T>{{0.199889, 0.0},         {0.0, 0.0}, {-0.330816, -0.127778}, {0.0, 0.0}, {-0.0546237, 0.176589},  {0.0, 0.0},
                              {-0.330816, 0.127778},   {0.0, 0.0}, {0.629179, 0.0},        {0.0, 0.0}, {-0.0224813, -0.327171}, {0.0, 0.0},
                              {-0.0546237, -0.176589}, {0.0, 0.0}, {-0.0224813, 0.327171}, {0.0, 0.0}, {0.170931, 0.0},         {0.0, 0.0}};

    TensorView<T, 2> A{data.data(), Dim<2>{3, 3}, Stride<2>{6, 2}};

    Tensor<T, 2> A2{"test", 3, 3};

    A2 = A;

    println(A);
    println(A2);

    einsums::linear_algebra::heev(&A2, &b);

    println(b);

    einsums::linear_algebra::heev(&A, &b);

    println(A);
    println(A2);

    println(b);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));

    T div1 = A2(0, 0) / A(0, 0), div2 = A2(0, 1) / A(0, 1), div3 = A2(0, 2) / A(0, 2);

    for (int i = 0; i < 3; i++) {
        CHECK_THAT(std::real(A(i, 0) * div1), Catch::Matchers::WithinAbs(std::real(A2(i, 0)), 0.00001));
        CHECK_THAT(std::imag(A(i, 0) * div1), Catch::Matchers::WithinAbs(std::imag(A2(i, 0)), 0.00001));
        CHECK_THAT(std::real(A(i, 1) * div2), Catch::Matchers::WithinAbs(std::real(A2(i, 1)), 0.00001));
        CHECK_THAT(std::imag(A(i, 1) * div2), Catch::Matchers::WithinAbs(std::imag(A2(i, 1)), 0.00001));
        CHECK_THAT(std::real(A(i, 2) * div3), Catch::Matchers::WithinAbs(std::real(A2(i, 2)), 0.00001));
        CHECK_THAT(std::imag(A(i, 2) * div3), Catch::Matchers::WithinAbs(std::imag(A2(i, 2)), 0.00001));
    }
}

TEMPLATE_TEST_CASE("heev", "[linear-algebra]", std::complex<float>, std::complex<double>) {
    heev_test<TestType>();
    heev_strided_test<TestType>();
}
