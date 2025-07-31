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

    auto evecs = VectorData<T>{{0.8780072708070924, -0.0},
                               {0.17092002693202832, 0.0},
                               {-0.44709012156771644, 0.0},
                               {0.27577083120554646, -0.126535079740327},
                               {0.5188807656355191, -0.09758502001471543},
                               {0.7399311533699682, -0.2857990997993605},
                               {0.0746560486794058, 0.3625865814626488},
                               {-0.06391810417942875, -0.8294219283777633},
                               {0.12217597040819662, 0.3949736931465739}};

    TensorView<T, 2> A{data.data(), Dim<2>{3, 3}, Stride<2>{6, 2}};
    TensorView<T, 2> evec_test(evecs.data(), Dim<2>{3, 3}, Stride<2>{3, 1});

    einsums::linear_algebra::heev(&A, &b);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));

    T div1 = evec_test(0, 0) / A(0, 0), div2 = evec_test(0, 1) / A(0, 1), div3 = evec_test(0, 2) / A(0, 2);

    for (int i = 0; i < 3; i++) {
        CHECK_THAT(std::real(A(i, 0) * div1), Catch::Matchers::WithinAbs(std::real(evec_test(i, 0)), 0.00001));
        CHECK_THAT(std::imag(A(i, 0) * div1), Catch::Matchers::WithinAbs(std::imag(evec_test(i, 0)), 0.00001));
        CHECK_THAT(std::real(A(i, 1) * div2), Catch::Matchers::WithinAbs(std::real(evec_test(i, 1)), 0.00001));
        CHECK_THAT(std::imag(A(i, 1) * div2), Catch::Matchers::WithinAbs(std::imag(evec_test(i, 1)), 0.00001));
        CHECK_THAT(std::real(A(i, 2) * div3), Catch::Matchers::WithinAbs(std::real(evec_test(i, 2)), 0.00001));
        CHECK_THAT(std::imag(A(i, 2) * div3), Catch::Matchers::WithinAbs(std::imag(evec_test(i, 2)), 0.00001));
    }
}

TEMPLATE_TEST_CASE("heev", "[linear-algebra]", std::complex<float>, std::complex<double>) {
    heev_test<TestType>();
    heev_strided_test<TestType>();
}
