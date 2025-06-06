//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Transpose", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace index;

    int numThreads = 1;
    int beta       = 0.;
    int alpha      = 2.;
    SECTION("Transpose 2x7 -> 2x7") {
        int dim      = 2;
        int size_[2] = {2, 7};
        int perm_[2] = {0, 1};

        Tensor<TestType, 2> A("A", 2, 7);
        A.vector_data() = einsums::VectorData<TestType>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0};

        Tensor<TestType, 2> B_hptt("B_hptt", 2, 7);
        B_hptt.vector_data() = einsums::VectorData<TestType>{13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

        auto plan = hptt::create_plan(perm_, dim, alpha, A.data(), size_, nullptr, beta, B_hptt.data(), nullptr, hptt::ESTIMATE, numThreads,
                                      nullptr, true);

        plan->execute();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 7; j++) {
                REQUIRE(B_hptt(i, j) == (TestType)2 * A(i, j));
            }
        }
    }

    SECTION("Transpose 7x2 -> 2x7") {
        int dim      = 2;
        int size_[2] = {7, 2};
        int perm_[2] = {1, 0};

        Tensor<TestType, 2> A("A", 7, 2);
        A.vector_data() = einsums::VectorData<TestType>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0};

        Tensor<TestType, 2> B_hptt("B_hptt", 2, 7);
        B_hptt.vector_data() = einsums::VectorData<TestType>{0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0};

        auto plan = hptt::create_plan(perm_, dim, alpha, A.data(), size_, nullptr, beta, B_hptt.data(), nullptr, hptt::ESTIMATE, numThreads,
                                      nullptr, true);

        plan->execute();

        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 2; j++) {
                REQUIRE(B_hptt(j, i) == (TestType)2 * A(i, j));
            }
        }
    }

    beta = 3.0;
    SECTION("Transpose 2x5x5 -> 5x2x5") {
        int dim      = 3;
        int size_[3] = {2, 5, 5};
        int perm_[3] = {1, 0, 2};

        Tensor<TestType, 3> A("A", 2, 5, 5);
        A.vector_data() = einsums::VectorData<TestType>{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                                        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                                        26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
                                                        39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0};

        Tensor<TestType, 3> B_hptt("B_hptt", 5, 2, 5);
        B_hptt.vector_data() = einsums::VectorData<TestType>{0.0,  1.0,  2.0,  3.0,  4.0,  25.0, 26.0, 27.0, 28.0, 29.0, 5.0,  6.0,  7.0,
                                                             8.0,  9.0,  30.0, 31.0, 32.0, 33.0, 34.0, 10.0, 11.0, 12.0, 13.0, 14.0, 35.0,
                                                             36.0, 37.0, 38.0, 39.0, 15.0, 16.0, 17.0, 18.0, 19.0, 40.0, 41.0, 42.0, 43.0,
                                                             44.0, 20.0, 21.0, 22.0, 23.0, 24.0, 45.0, 46.0, 47.0, 48.0, 49.0};

        auto plan = hptt::create_plan(perm_, dim, alpha, A.data(), size_, nullptr, beta, B_hptt.data(), nullptr, hptt::ESTIMATE, numThreads,
                                      nullptr, true);

        plan->execute();

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 5; k++) {
                    REQUIRE(B_hptt(i, j, k) == (TestType)5 * A(j, i, k));
                }
            }
        }
    }

    SECTION("Transpose 5x2x5 -> 5x5x2 with offsets") {
        int dim            = 3;
        int size_[3]       = {3, 2, 2};
        int perm_[3]       = {2, 0, 1};
        int outerSizeA_[3] = {5, 2, 5};
        int outerSizeB_[3] = {5, 5, 2};
        int offsetA_[3]    = {1, 0, 1};
        int offsetB_[3]    = {2, 2, 0};

        Tensor<TestType, 3> A("A", 5, 2, 5);
        A.vector_data() = einsums::VectorData<TestType>{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,  2.0,  0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0,
            0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        Tensor<TestType, 3> B_hptt("B_hptt", 5, 5, 2);
        B_hptt.vector_data() = einsums::VectorData<TestType>{
            0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            3.0, 5.0, 7.0, 9.0, 11.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        auto plan = hptt::create_plan(perm_, dim, alpha, A.data(), size_, outerSizeA_, offsetA_, beta, B_hptt.data(), outerSizeB_, offsetB_,
                                      hptt::ESTIMATE, numThreads, nullptr, true);

        plan->execute();

        for (int i = 2; i < 4; i++) {
            for (int j = 2; j < 5; j++) {
                for (int k = 0; k < 2; k++) {
                    REQUIRE(B_hptt(i, j, k) == (TestType)5 * A(j - 1, k, i - 1));
                }
            }
        }
    }

    SECTION("Transpose 3x3 in 5x5x2 -> 5x2x3 with offsets (1, 1, 0 and 2, 1, 0) and innerStrides") {
        int dim            = 2;
        int size_[2]       = {3, 3};
        int perm_[2]       = {1, 0};
        int outerSizeA_[2] = {5, 5};
        int outerSizeB_[2] = {5, 6};
        int offsetA_[2]    = {1, 1};
        int offsetB_[2]    = {2, 0};
        int innerStrideA   = 2;
        int innerStrideB   = 1;

        Tensor<TestType, 3> A("A", 5, 5, 2); // 5x5x2
        A.vector_data() = einsums::VectorData<TestType>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0,
                                                        0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0,
                                                        8.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        auto A_view     = TensorView<TestType, 2>{A, Dim<2>{3, 3}, Count<3>{3, 3, 1}, Offset<3>{1, 1, 0}, Stride<2>{10, 2}};

        Tensor<TestType, 3> B_hptt("B_hptt", 5, 2, 3); // 5x2x3
        B_hptt.vector_data() = einsums::VectorData<TestType>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                             1.0, 4.0, 7.0, 0.0, 0.0, 0.0, 2.0, 5.0, 8.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0};
        auto B_hptt_view     = TensorView<TestType, 2>{B_hptt, Dim<2>{3, 3}, Count<3>{3, 1, 3}, Offset<3>{2, 1, 0}, Stride<2>{6, 1}};

        auto plan =
            hptt::create_plan(perm_, dim, alpha, A_view.full_data(), size_, outerSizeA_, offsetA_, innerStrideA, beta,
                              B_hptt_view.full_data(), outerSizeB_, offsetB_, innerStrideB, hptt::ESTIMATE, numThreads, nullptr, true);

        plan->execute();

        for (int i = 2; i < 5; i++) {
            for (int j = 1; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE(B_hptt(i, j, k) == (TestType)5 * A(k + 1, i - 1, j - 1));
                }
            }
        }
    }
}