//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomSemidefinite.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

template <typename T>
void test_syev() {
    auto A = create_tensor<T>("A", 3, 3);
    auto x = create_tensor<T>("x", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));

    // A.vector_data() = VectorData<T>{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 2.0;
    A(1, 1) = 4.0;
    A(1, 2) = 5.0;
    A(2, 0) = 3.0;
    A(2, 1) = 5.0;
    A(2, 2) = 6.0;

    REQUIRE(A(0, 0) == 1.0);

    einsums::linear_algebra::syev(&A, &x);

    CHECK_THAT(x(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(x(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(x(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));

    auto check = VectorData<double>{-0.73697623, 0.59100905, -0.32798528, -0.32798528, -0.73697623,
                                    -0.59100905, 0.59100905, 0.32798528,  -0.73697623};

    T sign1 = std::copysign(T{1.0}, A(0, 0) / check[0]), sign2 = std::copysign(T{1.0}, A(0, 1) / check[1]),
      sign3 = std::copysign(T{1.0}, A(0, 2) / check[2]);

    CHECK_THAT(A(0, 0) * sign1, Catch::Matchers::WithinRel(check[0], 0.00001));
    CHECK_THAT(A(0, 1) * sign2, Catch::Matchers::WithinRel(check[1], 0.00001));
    CHECK_THAT(A(0, 2) * sign3, Catch::Matchers::WithinRel(check[2], 0.00001));
    CHECK_THAT(A(1, 0) * sign1, Catch::Matchers::WithinRel(check[3], 0.00001));
    CHECK_THAT(A(1, 1) * sign2, Catch::Matchers::WithinRel(check[4], 0.00001));
    CHECK_THAT(A(1, 2) * sign3, Catch::Matchers::WithinRel(check[5], 0.00001));
    CHECK_THAT(A(2, 0) * sign1, Catch::Matchers::WithinRel(check[6], 0.00001));
    CHECK_THAT(A(2, 1) * sign2, Catch::Matchers::WithinRel(check[7], 0.00001));
    CHECK_THAT(A(2, 2) * sign3, Catch::Matchers::WithinRel(check[8], 0.00001));
}

template <typename T>
void test_disk_syev() {
    // constexpr int size = 50;
    // auto             A = create_random_definite<T>("A", size, size);
    // auto             x = create_tensor<T>("x", size);
    // DiskTensor<T, 2> A_disk(fmt::format("/test/syev/{}/A", type_name<T>()), size, size);

    // A_disk.write(A);

    // einsums::linear_algebra::syev(&A, &x);

    // auto [vecs, vals] = einsums::linear_algebra::truncated_syev(A_disk, 5);

    // println(vecs.get());
    // println(A);
    // println(vals);
    // println(x);

    // for (int i = 0; i < vals.dim(0); i++) {
    //     bool found = false;
    //     int  index = 0;

    //     for (int j = 0; j < 10; j++) {
    //         if (std::abs(vals(i) - x(j)) < 1e-6) {
    //             index = j;
    //             found = true;
    //             break;
    //         }
    //     }

    //     REQUIRE(found);
    //     if (found) {
    //         auto vec   = vecs(All, index);
    //         T    scale = A(0, index) / vec.get()(0);

    //         for (int j = 0; j < 10; j++) {
    //             REQUIRE_THAT(scale * vec.get()(j), Catch::Matchers::WithinAbs(A(j, index), 1e-6));
    //         }
    //     }
    // }
}

template <typename T>
void test_strided_syev() {
    auto x = create_tensor<T>("x", 3);

    auto data = VectorData<T>{1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 6.0, 0.0};

    TensorView<T, 2> A{data.data(), Dim<2>{3, 3}, Stride<2>{6, 2}};

    auto data2 = VectorData<T>{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    TensorView<T, 2> A2{data2.data(), Dim<2>{3, 3}, Stride<2>{3, 1}};

    REQUIRE(A.stride(0) == 6);

    REQUIRE(A(1, 1) == 4.0);
    REQUIRE(A(2, 1) == 5.0);

    einsums::linear_algebra::syev<true>(&A2, &x);

    einsums::linear_algebra::syev<true>(&A, &x);

    CHECK_THAT(x(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(x(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(x(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));

    auto check = VectorData<double>{-0.73697623, 0.59100905, -0.32798528, -0.32798528, -0.73697623,
                                    -0.59100905, 0.59100905, 0.32798528,  -0.73697623};

    T sign1 = std::copysign(T{1.0}, A(0, 0) / check[0]), sign2 = std::copysign(T{1.0}, A(0, 1) / check[1]),
      sign3 = std::copysign(T{1.0}, A(0, 2) / check[2]);

    CHECK_THAT(A(0, 0) * sign1, Catch::Matchers::WithinRel(check[0], 0.00001));
    CHECK_THAT(A(0, 1) * sign2, Catch::Matchers::WithinRel(check[1], 0.00001));
    CHECK_THAT(A(0, 2) * sign3, Catch::Matchers::WithinRel(check[2], 0.00001));
    CHECK_THAT(A(1, 0) * sign1, Catch::Matchers::WithinRel(check[3], 0.00001));
    CHECK_THAT(A(1, 1) * sign2, Catch::Matchers::WithinRel(check[4], 0.00001));
    CHECK_THAT(A(1, 2) * sign3, Catch::Matchers::WithinRel(check[5], 0.00001));
    CHECK_THAT(A(2, 0) * sign1, Catch::Matchers::WithinRel(check[6], 0.00001));
    CHECK_THAT(A(2, 1) * sign2, Catch::Matchers::WithinRel(check[7], 0.00001));
    CHECK_THAT(A(2, 2) * sign3, Catch::Matchers::WithinRel(check[8], 0.00001));
}

TEMPLATE_TEST_CASE("syev", "[linear-algebra]", double) {
    test_syev<TestType>();
    test_strided_syev<TestType>();
    test_disk_syev<TestType>();
}

TEMPLATE_TEST_CASE("definite and semidefinite", "[linear-algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("positive definite") {
        auto A = create_random_definite<TestType>("A", size, size);

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        for (int i = 0; i < size; i++) {
            REQUIRE(B(i) >= -WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
    }

    SECTION("negative definite") {
        auto A = create_random_definite<TestType>("A", size, size, TestType{-1.0});

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        for (int i = 0; i < size; i++) {
            REQUIRE(B(i) <= WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
    }

    SECTION("positive semi-definite") {
        auto A = create_random_semidefinite<TestType>("A", size, size);

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        int zeros = 0;
        for (int i = 0; i < size; i++) {
            if (WithinStrict(TestType(0.0), TestType{10000.0}).match(B(i))) {
                zeros++;
            }
            REQUIRE(B(i) >= -WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
        REQUIRE(zeros >= 1);
    }

    SECTION("negative semi-definite") {
        auto A = create_random_semidefinite<TestType>("A", size, size, TestType{-1.0});

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        int zeros = 0;
        for (int i = 0; i < size; i++) {
            if (WithinStrict(TestType(0.0), TestType{10000.0}).match(B(i))) {
                zeros++;
            }
            REQUIRE(B(i) <= WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
        REQUIRE(zeros >= 1);
    }
}
