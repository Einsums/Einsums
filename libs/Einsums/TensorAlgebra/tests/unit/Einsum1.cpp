//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("einsum1", "[tensor]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    SECTION("ik=ij,jk") {
        Tensor A{"A", 3, 3};
        Tensor B{"B", 3, 3};
        Tensor C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
                B(i, j) = ij;
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B));

        // println(A);
        // println(B);
        // println(C);

        /*[[ 30,  36,  42],
           [ 66,  81,  96],
           [102, 126, 150]]*/
        REQUIRE(C(0, 0) == 30.0);
        REQUIRE(C(0, 1) == 36.0);
        REQUIRE(C(0, 2) == 42.0);
        REQUIRE(C(1, 0) == 66.0);
        REQUIRE(C(1, 1) == 81.0);
        REQUIRE(C(1, 2) == 96.0);
        REQUIRE(C(2, 0) == 102.0);
        REQUIRE(C(2, 1) == 126.0);
        REQUIRE(C(2, 2) == 150.0);
    }

    SECTION("il=ijk,jkl") {
        Tensor A{"A", 3, 3, 3};
        Tensor B{"B", 3, 3, 3};
        Tensor C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                    B(i, j, k) = ij;
                }
            }
        }

        // println(A);
        // println(B);
        // println(C);

        // einsum("il=ijk,jkl", &C, A, B);
        REQUIRE_NOTHROW(
            einsum(Indices{index::i, index::l}, &C, Indices{index::i, index::j, index::k}, A, Indices{index::j, index::k, index::l}, B));

        // println(C);

        // array([[ 765.,  810.,  855.],
        //        [1818., 1944., 2070.],
        //        [2871., 3078., 3285.]])
        REQUIRE(C(0, 0) == 765.0);
        REQUIRE(C(0, 1) == 810.0);
        REQUIRE(C(0, 2) == 855.0);
        REQUIRE(C(1, 0) == 1818.0);
        REQUIRE(C(1, 1) == 1944.0);
        REQUIRE(C(1, 2) == 2070.0);
        REQUIRE(C(2, 0) == 2871.0);
        REQUIRE(C(2, 1) == 3078.0);
        REQUIRE(C(2, 2) == 3285.0);
    }
}

TEMPLATE_TEST_CASE("TensorView einsum", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // Test if everything passed to einsum is a TensorView.
    Tensor     A = create_random_tensor<TestType>("A", 3, 5);
    Tensor     B = create_random_tensor<TestType>("B", 3, 5);
    TensorView A_view{A, Dim<2>{3, 3}};
    TensorView B_view{B, Dim<2>{3, 3}, Offset<2>{0, 2}};

    Tensor C = create_tensor<TestType>("C2", 10, 10);
    C.zero();
    TensorView C_view{C, Dim<2>{3, 3}, Offset<2>{5, 5}};

    // To perform the test we make an explicit copy of the TensorViews into their own Tensors
    Tensor A_copy = create_tensor<TestType>("A copy", 3, 3);
    Tensor B_copy = create_tensor<TestType>("B copy", 3, 3);

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            A_copy(x, y) = A_view(x, y);
            B_copy(x, y) = B_view(x, y);
        }
    }

    // The target solution is determined from not using views
    Tensor C_solution = create_tensor<TestType>("C solution", 3, 3);
    C_solution.zero();
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_solution, Indices{i, k}, A_copy, Indices{j, k}, B_copy));

    // einsum where everything is a TensorView
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_view, Indices{i, k}, A_view, Indices{j, k}, B_view));

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            CheckWithinRel(C_view(x, y), C_solution(x, y), RemoveComplexT<TestType>{0.001});
            // REQUIRE_THAT(C_view(x, y), Catch::Matchers::WithinAbs(C_solution(x, y), 0.001));
            CheckWithinRel(C(x + 5, y + 5), C_solution(x, y), RemoveComplexT<TestType>{0.001});
            // REQUIRE_THAT(C(x + 5, y + 5), Catch::Matchers::WithinAbs(C_solution(x, y), 0.001));
        }
    }
}
