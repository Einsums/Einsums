//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("einsum1", "[tensor]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    tensor_algebra::detail::AlgorithmChoice alg_choice;

    SECTION("ik=ij,jk") {
        std::vector<Tensor<double, 2>> A, B, C;

        for (int k = 0; k < 10; k++) {
            A.emplace_back("A", 3, 3);
            B.emplace_back("B", 3, 3);
            C.emplace_back("C", 3, 3);
            for (int i = 0, ij = 1; i < 3; i++) {
                for (int j = 0; j < 3; j++, ij++) {
                    A[k](i, j) = ij;
                    B[k](i, j) = ij;
                }
            }
        }

        REQUIRE_NOTHROW(
            einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B, &alg_choice));
        REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

        // println(A);
        // println(B);
        // println(C);

        /*[[ 30,  36,  42],
           [ 66,  81,  96],
           [102, 126, 150]]*/
        for (int k = 0; k < 10; k++) {
            REQUIRE(C[k](0, 0) == 30.0);
            REQUIRE(C[k](0, 1) == 36.0);
            REQUIRE(C[k](0, 2) == 42.0);
            REQUIRE(C[k](1, 0) == 66.0);
            REQUIRE(C[k](1, 1) == 81.0);
            REQUIRE(C[k](1, 2) == 96.0);
            REQUIRE(C[k](2, 0) == 102.0);
            REQUIRE(C[k](2, 1) == 126.0);
            REQUIRE(C[k](2, 2) == 150.0);
        }
    }

    SECTION("il=ijk,jkl") {
        std::vector<Tensor<double, 3>> A, B;
        std::vector<Tensor<double, 2>> C;

        for (int l = 0; l < 10; l++) {
            A.emplace_back("A", 3, 3, 3);
            B.emplace_back("B", 3, 3, 3);
            C.emplace_back("C", 3, 3);
            for (int i = 0, ij = 1; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++, ij++) {
                        A[l](i, j, k) = ij;
                        B[l](i, j, k) = ij;
                    }
                }
            }
        }

        // println(A);
        // println(B);
        // println(C);

        // einsum("il=ijk,jkl", &C, A, B);
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::l}, &C, Indices{index::i, index::j, index::k}, A,
                               Indices{index::j, index::k, index::l}, B, &alg_choice));
        REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

        // println(C);

        // array([[ 765.,  810.,  855.],
        //        [1818., 1944., 2070.],
        //        [2871., 3078., 3285.]])
        for (int l = 0; l < 10; l++) {
            REQUIRE(C[l](0, 0) == 765.0);
            REQUIRE(C[l](0, 1) == 810.0);
            REQUIRE(C[l](0, 2) == 855.0);
            REQUIRE(C[l](1, 0) == 1818.0);
            REQUIRE(C[l](1, 1) == 1944.0);
            REQUIRE(C[l](1, 2) == 2070.0);
            REQUIRE(C[l](2, 0) == 2871.0);
            REQUIRE(C[l](2, 1) == 3078.0);
            REQUIRE(C[l](2, 2) == 3285.0);
        }
    }
}
