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
