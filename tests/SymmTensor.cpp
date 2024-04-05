//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/_Index.hpp"
#include "einsums/_SymmIndex.hpp"

#include <catch2/catch_all.hpp>
#include <type_traits>

#include "einsums.hpp"

TEST_CASE("Symmetric tensor creation", "[tensor][symm]") {
    using namespace einsums;

    auto A = SymmTensor<double, 2,
                        symm_index::SymmIndex<symm_index::SymmIndex<struct tensor_algebra::index::i, struct tensor_algebra::index::j>,
                                              symm_index::SymmIndex<struct tensor_algebra::index::k, struct tensor_algebra::index::l>>>(
        "A", 3, 3);

    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 1) = 4.0;
    A(1, 2) = 5.0;
    A(2, 2) = 6.0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(A(i, j) == A(j, i));
        }
    }
}