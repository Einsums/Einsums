//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Tensor SYEVs", "[tensor]") {
    einsums::Tensor A(true, "A", 3, 3);
    einsums::Tensor x(true, "x", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));

    A.vector_data() = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    einsums::linear_algebra::syev(&A, &x);

    CHECK_THAT(x(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(x(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(x(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));
}