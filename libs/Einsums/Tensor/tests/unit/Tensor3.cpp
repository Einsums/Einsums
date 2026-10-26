//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Tensor GEMVs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor x("x", 3);
    einsums::Tensor y("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));

    A.vector_data() = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    x.vector_data() = {11.0, 22.0, 33.0};

    if (einsums::GlobalConfigMap::get_singleton().get_bool("row-major")) {
        einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0}));

        einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0}));
    } else {
        einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0}));

        einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0}));
    }
}