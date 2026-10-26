//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Tensor GEMMs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor B("B", 3, 3);
    einsums::Tensor C("C", 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    A.vector_data() = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    B.vector_data() = {11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};

    if (einsums::GlobalConfigMap::get_singleton().get_bool("row-major")) {
        einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0}));

        einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0}));

        einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0}));

        einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0}));
    } else {
        einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0}));

        einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0}));

        einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0}));

        einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0}));
    }
}