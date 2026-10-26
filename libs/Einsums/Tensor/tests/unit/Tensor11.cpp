//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/ARange.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("arange", "[tensor]", float, double) {
    auto A = einsums::arange<TestType>(0, 10);

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
}
