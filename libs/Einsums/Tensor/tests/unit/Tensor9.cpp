//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>
#include <tuple>
#include <complex>

TEMPLATE_TEST_CASE("types", "[tensor]", (std::pair<double, float>), (std::pair<std::complex<double>, float>), (std::pair<std::complex<float>, double>)) {
    using namespace einsums;

    auto A = create_random_tensor<typename TestType::second_type>("A", 10, 10);
    auto B = create_random_tensor<typename TestType::first_type>("B", 10, 10);

    B = A;
    
    for(size_t i = 0; i < A.dim(0); i++) {
        for(size_t j = 0; j < A.dim(1); j++) {
            REQUIRE_THAT(std::real(A(i, j)), Catch::Matchers::WithinAbs(std::real(B(i, j)), std::min(einsums::tolerance<typename decltype(A)::ValueType>(), einsums::tolerance<typename decltype(B)::ValueType>())));
        }
    }
}