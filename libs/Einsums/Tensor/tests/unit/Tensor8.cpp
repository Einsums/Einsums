//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Reshape: 1 guessed dimension", "[tensor]") {
    auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
    REQUIRE_NOTHROW(einsums::Tensor<double, 2>{std::move(C), "D", 10, -1});
    //  NOTE: At this point tensor C is no longer valid.
}

 TEST_CASE("Reshape: No guessed dimensions", "[tensor]") {
     auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
     auto D = einsums::Tensor<double, 2>{std::move(C), "D", 100, 10};
     // NOTE: At this point tensor C is no longer valid.

     // println(C); // <- This will cause a segfault when println tries to print the tensor elements
     // println(D); // <- This succeeds.
 }

 TEST_CASE("Reshape: 2 guessed dimensions", "[tensor]") {
     auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
     REQUIRE_THROWS(einsums::Tensor<double, 2>{std::move(C), "D", -1, -1});
     // NOTE: At this point tensor C is no longer valid.
 }

 TEST_CASE("Reshape: invalid size", "[tensor]") {
     auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
     REQUIRE_THROWS(einsums::Tensor<double, 2>{std::move(C), "D", 9, 9});
     // NOTE: At this point tensor C is no longer valid.
 }
