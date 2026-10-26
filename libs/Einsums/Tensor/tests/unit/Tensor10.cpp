//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Tensor from tensorview", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    auto   A  = create_incremented_tensor<TestType>("A", 10, 10);
    auto   vA = TensorView<TestType, 2>(A, Dim{2, 2}, Offset{4, 4});
    Tensor<TestType, 2> B  = vA;

    A.lock();

    int locked = 0;

#pragma omp parallel shared(locked)
    {
        if (vA.try_lock()) {
            locked++;
            vA.unlock();
        }
    }
    REQUIRE(locked == 1);

    A.unlock();

    vA.lock();

    locked = 0;

#pragma omp parallel shared(locked)
    {
        if (A.try_lock()) {
            locked++;
            A.unlock();
        }
    }
    REQUIRE(locked == 1);

    vA.unlock();

    REQUIRE(B(0, 0) == A(4, 4));
    REQUIRE(B(0, 1) == A(4, 5));
    REQUIRE(B(1, 0) == A(5, 4));
    REQUIRE(B(1, 1) == A(5, 5));
}