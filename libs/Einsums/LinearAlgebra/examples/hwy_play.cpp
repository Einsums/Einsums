//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy_play-inl.hpp"
#include <hwy/foreach_target.h> // IWYU pragma: keep

HWY_BEFORE_NAMESPACE();
namespace einsums {

void ddirprod(double alpha, Tensor<double, 1> const &A, Tensor<double, 1> const &B, Tensor<double, 1> *C) {
    HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(dirprod_kernel<double>)(alpha, A.data(), B.data(), C->data(), A.size());
}
} // namespace einsums
HWY_AFTER_NAMESPACE();

int einsums_main() {
    using namespace einsums;

    size_t i{10};

    auto A = create_random_tensor("A", i);
    auto B = create_random_tensor("B", i);
    auto C = create_zero_tensor("C", i);

    ddirprod(1.0, A, B, &C);

    println(A);
    println(B);
    println(C);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}