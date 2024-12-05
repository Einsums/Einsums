//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/TensorAlgebra/Sort.hpp>

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#if defined(I)
#    undef I
#endif

namespace einsums::tensor_algebra::detail {

void sort(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, float const beta, float *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void sort(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta, double *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void sort(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
          std::complex<float> const beta, std::complex<float> *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void sort(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
          std::complex<double> const beta, std::complex<double> *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

} // namespace einsums::tensor_algebra::detail