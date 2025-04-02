//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>
#include <chrono>

#if !defined(EINSUMS_WINDOWS)
#    include <Einsums/HPTT/HPTT.hpp>
#    include <Einsums/TensorAlgebra/Permute.hpp>

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#    if defined(I)
#        undef I
#    endif

namespace einsums::tensor_algebra::detail {

void permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, float const beta, float *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta, double *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
          std::complex<float> const beta, std::complex<float> *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
          std::complex<double> const beta, std::complex<double> *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, int const *offsetA, int const *outerSizeA,
             float const beta, float *B, int const *offsetB,  int const *outerSizeB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, int const *offsetA, int const *outerSizeA,
             double const beta, double *B, int const *offsetB,  int const *outerSizeB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, std::complex<float> const beta, std::complex<float> *B, int const *offsetB,
             int const *outerSizeB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, std::complex<double> const beta, std::complex<double> *B, int const *offsetB,
             int const *outerSizeB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, int const *offsetA, int const *outerSizeA, 
             int const innerStrideA, float const beta, float *B, int const *offsetB,  int const *outerSizeB, int const innerStrideB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, int const *offsetA, int const *outerSizeA, 
             int const innerStrideA, double const beta, double *B, int const *offsetB,  int const *outerSizeB, int const innerStrideB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, int const innerStrideA, std::complex<float> const beta, std::complex<float> *B, 
             int const *offsetB, int const *outerSizeB, int const innerStrideB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, int const innerStrideA, std::complex<double> const beta, std::complex<double> *B, 
             int const *offsetB, int const *outerSizeB, int const innerStrideB) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

} // namespace einsums::tensor_algebra::detail

#endif
