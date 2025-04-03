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
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta, double *B) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
          std::complex<float> const beta, std::complex<float> *B) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
          std::complex<double> const beta, std::complex<double> *B) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, int const *offsetA, int const *outerSizeA,
             float const beta, float *B, int const *offsetB,  int const *outerSizeB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, int const *offsetA, int const *outerSizeA,
             double const beta, double *B, int const *offsetB,  int const *outerSizeB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, std::complex<float> const beta, std::complex<float> *B, int const *offsetB,
             int const *outerSizeB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, std::complex<double> const beta, std::complex<double> *B, int const *offsetB,
             int const *outerSizeB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, hptt::ESTIMATE, omp_get_max_threads(),
                          nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 193e05e9 (FIX: TensorViews currently passed to permute to be combined using HPTT can have non-unitary interal strides across the source tensor. This commit enables non-unitary strides to be supplied to HPTT and be used to skip across a tensor's data. Untested optimisations for Linux ARM and AVX are included.)
void permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, int const *offsetA, int const *outerSizeA, 
             int const innerStrideA, float const beta, float *B, int const *offsetB,  int const *outerSizeB, int const innerStrideB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, int const *offsetA, int const *outerSizeA, 
             int const innerStrideA, double const beta, double *B, int const *offsetB,  int const *outerSizeB, int const innerStrideB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, int const innerStrideA, std::complex<float> const beta, std::complex<float> *B, 
             int const *offsetB, int const *outerSizeB, int const innerStrideB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

void permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
             int const *offsetA, int const *outerSizeA, int const innerStrideA, std::complex<double> const beta, std::complex<double> *B, 
             int const *offsetB, int const *outerSizeB, int const innerStrideB) {
    auto start = std::chrono::high_resolution_clock::now();
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                          hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
    auto time_to_execute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Time to execute: %d\n", time_to_execute);
}

<<<<<<< HEAD
=======
>>>>>>> 4c7befce (FEAT: Extended the use of HPTT to SubTensors following my recent pull request on the HPTT repository. Changes to HPTT are outlined with that commit. Below are changes specific to Einsums. The changes provide a significant computational speedup of the permute function.)
=======
>>>>>>> 193e05e9 (FIX: TensorViews currently passed to permute to be combined using HPTT can have non-unitary interal strides across the source tensor. This commit enables non-unitary strides to be supplied to HPTT and be used to skip across a tensor's data. Untested optimisations for Linux ARM and AVX are included.)
} // namespace einsums::tensor_algebra::detail

#endif
