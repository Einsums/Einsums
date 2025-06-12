//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <Einsums/HPTT/Transpose.hpp>

#include <memory>
#include <vector>

namespace hptt {

std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA,
                                                    int const *outerSizeA, float const beta, float *B, int const *outerSizeB,
                                                    const SelectionMethod selectionMethod, int const numThreads, int const *threadIds,
                                                    bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                       selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA,
                                                     int const *outerSizeA, double const beta, double *B, int const *outerSizeB,
                                                     const SelectionMethod selectionMethod, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                        selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, const FloatComplex alpha, FloatComplex const *A,
                                                           int const *sizeA, int const *outerSizeA, const FloatComplex beta,
                                                           FloatComplex *B, int const *outerSizeB, const SelectionMethod selectionMethod,
                                                           int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B,
                                                              beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, const DoubleComplex alpha,
                                                            DoubleComplex const *A, int const *sizeA, int const *outerSizeA,
                                                            const DoubleComplex beta, DoubleComplex *B, int const *outerSizeB,
                                                            const SelectionMethod selectionMethod, int const numThreads,
                                                            int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha,
                                                               B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<float>> create_plan(std::vector<int> const &perm, int const dim, float const alpha, float const *A,
                                                    std::vector<int> const &sizeA, std::vector<int> const &outerSizeA, float const beta,
                                                    float *B, std::vector<int> const &outerSizeB, const SelectionMethod selectionMethod,
                                                    int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1, dim, A,
                                                       alpha, B, beta, selectionMethod, numThreads,
                                                       (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(std::vector<int> const &perm, int const dim, double const alpha, double const *A,
                                                     std::vector<int> const &sizeA, std::vector<int> const &outerSizeA, double const beta,
                                                     double *B, std::vector<int> const &outerSizeB, const SelectionMethod selectionMethod,
                                                     int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1, dim, A,
                                                        alpha, B, beta, selectionMethod, numThreads,
                                                        (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(std::vector<int> const &perm, int const dim, const FloatComplex alpha, FloatComplex const *A, std::vector<int> const &sizeA,
            std::vector<int> const &outerSizeA, const FloatComplex beta, FloatComplex *B, std::vector<int> const &outerSizeB,
            const SelectionMethod selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1,
                                                              dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                              (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(std::vector<int> const &perm, int const dim, const DoubleComplex alpha, DoubleComplex const *A, std::vector<int> const &sizeA,
            std::vector<int> const &outerSizeA, const DoubleComplex beta, DoubleComplex *B, std::vector<int> const &outerSizeB,
            const SelectionMethod selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1,
                                                               dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                               (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA,
                                                    int const *outerSizeA, float const beta, float *B, int const *outerSizeB,
                                                    int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                    bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                       MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA,
                                                     int const *outerSizeA, double const beta, double *B, int const *outerSizeB,
                                                     int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                        MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, const FloatComplex alpha, FloatComplex const *A,
                                                           int const *sizeA, int const *outerSizeA, const FloatComplex beta,
                                                           FloatComplex *B, int const *outerSizeB, int const maxAutotuningCandidates,
                                                           int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B,
                                                              beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, const DoubleComplex alpha,
                                                            DoubleComplex const *A, int const *sizeA, int const *outerSizeA,
                                                            const DoubleComplex beta, DoubleComplex *B, int const *outerSizeB,
                                                            int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                            bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha,
                                                               B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), SelectionMethod Class,
 * --int-- Offsets, and ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA,
                                                    int const *outerSizeA, int const *offsetA, float const beta, float *B,
                                                    int const *outerSizeB, int const *offsetB, const SelectionMethod selectionMethod,
                                                    int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                       selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA,
                                                     int const *outerSizeA, int const *offsetA, double const beta, double *B,
                                                     int const *outerSizeB, int const *offsetB, const SelectionMethod selectionMethod,
                                                     int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                        selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, const FloatComplex alpha, FloatComplex const *A,
                                                           int const *sizeA, int const *outerSizeA, int const *offsetA,
                                                           const FloatComplex beta, FloatComplex *B, int const *outerSizeB,
                                                           int const *offsetB, const SelectionMethod selectionMethod, int const numThreads,
                                                           int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B,
                                                              beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(int const *perm, int const dim, const DoubleComplex alpha, DoubleComplex const *A, int const *sizeA, int const *outerSizeA,
            int const *offsetA, const DoubleComplex beta, DoubleComplex *B, int const *outerSizeB, int const *offsetB,
            const SelectionMethod selectionMethod, int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha,
                                                               B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), SelectionMethod Class,
 * --vector int-- Offsets, and vector ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(std::vector<int> const &perm, int const dim, float const alpha, float const *A,
                                                    std::vector<int> const &sizeA, std::vector<int> const &outerSizeA,
                                                    std::vector<int> const &offsetA, float const beta, float *B,
                                                    std::vector<int> const &outerSizeB, std::vector<int> const &offsetB,
                                                    const SelectionMethod selectionMethod, int const numThreads,
                                                    std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0], 1, 1,
                                                       dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                       (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(std::vector<int> const &perm, int const dim, double const alpha, double const *A,
                                                     std::vector<int> const &sizeA, std::vector<int> const &outerSizeA,
                                                     std::vector<int> const &offsetA, double const beta, double *B,
                                                     std::vector<int> const &outerSizeB, std::vector<int> const &offsetB,
                                                     const SelectionMethod selectionMethod, int const numThreads,
                                                     std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0], 1, 1,
                                                        dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                        (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(std::vector<int> const &perm, int const dim, const FloatComplex alpha, FloatComplex const *A, std::vector<int> const &sizeA,
            std::vector<int> const &outerSizeA, std::vector<int> const &offsetA, const FloatComplex beta, FloatComplex *B,
            std::vector<int> const &outerSizeB, std::vector<int> const &offsetB, const SelectionMethod selectionMethod,
            int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                              1, 1, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                              (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(std::vector<int> const &perm, int const dim, const DoubleComplex alpha, DoubleComplex const *A, std::vector<int> const &sizeA,
            std::vector<int> const &outerSizeA, std::vector<int> const &offsetA, const DoubleComplex beta, DoubleComplex *B,
            std::vector<int> const &outerSizeB, std::vector<int> const &offsetB, const SelectionMethod selectionMethod,
            int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0],
                                                               &offsetB[0], 1, 1, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                               (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), --int-- maxAutotuningCandidates,
 * --int-- Offsets, and ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA,
                                                    int const *outerSizeA, int const *offsetA, float const beta, float *B,
                                                    int const *outerSizeB, int const *offsetB, int const maxAutotuningCandidates,
                                                    int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                       MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA,
                                                     int const *outerSizeA, int const *offsetA, double const beta, double *B,
                                                     int const *outerSizeB, int const *offsetB, int const maxAutotuningCandidates,
                                                     int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                        MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, const FloatComplex alpha, FloatComplex const *A,
                                                           int const *sizeA, int const *outerSizeA, int const *offsetA,
                                                           const FloatComplex beta, FloatComplex *B, int const *outerSizeB,
                                                           int const *offsetB, int const maxAutotuningCandidates, int const numThreads,
                                                           int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B,
                                                              beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, const DoubleComplex alpha,
                                                            DoubleComplex const *A, int const *sizeA, int const *outerSizeA,
                                                            int const *offsetA, const DoubleComplex beta, DoubleComplex *B,
                                                            int const *outerSizeB, int const *offsetB, int const maxAutotuningCandidates,
                                                            int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha,
                                                               B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B),
 * SelectionMethod Class, --int-- Offsets, --int-- innerStrides, and ints (sizeA, outerSizeA, and
 * outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA,
                                                    int const *outerSizeA, int const *offsetA, int const innerStrideA, float const beta,
                                                    float *B, int const *outerSizeB, int const *offsetB, int const innerStrideB,
                                                    const SelectionMethod selectionMethod, int const numThreads, int const *threadIds,
                                                    bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                       dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA,
                                                     int const *outerSizeA, int const *offsetA, int const innerStrideA, double const beta,
                                                     double *B, int const *outerSizeB, int const *offsetB, int const innerStrideB,
                                                     const SelectionMethod selectionMethod, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                        dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, const FloatComplex alpha, FloatComplex const *A,
                                                           int const *sizeA, int const *outerSizeA, int const *offsetA,
                                                           int const innerStrideA, const FloatComplex beta, FloatComplex *B,
                                                           int const *outerSizeB, int const *offsetB, int const innerStrideB,
                                                           const SelectionMethod selectionMethod, int const numThreads,
                                                           int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                              innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds,
                                                              useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, const DoubleComplex alpha,
                                                            DoubleComplex const *A, int const *sizeA, int const *outerSizeA,
                                                            int const *offsetA, int const innerStrideA, const DoubleComplex beta,
                                                            DoubleComplex *B, int const *outerSizeB, int const *offsetB,
                                                            int const innerStrideB, const SelectionMethod selectionMethod,
                                                            int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                               innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds,
                                                               useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), SelectionMethod Class,
 * --vector int-- Offsets, --int-- innerStrides, and vector ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(std::vector<int> const &perm, int const dim, float const alpha, float const *A,
                                                    std::vector<int> const &sizeA, std::vector<int> const &outerSizeA,
                                                    std::vector<int> const &offsetA, int const innerStrideA, float const beta, float *B,
                                                    std::vector<int> const &outerSizeB, std::vector<int> const &offsetB,
                                                    int const innerStrideB, const SelectionMethod selectionMethod, int const numThreads,
                                                    std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                       innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                       (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(std::vector<int> const &perm, int const dim, double const alpha, double const *A,
                                                     std::vector<int> const &sizeA, std::vector<int> const &outerSizeA,
                                                     std::vector<int> const &offsetA, int const innerStrideA, double const beta, double *B,
                                                     std::vector<int> const &outerSizeB, std::vector<int> const &offsetB,
                                                     int const innerStrideB, const SelectionMethod selectionMethod, int const numThreads,
                                                     std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                        innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                        (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(std::vector<int> const &perm, int const dim, const FloatComplex alpha, FloatComplex const *A, std::vector<int> const &sizeA,
            std::vector<int> const &outerSizeA, std::vector<int> const &offsetA, int const innerStrideA, const FloatComplex beta,
            FloatComplex *B, std::vector<int> const &outerSizeB, std::vector<int> const &offsetB, int const innerStrideB,
            const SelectionMethod selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                              innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod,
                                                              numThreads, (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(std::vector<int> const &perm, int const dim, const DoubleComplex alpha, DoubleComplex const *A, std::vector<int> const &sizeA,
            std::vector<int> const &outerSizeA, std::vector<int> const &offsetA, int const innerStrideA, const DoubleComplex beta,
            DoubleComplex *B, std::vector<int> const &outerSizeB, std::vector<int> const &offsetB, int const innerStrideB,
            const SelectionMethod selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(
        &sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0], innerStrideA, innerStrideB, dim, A, alpha, B, beta,
        selectionMethod, numThreads, (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), --int-- maxAutotuningCandidates,
 * --int-- Offsets, --int-- innerStrides, and ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA,
                                                    int const *outerSizeA, int const *offsetA, int const innerStrideA, float const beta,
                                                    float *B, int const *outerSizeB, int const *offsetB, int const innerStrideB,
                                                    int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                    bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                       dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA,
                                                     int const *outerSizeA, int const *offsetA, int const innerStrideA, double const beta,
                                                     double *B, int const *outerSizeB, int const *offsetB, int const innerStrideB,
                                                     int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                        dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(int const *perm, int const dim, const FloatComplex alpha, FloatComplex const *A, int const *sizeA, int const *outerSizeA,
            int const *offsetA, int const innerStrideA, const FloatComplex beta, FloatComplex *B, int const *outerSizeB, int const *offsetB,
            int const innerStrideB, int const maxAutotuningCandidates, int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                              innerStrideB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds,
                                                              useRowMajor));
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, const DoubleComplex alpha,
                                                            DoubleComplex const *A, int const *sizeA, int const *outerSizeA,
                                                            int const *offsetA, int const innerStrideA, const DoubleComplex beta,
                                                            DoubleComplex *B, int const *outerSizeB, int const *offsetB,
                                                            int const innerStrideB, int const maxAutotuningCandidates, int const numThreads,
                                                            int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                               innerStrideB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds,
                                                               useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}
} // namespace hptt

#if 0
extern "C" {
void sTensorTranspose(const int *perm, const int dim, const float alpha,
                      const float *A, const int *sizeA, const int *outerSizeA,
                      const float beta, float *B, const int *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<float>>(
      sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1,
      dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void dTensorTranspose(const int *perm, const int dim, const double alpha,
                      const double *A, const int *sizeA, const int *outerSizeA,
                      const double beta, double *B, const int *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<double>>(
      sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1,
      dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void cTensorTranspose(const int *perm, const int dim,
                      const float _Complex alpha, bool conjA,
                      const float _Complex *A, const int *sizeA,
                      const int *outerSizeA, const float _Complex beta,
                      float _Complex *B, const int *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<hptt::FloatComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1,
      dim, (const hptt::FloatComplex *)A, (hptt::FloatComplex)alpha, 
      (hptt::FloatComplex *)B, (hptt::FloatComplex)beta, hptt::ESTIMATE, 
      numThreads, nullptr, useRowMajor));
  plan->setConjA(conjA);
  plan->execute();
}

void zTensorTranspose(const int *perm, const int dim,
                      const double _Complex alpha, bool conjA,
                      const double _Complex *A, const int *sizeA,
                      const int *outerSizeA, const double _Complex beta,
                      double _Complex *B, const int *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<hptt::DoubleComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1,
      dim, (const hptt::DoubleComplex *)A, (hptt::DoubleComplex)alpha, 
      (hptt::DoubleComplex *)B, (hptt::DoubleComplex)beta, hptt::ESTIMATE, 
      numThreads, nullptr, useRowMajor));
  plan->setConjA(conjA);
  plan->execute();
}

/* With Offset */
void sOffsetTensorTranspose(const int *perm, const int dim, const float alpha, 
                            const float *A, const int *sizeA, const int *outerSizeA, const int *offsetA,
                            const float beta, float *B, const int *outerSizeB, const int *offsetB,
                            const int numThreads, const int useRowMajor)
{
  auto plan(std::make_shared<hptt::Transpose<float> >(
      sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1 dim, A, alpha, 
      B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void dOffsetTensorTranspose(const int *perm, const int dim, const double alpha, 
                            const double *A, const int *sizeA, const int *outerSizeA, const int *offsetA, 
                            const double beta, double *B, const int *outerSizeB, const int *offsetB,
                            const int numThreads, const int useRowMajor)
{
  auto plan(std::make_shared<hptt::Transpose<double> >(
      sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1 dim, A, alpha, 
      B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void cOffsetTensorTranspose(const int *perm, const int dim,
                            const float _Complex alpha, bool conjA, 
                            const float _Complex *A, const int *sizeA, const int *outerSizeA, const int *offsetA, 
                            const float _Complex beta, float _Complex *B, const int *outerSizeB, const int *offsetB,
                            const int numThreads, const int useRowMajor)
{
  auto plan(std::make_shared<hptt::Transpose<hptt::FloatComplex> >(
      sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1 dim, 
      (const hptt::FloatComplex*) A, (hptt::FloatComplex)(__real__ alpha, __imag__ alpha), 
      (hptt::FloatComplex*) B, (hptt::FloatComplex)(__real__ beta, __imag__ beta), 
      hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->setConjA(conjA);
  plan->execute();
}

void zOffsetTensorTranspose(const int *perm, const int dim,
                            const double _Complex alpha, bool conjA, 
                            const double _Complex *A, const int *sizeA, const int *outerSizeA, const int *offsetA, 
                            const double _Complex beta, double _Complex *B, const int *outerSizeB, const int *offsetB,
                            const int numThreads, const int useRowMajor)
{
  auto plan(std::make_shared<hptt::Transpose<hptt::DoubleComplex> >(
      sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1 dim, 
      (const hptt::DoubleComplex*) A, (hptt::DoubleComplex)(__real__ alpha, __imag__ alpha), 
      (hptt::DoubleComplex*) B, (hptt::DoubleComplex)(__real__ beta, __imag__ beta), 
      hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->setConjA(conjA);
  plan->execute();
}

/* With Offset and innerStride */
void sInnerStrideTensorTranspose(const int *perm, const int dim, const float alpha, 
                                 const float *A, const int *sizeA, const int *outerSizeA, 
                                 const int *offsetA, const int innerStrideA, const float beta,
                                 float *B, const int *outerSizeB, const int *offsetB, 
                                 const int innerStrideB, const int numThreads, 
                                 const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(
        sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim,
        A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->execute();
}

void dInnerStrideTensorTranspose(const int *perm, const int dim, const double alpha, 
                                 const double *A, const int *sizeA, const int *outerSizeA, 
                                 const int *offsetA, const int innerStrideA, const double beta,
                                 double *B, const int *outerSizeB, const int *offsetB, 
                                 const int innerStrideB, const int numThreads, 
                                 const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(
        sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, 
        A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->execute();
}

void cInnerStrideTensorTranspose(const int *perm, const int dim, const float _Complex alpha, 
                                 bool conjA, const float _Complex *A, const int *sizeA, 
                                 const int *outerSizeA, const int *offsetA, const int innerStrideA, 
                                 const float _Complex beta, float _Complex *B, const int *outerSizeB, 
                                 const int *offsetB, const int innerStrideB, const int numThreads, 
                                 const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<hptt::FloatComplex> >(
        sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, 
        (const hptt::FloatComplex*) A, (hptt::FloatComplex)(__real__ alpha, __imag__ alpha), 
        (hptt::FloatComplex*) B, (hptt::FloatComplex)(__real__ beta, __imag__ beta), hptt::ESTIMATE, 
        numThreads, nullptr, useRowMajor));
   plan->setConjA(conjA);
   plan->execute();
}

void zInnerStrideTensorTranspose(const int *perm, const int dim, const double _Complex alpha, 
                                 bool conjA, const double _Complex *A, const int *sizeA, 
                                 const int *outerSizeA, const int *offsetA, const int innerStrideA, 
                                 const double _Complex beta, double _Complex *B, const int *outerSizeB, 
                                 const int *offsetB, const int innerStrideB, const int numThreads, 
                                 const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<hptt::DoubleComplex> >(
        sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, 
        (const hptt::DoubleComplex*) A, (hptt::DoubleComplex)(__real__ alpha, __imag__ alpha), 
        (hptt::DoubleComplex*) B, (hptt::DoubleComplex)(__real__ beta, __imag__ beta), hptt::ESTIMATE, 
        numThreads, nullptr, useRowMajor));
   plan->setConjA(conjA);
   plan->execute();
}
}
#endif