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

std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                    size_t const *outerSizeA, float const beta, float *B, size_t const *outerSizeB,
                                                    SelectionMethod const selectionMethod, int const numThreads, int const *threadIds,
                                                    bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                       selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A,
                                                     size_t const *sizeA, size_t const *outerSizeA, double const beta, double *B,
                                                     size_t const *outerSizeB, SelectionMethod const selectionMethod, int const numThreads,
                                                     int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                        selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, FloatComplex const alpha, FloatComplex const *A,
                                                           size_t const *sizeA, size_t const *outerSizeA, FloatComplex const beta,
                                                           FloatComplex *B, size_t const *outerSizeB, SelectionMethod const selectionMethod,
                                                           int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B,
                                                              beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, DoubleComplex const alpha,
                                                            DoubleComplex const *A, size_t const *sizeA, size_t const *outerSizeA,
                                                            DoubleComplex const beta, DoubleComplex *B, size_t const *outerSizeB,
                                                            SelectionMethod const selectionMethod, int const numThreads,
                                                            int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha,
                                                               B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<float>> create_plan(std::vector<int> const &perm, int const dim, float const alpha, float const *A,
                                                    std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                    float const beta, float *B, std::vector<size_t> const &outerSizeB,
                                                    SelectionMethod const selectionMethod, int const numThreads,
                                                    std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1, dim, A,
                                                       alpha, B, beta, selectionMethod, numThreads,
                                                       (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(std::vector<int> const &perm, int const dim, double const alpha, double const *A,
                                                     std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                     double const beta, double *B, std::vector<size_t> const &outerSizeB,
                                                     SelectionMethod const selectionMethod, int const numThreads,
                                                     std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1, dim, A,
                                                        alpha, B, beta, selectionMethod, numThreads,
                                                        (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(std::vector<int> const &perm, int const dim, FloatComplex const alpha, FloatComplex const *A, std::vector<size_t> const &sizeA,
            std::vector<size_t> const &outerSizeA, FloatComplex const beta, FloatComplex *B, std::vector<size_t> const &outerSizeB,
            SelectionMethod const selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1,
                                                              dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                              (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(std::vector<int> const &perm, int const dim, DoubleComplex const alpha,
                                                            DoubleComplex const *A, std::vector<size_t> const &sizeA,
                                                            std::vector<size_t> const &outerSizeA, DoubleComplex const beta,
                                                            DoubleComplex *B, std::vector<size_t> const &outerSizeB,
                                                            SelectionMethod const selectionMethod, int const numThreads,
                                                            std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], nullptr, nullptr, 1, 1,
                                                               dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                               (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                    size_t const *outerSizeA, float const beta, float *B, size_t const *outerSizeB,
                                                    int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                    bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                       MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A,
                                                     size_t const *sizeA, size_t const *outerSizeA, double const beta, double *B,
                                                     size_t const *outerSizeB, int const maxAutotuningCandidates, int const numThreads,
                                                     int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta,
                                                        MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, FloatComplex const alpha, FloatComplex const *A,
                                                           size_t const *sizeA, size_t const *outerSizeA, FloatComplex const beta,
                                                           FloatComplex *B, size_t const *outerSizeB, int const maxAutotuningCandidates,
                                                           int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B,
                                                              beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, DoubleComplex const alpha,
                                                            DoubleComplex const *A, size_t const *sizeA, size_t const *outerSizeA,
                                                            DoubleComplex const beta, DoubleComplex *B, size_t const *outerSizeB,
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
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                    size_t const *outerSizeA, size_t const *offsetA, float const beta, float *B,
                                                    size_t const *outerSizeB, size_t const *offsetB, SelectionMethod const selectionMethod,
                                                    int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                       selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A,
                                                     size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                     double const beta, double *B, size_t const *outerSizeB, size_t const *offsetB,
                                                     SelectionMethod const selectionMethod, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                        selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, FloatComplex const alpha, FloatComplex const *A,
                                                           size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                           FloatComplex const beta, FloatComplex *B, size_t const *outerSizeB,
                                                           size_t const *offsetB, SelectionMethod const selectionMethod,
                                                           int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B,
                                                              beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, DoubleComplex const alpha,
                                                            DoubleComplex const *A, size_t const *sizeA, size_t const *outerSizeA,
                                                            size_t const *offsetA, DoubleComplex const beta, DoubleComplex *B,
                                                            size_t const *outerSizeB, size_t const *offsetB,
                                                            SelectionMethod const selectionMethod, int const numThreads,
                                                            int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha,
                                                               B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), SelectionMethod Class,
 * --vector int-- Offsets, and vector ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(std::vector<int> const &perm, int const dim, float const alpha, float const *A,
                                                    std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                    std::vector<size_t> const &offsetA, float const beta, float *B,
                                                    std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB,
                                                    SelectionMethod const selectionMethod, int const numThreads,
                                                    std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0], 1, 1,
                                                       dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                       (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(std::vector<int> const &perm, int const dim, double const alpha, double const *A,
                                                     std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                     std::vector<size_t> const &offsetA, double const beta, double *B,
                                                     std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB,
                                                     SelectionMethod const selectionMethod, int const numThreads,
                                                     std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0], 1, 1,
                                                        dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                        (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(std::vector<int> const &perm, int const dim, FloatComplex const alpha, FloatComplex const *A, std::vector<size_t> const &sizeA,
            std::vector<size_t> const &outerSizeA, std::vector<size_t> const &offsetA, FloatComplex const beta, FloatComplex *B,
            std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB, SelectionMethod const selectionMethod,
            int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                              1, 1, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                              (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(std::vector<int> const &perm, int const dim, DoubleComplex const alpha, DoubleComplex const *A,
            std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA, std::vector<size_t> const &offsetA,
            DoubleComplex const beta, DoubleComplex *B, std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB,
            SelectionMethod const selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0],
                                                               &offsetB[0], 1, 1, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                               (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), --int-- maxAutotuningCandidates,
 * --int-- Offsets, and ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                    size_t const *outerSizeA, size_t const *offsetA, float const beta, float *B,
                                                    size_t const *outerSizeB, size_t const *offsetB, int const maxAutotuningCandidates,
                                                    int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                       MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A,
                                                     size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                     double const beta, double *B, size_t const *outerSizeB, size_t const *offsetB,
                                                     int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta,
                                                        MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, FloatComplex const alpha, FloatComplex const *A,
                                                           size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                           FloatComplex const beta, FloatComplex *B, size_t const *outerSizeB,
                                                           size_t const *offsetB, int const maxAutotuningCandidates, int const numThreads,
                                                           int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B,
                                                              beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(int const *perm, int const dim, DoubleComplex const alpha, DoubleComplex const *A, size_t const *sizeA,
            size_t const *outerSizeA, size_t const *offsetA, DoubleComplex const beta, DoubleComplex *B, size_t const *outerSizeB,
            size_t const *offsetB, int const maxAutotuningCandidates, int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha,
                                                               B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B),
 * SelectionMethod Class, --int-- Offsets, --int-- innerStrides, and ints (sizeA, outerSizeA, and
 * outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                    size_t const *outerSizeA, size_t const *offsetA, size_t const innerStrideA,
                                                    float const beta, float *B, size_t const *outerSizeB, size_t const *offsetB,
                                                    size_t const innerStrideB, SelectionMethod const selectionMethod, int const numThreads,
                                                    int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                       dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A,
                                                     size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                     size_t const innerStrideA, double const beta, double *B, size_t const *outerSizeB,
                                                     size_t const *offsetB, size_t const innerStrideB,
                                                     SelectionMethod const selectionMethod, int const numThreads, int const *threadIds,
                                                     bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                        dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, FloatComplex const alpha, FloatComplex const *A,
                                                           size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                           size_t const innerStrideA, FloatComplex const beta, FloatComplex *B,
                                                           size_t const *outerSizeB, size_t const *offsetB, size_t const innerStrideB,
                                                           SelectionMethod const selectionMethod, int const numThreads,
                                                           int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                              innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds,
                                                              useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, DoubleComplex const alpha,
                                                            DoubleComplex const *A, size_t const *sizeA, size_t const *outerSizeA,
                                                            size_t const *offsetA, size_t const innerStrideA, DoubleComplex const beta,
                                                            DoubleComplex *B, size_t const *outerSizeB, size_t const *offsetB,
                                                            size_t const innerStrideB, SelectionMethod const selectionMethod,
                                                            int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                               innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds,
                                                               useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), SelectionMethod Class,
 * --vector int-- Offsets, --int-- innerStrides, and vector ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(std::vector<int> const &perm, int const dim, float const alpha, float const *A,
                                                    std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                    std::vector<size_t> const &offsetA, size_t const innerStrideA, float const beta,
                                                    float *B, std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB,
                                                    size_t const innerStrideB, SelectionMethod const selectionMethod, int const numThreads,
                                                    std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                       innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                       (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(std::vector<int> const &perm, int const dim, double const alpha, double const *A,
                                                     std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                     std::vector<size_t> const &offsetA, size_t const innerStrideA, double const beta,
                                                     double *B, std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB,
                                                     size_t const innerStrideB, SelectionMethod const selectionMethod, int const numThreads,
                                                     std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                        innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                                        (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(std::vector<int> const &perm, int const dim, FloatComplex const alpha, FloatComplex const *A, std::vector<size_t> const &sizeA,
            std::vector<size_t> const &outerSizeA, std::vector<size_t> const &offsetA, size_t const innerStrideA, FloatComplex const beta,
            FloatComplex *B, std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB, size_t const innerStrideB,
            SelectionMethod const selectionMethod, int const numThreads, std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0],
                                                              innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod,
                                                              numThreads, (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(std::vector<int> const &perm, int const dim, DoubleComplex const alpha, DoubleComplex const *A,
            std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA, std::vector<size_t> const &offsetA,
            size_t const innerStrideA, DoubleComplex const beta, DoubleComplex *B, std::vector<size_t> const &outerSizeB,
            std::vector<size_t> const &offsetB, size_t const innerStrideB, SelectionMethod const selectionMethod, int const numThreads,
            std::vector<int> const &threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(
        &sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], &offsetA[0], &offsetB[0], innerStrideA, innerStrideB, dim, A, alpha, B, beta,
        selectionMethod, numThreads, (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
    return plan;
}

/* Methods with (floats, doubles, FloatComplexes, and DoubleComplexes) (alpha, A, beta, and B), --int-- maxAutotuningCandidates,
 * --int-- Offsets, --int-- innerStrides, and ints (sizeA, outerSizeA, and outerSizeB). */
std::shared_ptr<hptt::Transpose<float>> create_plan(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                    size_t const *outerSizeA, size_t const *offsetA, size_t const innerStrideA,
                                                    float const beta, float *B, size_t const *outerSizeB, size_t const *offsetB,
                                                    size_t const innerStrideB, int const maxAutotuningCandidates, int const numThreads,
                                                    int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<float>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                       dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<double>> create_plan(int const *perm, int const dim, double const alpha, double const *A,
                                                     size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                     size_t const innerStrideA, double const beta, double *B, size_t const *outerSizeB,
                                                     size_t const *offsetB, size_t const innerStrideB, int const maxAutotuningCandidates,
                                                     int const numThreads, int const *threadIds, bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<double>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                        dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor));
    plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>> create_plan(int const *perm, int const dim, FloatComplex const alpha, FloatComplex const *A,
                                                           size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                                                           size_t const innerStrideA, FloatComplex const beta, FloatComplex *B,
                                                           size_t const *outerSizeB, size_t const *offsetB, size_t const innerStrideB,
                                                           int const maxAutotuningCandidates, int const numThreads, int const *threadIds,
                                                           bool const useRowMajor) {
    auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA,
                                                              innerStrideB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds,
                                                              useRowMajor));
    plan->createPlan();
    return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>> create_plan(int const *perm, int const dim, DoubleComplex const alpha,
                                                            DoubleComplex const *A, size_t const *sizeA, size_t const *outerSizeA,
                                                            size_t const *offsetA, size_t const innerStrideA, DoubleComplex const beta,
                                                            DoubleComplex *B, size_t const *outerSizeB, size_t const *offsetB,
                                                            size_t const innerStrideB, int const maxAutotuningCandidates,
                                                            int const numThreads, int const *threadIds, bool const useRowMajor) {
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
                      const float *A, const size_t *sizeA, const size_t *outerSizeA,
                      const float beta, float *B, const size_t *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<float>>(
      sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1,
      dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void dTensorTranspose(const int *perm, const int dim, const double alpha,
                      const double *A, const size_t *sizeA, const size_t *outerSizeA,
                      const double beta, double *B, const size_t *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<double>>(
      sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1,
      dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void cTensorTranspose(const int *perm, const int dim,
                      const float _Complex alpha, bool conjA,
                      const float _Complex *A, const size_t *sizeA,
                      const size_t *outerSizeA, const float _Complex beta,
                      float _Complex *B, const size_t *outerSizeB,
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
                      const double _Complex *A, const size_t *sizeA,
                      const size_t *outerSizeA, const double _Complex beta,
                      double _Complex *B, const size_t *outerSizeB,
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
                            const float *A, const size_t *sizeA, const size_t *outerSizeA, const size_t *offsetA,
                            const float beta, float *B, const size_t *outerSizeB, const size_t *offsetB,
                            const int numThreads, const int useRowMajor)
{
  auto plan(std::make_shared<hptt::Transpose<float> >(
      sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1 dim, A, alpha, 
      B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void dOffsetTensorTranspose(const int *perm, const int dim, const double alpha, 
                            const double *A, const size_t *sizeA, const size_t *outerSizeA, const size_t *offsetA, 
                            const double beta, double *B, const size_t *outerSizeB, const size_t *offsetB,
                            const int numThreads, const int useRowMajor)
{
  auto plan(std::make_shared<hptt::Transpose<double> >(
      sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1 dim, A, alpha, 
      B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void cOffsetTensorTranspose(const int *perm, const int dim,
                            const float _Complex alpha, bool conjA, 
                            const float _Complex *A, const size_t *sizeA, const size_t *outerSizeA, const size_t *offsetA, 
                            const float _Complex beta, float _Complex *B, const size_t *outerSizeB, const size_t *offsetB,
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
                            const double _Complex *A, const size_t *sizeA, const size_t *outerSizeA, const size_t *offsetA, 
                            const double _Complex beta, double _Complex *B, const size_t *outerSizeB, const size_t *offsetB,
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
                                 const float *A, const size_t *sizeA, const size_t *outerSizeA, 
                                 const size_t *offsetA, const size_t innerStrideA, const float beta,
                                 float *B, const size_t *outerSizeB, const size_t *offsetB, 
                                 const size_t innerStrideB, const int numThreads, 
                                 const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(
        sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim,
        A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->execute();
}

void dInnerStrideTensorTranspose(const int *perm, const int dim, const double alpha, 
                                 const double *A, const size_t *sizeA, const size_t *outerSizeA, 
                                 const size_t *offsetA, const size_t innerStrideA, const double beta,
                                 double *B, const size_t *outerSizeB, const size_t *offsetB, 
                                 const size_t innerStrideB, const int numThreads, 
                                 const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(
        sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, 
        A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->execute();
}

void cInnerStrideTensorTranspose(const int *perm, const int dim, const float _Complex alpha, 
                                 bool conjA, const float _Complex *A, const size_t *sizeA, 
                                 const size_t *outerSizeA, const size_t *offsetA, const size_t innerStrideA, 
                                 const float _Complex beta, float _Complex *B, const size_t *outerSizeB, 
                                 const size_t *offsetB, const size_t innerStrideB, const int numThreads, 
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
                                 bool conjA, const double _Complex *A, const size_t *sizeA, 
                                 const size_t *outerSizeA, const size_t *offsetA, const size_t innerStrideA, 
                                 const double _Complex beta, double _Complex *B, const size_t *outerSizeB, 
                                 const size_t *offsetB, const size_t innerStrideB, const int numThreads, 
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