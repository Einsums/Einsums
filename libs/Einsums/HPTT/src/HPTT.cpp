//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <memory>
#include <vector>

#include <Einsums/HPTT/Transpose.hpp>

namespace hptt {

std::shared_ptr<hptt::Transpose<float>>
create_plan(const int *perm, const int dim, const float alpha, const float *A,
            const int *sizeA, const int *outerSizeA, const float beta, float *B,
            const int *outerSizeB, const SelectionMethod selectionMethod,
            const int numThreads, const int *threadIds,
            const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<float>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta,
      selectionMethod, numThreads, threadIds, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<double>>
create_plan(const int *perm, const int dim, const double alpha, const double *A,
            const int *sizeA, const int *outerSizeA, const double beta,
            double *B, const int *outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const int *threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<double>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta,
      selectionMethod, numThreads, threadIds, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(const int *perm, const int dim, const FloatComplex alpha,
            const FloatComplex *A, const int *sizeA, const int *outerSizeA,
            const FloatComplex beta, FloatComplex *B, const int *outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const int *threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta,
      selectionMethod, numThreads, threadIds, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(const int *perm, const int dim, const DoubleComplex alpha,
            const DoubleComplex *A, const int *sizeA, const int *outerSizeA,
            const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const int *threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta,
      selectionMethod, numThreads, threadIds, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<float>>
create_plan(const std::vector<int> &perm, const int dim, const float alpha,
            const float *A, const std::vector<int> &sizeA,
            const std::vector<int> &outerSizeA, const float beta, float *B,
            const std::vector<int> &outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const std::vector<int> &threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<float>>(
      &sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B,
      beta, selectionMethod, numThreads,
      (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<double>>
create_plan(const std::vector<int> &perm, const int dim, const double alpha,
            const double *A, const std::vector<int> &sizeA,
            const std::vector<int> &outerSizeA, const double beta, double *B,
            const std::vector<int> &outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const std::vector<int> &threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<double>>(
      &sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B,
      beta, selectionMethod, numThreads,
      (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(const std::vector<int> &perm, const int dim,
            const FloatComplex alpha, const FloatComplex *A,
            const std::vector<int> &sizeA, const std::vector<int> &outerSizeA,
            const FloatComplex beta, FloatComplex *B,
            const std::vector<int> &outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const std::vector<int> &threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(
      &sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B,
      beta, selectionMethod, numThreads,
      (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(const std::vector<int> &perm, const int dim,
            const DoubleComplex alpha, const DoubleComplex *A,
            const std::vector<int> &sizeA, const std::vector<int> &outerSizeA,
            const DoubleComplex beta, DoubleComplex *B,
            const std::vector<int> &outerSizeB,
            const SelectionMethod selectionMethod, const int numThreads,
            const std::vector<int> &threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(
      &sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B,
      beta, selectionMethod, numThreads,
      (threadIds.size() > 0) ? &threadIds[0] : nullptr, useRowMajor));
  return plan;
}

std::shared_ptr<hptt::Transpose<float>>
create_plan(const int *perm, const int dim, const float alpha, const float *A,
            const int *sizeA, const int *outerSizeA, const float beta, float *B,
            const int *outerSizeB, const int maxAutotuningCandidates,
            const int numThreads, const int *threadIds,
            const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<float>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE,
      numThreads, threadIds, useRowMajor));
  plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
  plan->createPlan();
  return plan;
}

std::shared_ptr<hptt::Transpose<double>>
create_plan(const int *perm, const int dim, const double alpha, const double *A,
            const int *sizeA, const int *outerSizeA, const double beta,
            double *B, const int *outerSizeB, const int maxAutotuningCandidates,
            const int numThreads, const int *threadIds,
            const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<double>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE,
      numThreads, threadIds, useRowMajor));
  plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
  plan->createPlan();
  return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex>>
create_plan(const int *perm, const int dim, const FloatComplex alpha,
            const FloatComplex *A, const int *sizeA, const int *outerSizeA,
            const FloatComplex beta, FloatComplex *B, const int *outerSizeB,
            const int maxAutotuningCandidates, const int numThreads,
            const int *threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<FloatComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE,
      numThreads, threadIds, useRowMajor));
  plan->createPlan();
  return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex>>
create_plan(const int *perm, const int dim, const DoubleComplex alpha,
            const DoubleComplex *A, const int *sizeA, const int *outerSizeA,
            const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB,
            const int maxAutotuningCandidates, const int numThreads,
            const int *threadIds, const bool useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<DoubleComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE,
      numThreads, threadIds, useRowMajor));
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
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta,
      hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void dTensorTranspose(const int *perm, const int dim, const double alpha,
                      const double *A, const int *sizeA, const int *outerSizeA,
                      const double beta, double *B, const int *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<double>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta,
      hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
  plan->execute();
}

void cTensorTranspose(const int *perm, const int dim,
                      const float _Complex alpha, bool conjA,
                      const float _Complex *A, const int *sizeA,
                      const int *outerSizeA, const float _Complex beta,
                      float _Complex *B, const int *outerSizeB,
                      const int numThreads, const int useRowMajor) {
  auto plan(std::make_shared<hptt::Transpose<hptt::FloatComplex>>(
      sizeA, perm, outerSizeA, outerSizeB, dim, (const hptt::FloatComplex *)A,
      (hptt::FloatComplex)alpha, (hptt::FloatComplex *)B,
      (hptt::FloatComplex)beta, hptt::ESTIMATE, numThreads, nullptr,
      useRowMajor));
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
      sizeA, perm, outerSizeA, outerSizeB, dim, (const hptt::DoubleComplex *)A,
      (hptt::DoubleComplex)alpha, (hptt::DoubleComplex *)B,
      (hptt::DoubleComplex)beta, hptt::ESTIMATE, numThreads, nullptr,
      useRowMajor));
  plan->setConjA(conjA);
  plan->execute();
}
}
#endif