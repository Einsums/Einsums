//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/../../backends/linear_algebra/hipblas/hipblas.hpp"
#include "einsums/DeviceTensor.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/utility/ComplexTraits.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <hipblas/hipblas.h>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra::gpu)

namespace detail {

/**
 * GPU kernel for the dot product.
 */
template <typename CDataType, typename ADataType, typename BDataType>
__global__ void dot_kernel(CDataType C_prefactor, CDataType *C,
                           std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const ADataType *A,
                           const BDataType *B, size_t elements);

/**
 * Internal gemm functions.
 */
EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *a,
                         int lda, const float *b, int ldb, const float *beta, float *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *a,
                         int lda, const double *b, int ldb, const double *beta, double *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipComplex *alpha,
                         const hipComplex *a, int lda, const hipComplex *b, int ldb, const hipComplex *beta, hipComplex *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipDoubleComplex *alpha,
                         const hipDoubleComplex *a, int lda, const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta,
                         hipDoubleComplex *c, int ldc);

/**
 * Internal ger functions.
 */
EINSUMS_EXPORT void ger(int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);

EINSUMS_EXPORT void ger(int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);

EINSUMS_EXPORT void ger(int m, int n, const hipComplex *alpha, const hipComplex *x, int incx, const hipComplex *y, int incy, hipComplex *A,
                        int lda);

EINSUMS_EXPORT void ger(int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *x, int incx, const hipDoubleComplex *y,
                        int incy, hipDoubleComplex *A, int lda);

/**
 * Internal gemv functions.
 */
EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const float *alpha, const float *a, int lda, const float *x, int incx,
                         const float *beta, float *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const double *alpha, const double *a, int lda, const double *x, int incx,
                         const double *beta, double *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const hipComplex *alpha, const hipComplex *a, int lda,
                         const hipComplex *x, int incx, const hipComplex *beta, hipComplex *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
                         const hipDoubleComplex *x, int incx, const hipDoubleComplex *beta, hipDoubleComplex *y, int incy);

/**
 * Internal scale functions.
 */
EINSUMS_EXPORT void scal(int size, const float *alpha, float *x, int incx);

EINSUMS_EXPORT void scal(int size, const double *alpha, double *x, int incx);

EINSUMS_EXPORT void scal(int size, const hipComplex *alpha, hipComplex *x, int incx);

EINSUMS_EXPORT void scal(int size, const hipDoubleComplex *alpha, hipDoubleComplex *x, int incx);

/**
 * Symmetric multiplication kernels.
 */
__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const float *A, int lda, const float *B, int ldb, float *C,
                                         int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const double *A, int lda, const double *B, int ldb,
                                         double *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const hipComplex *A, int lda, const hipComplex *B, int ldb,
                                         hipComplex *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const hipDoubleComplex *A, int lda,
                                         const hipDoubleComplex *B, int ldb, hipDoubleComplex *C, int ldc);

} // namespace detail

/**
 * @brief Wrapper for matrix multiplication.
 *
 * Performs @f$ C = \alpha OP(A)OP(B) + \beta C @f$.
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C, hipStream_t stream = 0);

/**
 * @brief Performs the dot product.
 */
template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B, hipStream_t stream = 0);

/**
 * @brief Wrapper for vector outer product.
 *
 * Performs @f$ A = A + \alpha X Y^T @f$.
 */
template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A, hipStream_t stream = 0);

/**
 * @brief Wrapper for matrix-vector mulitplication.
 *
 * Performs @f$ y = \alpha OP(A)x + \beta y @f$.
 */
template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y, hipStream_t stream = 0);

/**
 * @brief Scales all the elements in a tensor by a scalar.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A, hipStream_t stream = 0);

/**
 * @brief Computes a common double multiplication between two matrices.
 *
 * Computes @f$ C = OP(B)^T OP(A) OP(B) @f$.
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C, hipStream_t stream = 0);

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra::gpu)

#include "einsums/gpu/GPULinearAlgebra.imp.hip"