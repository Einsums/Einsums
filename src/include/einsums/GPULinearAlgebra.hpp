//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"

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
#include <hipsolver/internal/hipsolver-compat.h>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)

namespace detail {
    namespace gpu {

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

/**
 * Internal solvers.
 */
EINSUMS_EXPORT int gesv(int n, int nrhs, float *A, int lda, int *ipiv, float *B, int ldb, float *X, int ldx);
EINSUMS_EXPORT int gesv(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, double *X, int ldx);
EINSUMS_EXPORT int gesv(int n, int nrhs, hipComplex *A, int lda, int *ipiv, hipComplex *B, int ldb, hipComplex *X, int ldx);
EINSUMS_EXPORT int gesv(int n, int nrhs, hipDoubleComplex *A, int lda, int *ipiv, hipDoubleComplex *B, int ldb, hipDoubleComplex *X,
                        int ldx);

/**
 * Internal axpy.
 */
EINSUMS_EXPORT void axpy(int n, const float *alpha, const float *X, int incx, float *Y, int incy);
EINSUMS_EXPORT void axpy(int n, const double *alpha, const double *X, int incx, double *Y, int incy);
EINSUMS_EXPORT void axpy(int n, const hipComplex *alpha, const hipComplex *X, int incx, hipComplex *Y, int incy);
EINSUMS_EXPORT void axpy(int n, const hipDoubleComplex *alpha, const hipDoubleComplex *X, int incx, hipDoubleComplex *Y, int incy);

EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float *A, int lda, float *D);
EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double *A, int lda, double *D);
    }

} // namespace detail

/**
 * @brief Wrapper for matrix multiplication.
 *
 * Performs @f$ C = \alpha OP(A)OP(B) + \beta C @f$.
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<CType<T, Rank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (::einsums::DeviceRankBlockTensor<AType<T, Rank>, 2, T> &&
                      ::einsums::DeviceRankBlockTensor<BType<T, Rank>, 2, T>);
        requires(!std::is_pointer_v<T>);
    }
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C);

/**
 * @brief Performs the dot product.
 */
template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
    requires requires {
        requires ::einsums::DeviceRankTensor<CType<CDataType, 0>, 0, CDataType>;
        requires ::einsums::DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires ::einsums::DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B);

/**
 * @brief Performs the dot product.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, typename T, size_t BRank>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, ARank, T>;
        requires ::einsums::DeviceRankTensor<BType<T, BRank>, BRank, T>;
    }
T dot(const AType<T, ARank> &A, const BType<T, BRank> &B) {
    einsums::DeviceTensor<T, 0> out("(unnamed)", einsums::detail::DEV_ONLY);

    dot(T{0.0}, out, T{1.0}, A, B);

    return (T)out;
}

/**
 * @brief Wrapper for vector outer product.
 *
 * Performs @f$ A = A + \alpha X Y^T @f$.
 */
template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<AType<T, ARank>, 2, T>;
        requires !std::is_pointer_v<T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A);

/**
 * @brief Wrapper for matrix-vector mulitplication.
 *
 * Performs @f$ y = \alpha OP(A)x + \beta y @f$.
 */
template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<YType<T, XYRank>, 1, T>;
        requires !std::is_pointer_v<T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y);

/**
 * @brief Scales all the elements in a tensor by a scalar.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, ARank, T>;
        requires !std::is_pointer_v<T>;
    }
void scale(T scale, AType<T, ARank> *A);

/**
 * @brief Computes a common double multiplication between two matrices.
 *
 * Computes @f$ C = OP(B)^T OP(A) OP(B) @f$.
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C);

/**
 * @brief Performs @f$y = a*x + y@f$.
 */
template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires(!std::is_pointer_v<T>);
        requires DeviceRankTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y);

/**
 * @brief Solves a linear system.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, BRank>, 2, T>;
    }
int gesv(AType<T, ARank> *A, BType<T, BRank> *B);

/**
 * @brief Wrapper for matrix multiplication.
 *
 * Performs @f$ C = \alpha OP(A)OP(B) + \beta C @f$.
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<CType<T, Rank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (::einsums::DeviceRankBlockTensor<AType<T, Rank>, 2, T> &&
                      ::einsums::DeviceRankBlockTensor<BType<T, Rank>, 2, T>);
    }
void gemm(const T *alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T *beta, CType<T, Rank> *C);

/**
 * @brief Wrapper for vector outer product.
 *
 * Performs @f$ A = A + \alpha X Y^T @f$.
 */
template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<AType<T, ARank>, 2, T>;
    }
void ger(const T *alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A);

/**
 * @brief Wrapper for matrix-vector mulitplication.
 *
 * Performs @f$ y = \alpha OP(A)x + \beta y @f$.
 */
template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(const T *alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const T *beta, YType<T, XYRank> *y);

/**
 * @brief Scales all the elements in a tensor by a scalar.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(const T *scale, AType<T, ARank> *A);

/**
 * @brief Performs @f$y = a*x + y@f$.
 */
template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(const T *alpha, const XType<T, Rank> &X, YType<T, Rank> *Y);

/**
 * @brief Computes the eigenvalues and eigenvectors of a symmetric matrix.
 */
template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<WType<T, WRank>, 1, T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W);

/**
 * @brief Computes the power of a symmetric matrix.
 */
template <template <typename, size_t> typename AType, typename T, size_t ARank>
    requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>
AType<T, ARank> pow(const AType<T, ARank> &A, T expo);

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)

#include "einsums/gpu/GPULinearAlgebra.hpp"