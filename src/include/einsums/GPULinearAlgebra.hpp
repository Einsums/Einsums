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

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::gpu::linear_algebra)

using namespace einsums::gpu::detail;
using namespace einsums::backend::linear_algebra::hipblas;
using namespace einsums::backend::linear_algebra::hipblas::detail;

namespace detail {

template <typename CDataType, typename ADataType, typename BDataType>
__global__ void dot_kernel(CDataType C_prefactor, CDataType *C,
                           std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const ADataType *A,
                           const BDataType *B, size_t elements);

template <typename T>
void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const T *alpha, const T *a, int lda, const T *b,
          int ldb, const T *beta, T *c, int ldc);

template <typename T>
void ger(int m, int n, const T *alpha, const T *x, int incx, const T *y, int incy, T *A, int lda);

template <typename T>
void gemv(hipblasOperation_t transa, int m, int n, const T *alpha, const T *a, int lda, const T *x, int incx, const T *beta, T *y,
          int incy);

template <typename T>
void scal(int size, const T *alpha, T *x, int incx);

} // namespace detail

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C);

template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B, dim3 threads = dim3(32), dim3 blocks = dim3(32));

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires DeviceRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A);

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y);

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A);

END_EINSUMS_NAMESPACE_HPP(einsums::gpu::linear_algebra)

#include "einsums/gpu/GPULinearAlgebra.imp.hip"