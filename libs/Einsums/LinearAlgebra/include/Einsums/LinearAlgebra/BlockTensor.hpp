//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/LinearAlgebra/Unoptimized.hpp>
#include <Einsums/Print.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/LinearAlgebra/GPULinearAlgebra.hpp>

#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums::linear_algebra::detail {

template <bool TransA, bool TransB, BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType, typename U>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires MatrixConcept<AType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "gemm: Tensors need the same number of blocks.");
    }

    using T = typename AType::ValueType;

#if defined(EINSUMS_COMPUTE_CODE)
    if constexpr (IsDeviceTensorV<AType>) {
        using dev_datatype = typename AType::dev_datatype;
        dev_datatype *alpha_gpu, *beta_gpu;

        hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
        hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

        hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>((T *)alpha_gpu, A.block(i), B.block(i), (T *)beta_gpu, &(C->block(i)));
        }

        hip_catch(hipFree((void *)alpha_gpu));
        hip_catch(hipFree((void *)beta_gpu));
    } else {
#endif

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>(static_cast<T>(alpha), A.block(i), B.block(i), static_cast<T>(beta), &(C->block(i)));
        }
#if defined(EINSUMS_COMPUTE_CODE)
    }
#endif
}

template <bool TransA, BlockTensorConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    using T = typename AType::ValueType;
    if (beta == U(0.0)) {
        y->zero();
    } else {
        *y *= beta;
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        gemv<TransA>(static_cast<T>(alpha), A.block(i), x(A.block_range(i)), static_cast<T>(1.0), &((*y)(A.block_range(i))));
    }
}

template <bool ComputeEigenvectors = true, BlockTensorConcept AType, VectorConcept WType>
    requires requires {
        requires SameUnderlying<AType, WType>;
        requires NotComplex<AType>;
        requires MatrixConcept<AType>;
    }
void syev(AType *A, WType *W) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        syev<ComputeEigenvectors>(&(A->block(i)), &out_block);
    }
}

template <bool ComputeEigenvectors = true, BlockTensorConcept AType, VectorConcept WType>
    requires requires {
        requires std::is_same_v<AddComplexT<typename AType::ValueType>, typename WType::ValueType>;
        requires MatrixConcept<AType>;
    }
void geev(AType *A, WType *W, AType *lvecs, AType *rvecs) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        geev<ComputeEigenvectors>(&(A->block(i)), &out_block, lvecs, rvecs);
    }
}

template <bool ComputeEigenvectors = true, BlockTensorConcept AType, VectorConcept WType>
    requires requires {
        requires MatrixConcept<AType>;
        requires Complex<AType>;
        requires std::is_same_v<RemoveComplexT<typename AType::ValueType>, typename WType::ValueType>;
    }
void heev(AType *A, WType *W) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        heev<ComputeEigenvectors>(&(A->block(i)), &out_block);
    }
}

template <BlockTensorConcept AType, BlockTensorConcept BType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType>;
    }
auto gesv(AType *A, BType *B) -> int {
    if (A->num_blocks() != B->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "gesv: Tensors need the same number of blocks.");
    }

    int info_out = 0;

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        int info = gesv(&(A->block(i)), &(B->block(i)));

        info_out |= info;

        if (info != 0) {
            println_warn("gesv: Got non-zero return: %d", info);
        }
    }

    return info_out;
}

template <BlockTensorConcept AType>
void scale(typename AType::ValueType alpha, AType *A) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        scale(alpha, &(A->block(i)));
    }
}

template <BlockTensorConcept AType>
    requires MatrixConcept<AType>
void scale_row(size_t row, typename AType::ValueType alpha, AType *A) {
    int  block_ind = A->block_of(row);
    auto temp      = A->block(block_ind)(row - A->block_range(block_ind)[0], All);
    scale(alpha, &temp);
}

template <BlockTensorConcept AType>
    requires MatrixConcept<AType>
void scale_column(size_t column, typename AType::ValueType alpha, AType *A) {
    int  block_ind = A->block_of(column);
    auto temp      = A->block(block_ind)(All, column - A->block_range(block_ind)[0]);
    scale(alpha, &temp);
}

template <BlockTensorConcept AType, BlockTensorConcept BType>
    requires SameRank<AType, BType>
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    if (A.num_blocks() != B.num_blocks()) {
        return dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    if (A.ranges() != B.ranges()) {
        return dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    T out{0};

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        out += dot(A.block(i), B.block(i));
    }

    return out;
}

template <BlockTensorConcept AType, BlockTensorConcept BType>
    requires SameRank<AType, BType>
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    if (A.num_blocks() != B.num_blocks()) {
        return true_dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    if (A.ranges() != B.ranges()) {
        return true_dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    T out{0};

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        out += true_dot(A.block(i), B.block(i));
    }

    return out;
}

template <BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType>
    requires SameRank<AType, BType, CType>
auto dot(AType const &A, BType const &B, CType const &C)
    -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType> {
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C.num_blocks() || B.num_blocks() != C.num_blocks()) {
        return dot(AType::StoredType(A), BType::StoredType(B), CType::StoredType(C));
    }

    if (A.ranges() != B.ranges() || A.ranges() != C.ranges() || B.ranges() != C.ranges()) {
        return dot(AType::StoredType(A), BType::StoredType(B), CType::StoredType(C));
    }

    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType>;

    T out{0};

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        out += dot(A.block(i), B.block(i), C.block(i));
    }

    return out;
}

template <BlockTensorConcept XType, BlockTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {

    if (X.num_blocks() != Y->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "axpy: Tensors need to have the same number of blocks.");
    }

    if (X.ranges() != Y->ranges()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "axpy: Tensor blocks need to be compatible.");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < X.num_blocks(); i++) {
        if (X.block_dim(i) == 0) {
            continue;
        }

        axpy(alpha, X[i], &(Y->block(i)));
    }
}

template <BlockTensorConcept XType, BlockTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpby(typename XType::ValueType alpha, XType const &X, typename YType::ValueType beta, YType *Y) {

    if (X.num_blocks() != Y->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "axpby: Tensors need to have the same number of blocks.");
    }

    if (X.ranges() != Y->ranges()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "axpby: Tensor blocks need to be compatible.");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < X.num_blocks(); i++) {
        if (X.block_dim(i) == 0) {
            continue;
        }
        axpby(alpha, X[i], beta, &(Y->block(i)));
    }
}

template <BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_product(typename AType::ValueType alpha, AType const &A, BType const &B, typename CType::ValueType beta, CType *C) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        direct_product(alpha, A.block(i), B.block(i), beta, &(C->block(i)));
    }
}

template <BlockTensorConcept AType>
    requires MatrixConcept<AType>
auto pow(AType const &a, typename AType::ValueType alpha,
         typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon()) -> RemoveViewT<AType> {
    RemoveViewT<AType> out{"pow result", a.vector_dims()};

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < a.num_blocks(); i++) {
        if (a.block_dim(i) == 0) {
            continue;
        }
        out[i] = pow(a[i], alpha, cutoff);
    }

    return out;
}

} // namespace einsums::linear_algebra::detail