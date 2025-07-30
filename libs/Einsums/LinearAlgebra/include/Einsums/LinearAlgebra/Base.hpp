//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Assert.hpp>
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/LinearAlgebra/Bases/gemm.hpp>
#include <Einsums/LinearAlgebra/Bases/gemv.hpp>
#include <Einsums/LinearAlgebra/Bases/sum_square.hpp>
#include <Einsums/LinearAlgebra/Bases/syev.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

#include "Einsums/Errors/Error.hpp"
#include "Einsums/TensorImpl/TensorImpl.hpp"

namespace einsums::linear_algebra::detail {

template <typename T>
void sum_square(einsums::detail::TensorImpl<T> const &a, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) {
    impl_sum_square(a, scale, sumsq);
}

template <CoreBasicTensorConcept AType>
void sum_square(AType const &a, RemoveComplexT<typename AType::ValueType> *scale, RemoveComplexT<typename AType::ValueType> *sumsq) {
    sum_square(a.impl(), scale, sumsq);
}

template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void gemm(char transA, char transB, AlphaType const alpha, einsums::detail::TensorImpl<AType> const &A,
          einsums::detail::TensorImpl<BType> const &B, BetaType const beta, einsums::detail::TensorImpl<CType> *C) {
    char const tA = std::tolower(transA), tB = std::tolower(transB);
    // Check for gemmability.
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to gemm need to be matrices! Got ranks {}, {}, and {}.", A.rank(), B.rank(),
                                C->rank());
    }
    if (tA == 'n' && tB == 'n') {
        if (A.dim(1) != B.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(1), B.dim(0));
        }

        if (A.dim(0) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(0), C->dim(0));
        }

        if (B.dim(1) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(1), C->dim(1));
        }
    } else if (tA != 'n' && tB == 'n') {
        if (A.dim(0) != B.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(0), B.dim(0));
        }

        if (A.dim(1) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(1), C->dim(0));
        }

        if (B.dim(1) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(1), C->dim(1));
        }
    } else if (tA == 'n' && tB != 'n') {
        if (A.dim(1) != B.dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(1), B.dim(1));
        }

        if (A.dim(0) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(0), C->dim(0));
        }

        if (B.dim(0) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(0), C->dim(1));
        }
    } else if (tA != 'n' && tB != 'n') {
        if (A.dim(0) != B.dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(0), B.dim(1));
        }

        if (A.dim(1) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(1), C->dim(0));
        }

        if (B.dim(0) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(0), C->dim(1));
        }
    }

    impl_gemm(transA, transB, alpha, A, B, beta, C);
}

template <bool TransA, bool TransB, typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void gemm(AlphaType const alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B,
          BetaType const beta, einsums::detail::TensorImpl<CType> *C) {
    gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', alpha, A, B, beta, C);
}

template <bool TransA, bool TransB, typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires CoreBasicTensorConcept<AType>;
        requires SameUnderlying<AType, BType, CType>;
        requires(std::convertible_to<U, typename AType::ValueType>);
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    gemm<TransA, TransB>(alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires CoreBasicTensorConcept<AType>;
        requires SameUnderlying<AType, BType, CType>;
        requires(std::convertible_to<U, typename AType::ValueType>);
    }
void gemm(char transA, char transB, U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    gemm(transA, transB, alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <typename AType, typename XType, typename YType, typename AlphaType, typename BetaType>
void gemv(char transA, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<XType> const &X,
          BetaType beta, einsums::detail::TensorImpl<YType> *Y) {
    if (A.rank() != 2 || X.rank() != 1 || Y->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(
            rank_error, "The ranks of the tensors passed to gemv are incompatible! Requires a matrix and to vectors. Got {}, {}, and {}.",
            A.rank(), X.rank(), Y->rank());
    }
    if (std::tolower(transA) == 'n') {
        if (A.dim(1) != Y->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The dimensions of the input matrix and output tensor do not match! Got {} and {}.", A.dim(1),
                                    Y->dim(0));
        }
        if (A.dim(0) != X.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the input matrix and input tensor do not match! Got {} and {}.",
                                    A.dim(0), X.dim(0));
        }
    } else {
        if (A.dim(0) != Y->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The dimensions of the input matrix and output tensor do not match! Got {} and {}.", A.dim(0),
                                    Y->dim(0));
        }
        if (A.dim(1) != X.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the input matrix and input tensor do not match! Got {} and {}.",
                                    A.dim(1), X.dim(0));
        }
    }
    impl_gemv(transA, alpha, A, X, beta, Y);
}

template <bool TransA, typename AType, typename XType, typename YType, typename AlphaType, typename BetaType>
void gemv(AlphaType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<XType> const &X, BetaType beta,
          einsums::detail::TensorImpl<YType> *Y) {

    gemv((TransA) ? 't' : 'n', alpha, A, X, beta, Y);
}

template <bool TransA, typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires CoreBasicTensorConcept<AType>;
        requires CoreBasicTensorConcept<XType>;
        requires CoreBasicTensorConcept<YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    gemv<TransA>(alpha, A.impl(), z.impl(), beta, &y->impl());
}

template <typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires CoreBasicTensorConcept<AType>;
        requires CoreBasicTensorConcept<XType>;
        requires CoreBasicTensorConcept<YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(char transA, U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    gemv(transA, alpha, A.impl(), z.impl(), beta, &y->impl());
}

template <bool ComputeEigenvectors = true, typename AType>
void syev(einsums::detail::TensorImpl<AType> *A, einsums::detail::TensorImpl<RemoveComplexT<AType>> *W) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error,
                                "The inputs to syev/heev need to be a pointer to a matrix and a pointer to a vector. Got ranks {} and {}.",
                                A->rank(), W->rank());
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The input matrix to syev/heev needs to be square and symmetric.");
    }

    if (A->dim(0) != W->dim(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The input and output to syev/heev have incompatible dimensions.");
    }

    if (A->dim(0) == 0) {
        return;
    }

    auto n     = A->dim(0);
    auto lda   = A->stride(0);
    int  lwork = 3 * n;

    BufferVector<AType> work(lwork);

    if constexpr (IsComplexV<AType>) {
        BufferVector<RemoveComplexT<AType>> rwork(std::max((ptrdiff_t)1, 3 * (ptrdiff_t)n - 2));
        if (A->is_gemmable(&lda)) {
            if (W->get_incx() != 1) {
                BufferVector<RemoveComplexT<AType>> temp(A->dim(0));
                blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, temp.data(), work.data(), lwork, rwork.data());

                for (int i = 0; i < W->dim(0); i++) {
                    W->subscript(i) = temp[i];
                }
            } else {
                blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
            }
        } else {
            impl_strided_heev((ComputeEigenvectors) ? 'v' : 'n', A, W, work.data(), rwork.data());
        }
    } else {
        if (A->is_gemmable(&lda)) {
            if (W->get_incx() != 1) {
                BufferVector<RemoveComplexT<AType>> temp(A->dim(0));
                blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, temp.data(), work.data(), lwork);

                for (int i = 0; i < W->dim(0); i++) {
                    W->subscript(i) = temp[i];
                }
            } else {
                blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
            }
        } else {
            impl_strided_syev((ComputeEigenvectors) ? 'v' : 'n', A, W, work.data());
        }
    }
}

template <bool ComputeEigenvectors = true, typename AType>
void heev(einsums::detail::TensorImpl<AType> *A, einsums::detail::TensorImpl<AType> *W) {
    syev(A, W);
}

template <bool ComputeEigenvectors = true, CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires SameUnderlying<AType, WType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<WType, 1>;
        requires NotComplex<AType>;
    }
void syev(AType *A, WType *W) {
    syev<ComputeEigenvectors>(&A->impl(), &W->impl());
}

template <bool ComputeLeftRightEigenvectors = true, CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires std::is_same_v<AddComplexT<typename AType::ValueType>, typename WType::ValueType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<WType, 1>;
    }
void geev(AType *A, WType *W, AType *lvecs, AType *rvecs) {
    EINSUMS_ASSERT(A->dim(0) == A->dim(1));
    EINSUMS_ASSERT(W->dim(0) == A->dim(0));
    EINSUMS_ASSERT(A->dim(0) == lvecs->dim(0));
    EINSUMS_ASSERT(A->dim(1) == lvecs->dim(1));
    EINSUMS_ASSERT(A->dim(0) == rvecs->dim(0));
    EINSUMS_ASSERT(A->dim(1) == rvecs->dim(1));

    blas::geev(ComputeLeftRightEigenvectors ? 'v' : 'n', ComputeLeftRightEigenvectors ? 'v' : 'n', A->dim(0), A->data(), A->stride(0),
               W->data(), lvecs->data(), lvecs->stride(0), rvecs->data(), rvecs->stride(0));
}

template <bool ComputeEigenvectors = true, CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires NotComplex<WType>;
        requires std::is_same_v<typename AType::ValueType, AddComplexT<typename WType::ValueType>>;
        requires MatrixConcept<AType>;
        requires VectorConcept<WType>;
    }
void heev(AType *A, WType *W) {
    syev<ComputeEigenvectors>(&A->impl(), &W->impl());
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType>;
        requires MatrixConcept<AType>;
    }
auto gesv(AType *A, BType *B) -> int {
    auto n   = A->dim(0);
    auto lda = A->dim(0);
    auto ldb = B->dim(1);

    auto nrhs = B->dim(0);

    int                      lwork = n;
    std::vector<blas::int_t> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

template <CoreBasicTensorConcept AType>
void scale(typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(0) * A->stride(0), scale, A->data(), 1);
}

template <CoreBasicTensorConcept AType>
    requires(MatrixConcept<AType>)
void scale_row(size_t row, typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <CoreBasicTensorConcept AType>
void scale_column(size_t col, typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(0), scale, A->data(0ul, col), A->stride(0));
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires VectorConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType>;
    }
auto dot(AType const &A, BType const &B) -> typename AType::ValueType {
    EINSUMS_ASSERT(A.dim(0) == B.dim(0));

    auto result = blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    return result;
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires VectorConcept<AType>;
        requires SameRank<AType, BType>;
        requires !SameUnderlying<AType, BType>;
    }
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    EINSUMS_ASSERT(A.dim(0) == B.dim(0));

    using OutType = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    OutType result = OutType{0.0};

    auto const *A_data   = A.data();
    auto const *B_data   = B.data();
    auto const  A_stride = A.stride(0);
    auto const  B_stride = B.stride(0);

    EINSUMS_OMP_SIMD
    for (size_t i = 0; i < A.dim(0); i++) {
        result += A_data[A_stride * i] * B_data[B_stride * i];
    }

    return result;
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType>;
        requires !VectorConcept<AType>;
    }
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    if (A.full_view_of_underlying() && B.full_view_of_underlying()) {
        Dim<1> dim{1};

        for (size_t i = 0; i < AType::Rank; i++) {
            assert(A.dim(i) == B.dim(i));
            dim[0] *= A.dim(i);
        }

        return dot(TensorView<typename AType::ValueType, 1>(const_cast<AType &>(A), dim),
                   TensorView<typename BType::ValueType, 1>(const_cast<BType &>(B), dim));
    } else {
        auto dims = A.dims();

        std::array<size_t, AType::Rank> strides;
        strides[AType::Rank - 1] = 1;
        std::array<size_t, AType::Rank> index;

        for (int i = AType::Rank - 1; i > 0; i--) {
            strides[i - 1] = strides[i] * dims[i];
        }

        T out{0.0};

        for (size_t sentinel = 0; sentinel < strides[0] * dims[0]; sentinel++) {
            sentinel_to_indices(sentinel, strides, index);
            out += subscript_tensor(A, index) * subscript_tensor(B, index);
        }

        return out;
    }
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType>;
        requires VectorConcept<AType>;
    }
auto true_dot(AType const &A, BType const &B) -> typename AType::ValueType {
    assert(A.dim(0) == B.dim(0));

    if constexpr (IsComplexV<AType>) {
        return blas::dotc(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    } else {
        return blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    }
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires VectorConcept<AType>;
        requires SameRank<AType, BType>;
        requires !SameUnderlying<AType, BType>;
    }
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    EINSUMS_ASSERT(A.dim(0) == B.dim(0));

    using OutType = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    OutType result = OutType{0.0};

    auto const *A_data   = A.data();
    auto const *B_data   = B.data();
    auto const  A_stride = A.stride(0);
    auto const  B_stride = B.stride(0);

    EINSUMS_OMP_SIMD
    for (size_t i = 0; i < A.dim(0); i++) {
        if constexpr (IsComplexV<typename AType::ValueType>) {
            result += A_data[A_stride * i].conj() * B_data[B_stride * i];
        } else {
            result += A_data[A_stride * i] * B_data[B_stride * i];
        }
    }

    return result;
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires SameRank<AType, BType>;
        requires !VectorConcept<AType>;
    }
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    if (A.full_view_of_underlying() && B.full_view_of_underlying()) {
        Dim<1> dim{1};

        for (size_t i = 0; i < AType::Rank; i++) {
            assert(A.dim(i) == B.dim(i));
            dim[0] *= A.dim(i);
        }

        return true_dot(TensorView<typename AType::ValueType, 1>(const_cast<AType &>(A), dim),
                        TensorView<typename BType::ValueType, 1>(const_cast<BType &>(B), dim));
    } else {
        auto dims = A.dims();

        std::array<size_t, AType::Rank> strides;
        strides[AType::Rank - 1] = 1;
        std::array<size_t, AType::Rank> index;

        for (int i = AType::Rank - 1; i > 0; i--) {
            strides[i - 1] = strides[i] * dims[i];
        }

        BiggestTypeT<typename AType::ValueType, typename BType::ValueType> out{0.0};

        for (size_t sentinel = 0; sentinel < strides[0] * dims[0]; sentinel++) {
            sentinel_to_indices(sentinel, strides, index);

            if constexpr (IsComplexV<AType>) {
                out += std::conj(subscript_tensor(A, index)) * subscript_tensor(B, index);
            } else {
                out += subscript_tensor(A, index) * subscript_tensor(B, index);
            }
        }

        return out;
    }
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameRank<AType, BType, CType>
auto dot(AType const &A, BType const &B, CType const &C)
    -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType> {
    Dim<1> dim{1};
    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType>;

    for (size_t i = 0; i < AType::Rank; i++) {
        assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
        dim[0] *= A.dim(i);
    }

    auto vA = TensorView<T, 1>(const_cast<AType &>(A), dim);
    auto vB = TensorView<T, 1>(const_cast<BType &>(B), dim);
    auto vC = TensorView<T, 1>(const_cast<CType &>(C), dim);

    T result{0};
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < dim[0]; i++) {
        result += subscript_tensor(vA, i) * subscript_tensor(vB, i) * subscript_tensor(vC, i);
    }
    return result;
}

template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {
    blas::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
}

template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpby(typename XType::ValueType alpha, XType const &X, typename YType::ValueType beta, YType *Y) {
    blas::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void ger(typename XYType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    blas::ger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <bool TransA, bool TransB, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires MatrixConcept<AType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
    int temp_rows, temp_cols;
    if constexpr (TransA && TransB) {
        EINSUMS_ASSERT(B.dim(0) == A.dim(0) && A.dim(1) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else if constexpr (TransA && !TransB) {
        EINSUMS_ASSERT(B.dim(1) == A.dim(0) && A.dim(1) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    } else if constexpr (!TransA && TransB) {
        EINSUMS_ASSERT(B.dim(0) == A.dim(1) && A.dim(0) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else {
        EINSUMS_ASSERT(B.dim(1) == A.dim(1) && A.dim(0) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    }

    if constexpr (TransA) {
        temp_rows = A.dim(1);
    } else {
        temp_rows = A.dim(0);
    }

    if constexpr (TransB) {
        temp_cols = B.dim(0);
    } else {
        temp_cols = B.dim(1);
    }

    *C = typename CType::ValueType(0.0);

    Tensor<typename AType::ValueType, 2> temp{"temp", temp_rows, temp_cols};

    gemm<TransA, TransB>(typename AType::ValueType{1.0}, A, B, typename CType::ValueType{0.0}, &temp);
    gemm<!TransB, false>(typename AType::ValueType{1.0}, B, temp, typename CType::ValueType{0.0}, C);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_product(typename AType::ValueType alpha, AType const &A, BType const &B, typename CType::ValueType beta, CType *C) {
    LabeledSection0();

    using T = typename AType::ValueType;

    // Ensure the various tensors passed in are the same dimensionality
    if (((C->dims() != A.dims()) || C->dims() != B.dims())) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "direct_product: at least one tensor does not have same dimensionality as destination");
    }

    // Horrible hack. For some reason, in the for loop below, the result could be
    // NAN if the target_value is initially a trash value.
    if constexpr (IsComplexV<typename CType::ValueType>) {
        if (beta == typename CType::ValueType{0.0, 0.0}) {
            C->zero();
        }
    } else {
        if (beta == T(0)) {
            C->zero();
        }
    }

    std::array<size_t, AType::Rank> index_strides;

    size_t elements = dims_to_strides(A.dims(), index_strides);

    if (!A.full_view_of_underlying() || !B.full_view_of_underlying() || !C->full_view_of_underlying()) {
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < elements; item++) {

            size_t A_ord, B_ord, C_ord;

            sentinel_to_sentinels(item, index_strides, A.strides(), A_ord, B.strides(), B_ord, C->strides(), C_ord);

            C->data()[C_ord] = beta * C->data()[C_ord] + alpha * (A.data()[A_ord] * B.data()[B_ord]);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t item = 0; item < elements; item++) {
            C->data()[item] = beta * C->data()[item] + alpha * (A.data()[item] * B.data()[item]);
        }
    }
}

template <CoreBasicTensorConcept AType>
    requires MatrixConcept<AType>
auto pow(AType const &a, typename AType::ValueType alpha,
         typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon())
    -> Tensor<typename AType::ValueType, 2> {
    assert(a.dim(0) == a.dim(1));

    using T = typename AType::ValueType;

    size_t             n      = a.dim(0);
    RemoveViewT<AType> a1     = a;
    RemoveViewT<AType> result = create_tensor_like(a);
    result.set_name("pow result");
    Tensor<RemoveComplexT<T>, 1> e{"e", n};
    result.zero();

    // Diagonalize
    if constexpr (IsComplexV<AType>) {
        hyev<true>(&a1, &e);
    } else {
        syev<true>(&a1, &e);
    }

    RemoveViewT<AType> a2(a1);

    // Determine the largest magnitude of the eigenvalues to use as a scaling factor for the cutoff.

    T max_e{0.0};
    // Block tensors don't have sorted eigenvalues, so we can't make assumptions about ordering.
    for (int i = 0; i < n; i++) {
        if (std::fabs(e(i)) > max_e) {
            max_e = std::fabs(e(i));
        }
    }

    for (size_t i = 0; i < n; i++) {
        if (alpha < 0.0 && std::fabs(e(i)) < cutoff * max_e) {
            e(i) = 0.0;
        } else {
            e(i) = std::pow(e(i), alpha);
            if (!std::isfinite(e(i))) {
                e(i) = 0.0;
            }
        }

        scale_row(i, e(i), &a2);
    }

    gemm<true, false>(1.0, a2, a1, 0.0, &result);

    return result;
}

} // namespace einsums::linear_algebra::detail