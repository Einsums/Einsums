//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Assert.hpp>
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

namespace einsums::linear_algebra::detail {

template <CoreBasicTensorConcept AType>
    requires(RankTensorConcept<AType, 1>)
void sum_square(AType const &a, RemoveComplexT<typename AType::ValueType> *scale, RemoveComplexT<typename AType::ValueType> *sumsq) {
    int n    = a.dim(0);
    int incx = a.stride(0);
    blas::lassq(n, a.data(), incx, scale, sumsq);
}

template <bool TransA, bool TransB, typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires RankTensorConcept<AType, 2>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires(std::convertible_to<U, typename AType::ValueType>);
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    blas::gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, static_cast<typename AType::ValueType>(alpha), A.data(), lda, B.data(), ldb,
               static_cast<typename AType::ValueType>(beta), C->data(), ldc);
}

template <bool TransA, typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>

    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<XType, 1>;
        requires RankTensorConcept<YType, 1>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    auto m = A.dim(0), n = A.dim(1);
    auto lda  = A.stride(0);
    auto incx = z.stride(0);
    auto incy = y->stride(0);

    blas::gemv(TransA ? 't' : 'n', m, n, static_cast<typename AType::ValueType>(alpha), A.data(), lda, z.data(), incx,
               static_cast<typename AType::ValueType>(beta), y->data(), incy);
}

template <bool ComputeEigenvectors = true, CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires SameUnderlying<AType, WType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<WType, 1>;
        requires NotComplex<AType>;
    }
void syev(AType *A, WType *W) {
    assert(A->dim(0) == A->dim(1));

    auto                                   n     = A->dim(0);
    auto                                   lda   = A->stride(0);
    int                                    lwork = 3 * n;
    std::vector<typename AType::ValueType> work(lwork);

    blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
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
    EINSUMS_ASSERT(A->dim(0) == A->dim(1));

    auto                                   n     = A->dim(0);
    auto                                   lda   = A->stride(0);
    int                                    lwork = 2 * n;
    std::vector<typename AType::ValueType> work(lwork);
    std::vector<typename WType::ValueType> rwork(3 * n);

    blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
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
    auto const  A_stride = A.strides(0);
    auto const  B_stride = B.strides(0);

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

        BiggestTypeT<typename AType::ValueType, typename BType::ValueType> out{0.0};

        for (size_t sentinel = 0; sentinel < strides[0] * dims[0]; sentinel++) {
            sentinel_to_indices(sentinel, strides, index);

            out += std::apply(A, index) * std::apply(B, index);
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
    auto const  A_stride = A.strides(0);
    auto const  B_stride = B.strides(0);

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
                out += std::conj(std::apply(A, index)) * std::apply(B, index);
            } else {
                out += std::apply(A, index) * std::apply(B, index);
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
        result += vA(i) * vB(i) * vC(i);
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
    auto target_dims = get_dim_ranges<CType::Rank>(*C);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);
    using T          = typename AType::ValueType;

    // Ensure the various tensors passed in are the same dimensionality
    if (((C->dims() != A.dims()) || C->dims() != B.dims())) {
        println_abort("direct_product: at least one tensor does not have same dimensionality as destination");
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

    EINSUMS_OMP_PARALLEL_FOR
    for (auto it = view.begin(); it != view.end(); it++) {
        T &target_value = std::apply(*C, *it);
        T  AB_product   = std::apply(A, *it) * std::apply(B, *it);
        target_value    = beta * target_value + alpha * AB_product;
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