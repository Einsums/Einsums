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
#include <Einsums/Errors/Error.hpp>
#include <Einsums/LinearAlgebra/Bases/direct_product.hpp>
#include <Einsums/LinearAlgebra/Bases/dot.hpp>
#include <Einsums/LinearAlgebra/Bases/gemm.hpp>
#include <Einsums/LinearAlgebra/Bases/gemv.hpp>
#include <Einsums/LinearAlgebra/Bases/ger.hpp>
#include <Einsums/LinearAlgebra/Bases/sum_square.hpp>
#include <Einsums/LinearAlgebra/Bases/syev.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

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

    size_t const n   = A->dim(0);
    size_t       lda = 0;

    blas::int_t lwork;

    BufferVector<AType> work;

    constexpr char jobz = (ComputeEigenvectors) ? 'v' : 'n';

    if constexpr (IsComplexV<AType>) {
        // Check if we can use LAPACK.
        if (A->is_gemmable(&lda)) {
            // Transpose A if necessary.
            if constexpr (ComputeEigenvectors) {
                if (A->is_row_major()) {
                    for (size_t i = 0; i < n; i++) {
                        for (size_t j = i + 1; j < n; j++) {
                            std::swap(A->subscript(i, j), A->subscript(j, i));
                        }
                    }
                }
            }

            // Query buffer params.
            AType lwork_complex = AType{(RemoveComplexT<AType>)(2 * n - 1)};

            blas::heev(jobz, 'u', n, A->data(), lda, (RemoveComplexT<AType> *)nullptr, &lwork_complex, -1,
                       (RemoveComplexT<AType> *)nullptr);

            lwork = (blas::int_t)std::real(lwork_complex);

            work.resize(lwork);

            // Set up real work buffer.
            BufferVector<RemoveComplexT<AType>> rwork(std::max((ptrdiff_t)1, 3 * (ptrdiff_t)n - 2));

            // Check if we need to make a temporary buffer for the eigenvalues.
            if (W->get_incx() != 1) {

                // Make a temporary buffer for the eigenvalues, then copy after.
                BufferVector<RemoveComplexT<AType>> temp_vals(A->dim(0));
                blas::heev(jobz, 'u', n, A->data(), lda, temp_vals.data(), work.data(), lwork, rwork.data());

                for (int i = 0; i < W->dim(0); i++) {
                    W->subscript(i) = temp_vals[i];
                }
            } else {
                // No temporary buffer needed.
                blas::heev(jobz, 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
            }
        } else {
            // We can't use LAPACK, so use our own version. Note that the eigenvectors may be off by some complex phase factor.
            // They are still eigenvectors, they just don't match when directly compared with the results of LAPACK.
            // Also, the sizes of the buffers are different.
            lwork = impl_heev_get_work_length(jobz, A, W);

            work.resize(lwork);
            BufferVector<RemoveComplexT<AType>> rwork(std::max((ptrdiff_t)1, 2 * (ptrdiff_t)n + 2));
            rwork[n]         = 0.0;
            rwork[2 * n]     = 0.0;
            rwork[2 * n + 1] = 0.0;

            impl_strided_heev(jobz, A, W, work.data(), rwork.data());
        }
    } else {
        // Check if we can use LAPACK.
        if (A->is_gemmable(&lda)) {
            // Query buffer params.
            AType lwork_real = (AType)(3 * n - 2);

            blas::syev(jobz, 'u', n, A->data(), lda, (AType *)nullptr, &lwork_real, -1);

            lwork = (blas::int_t)lwork_real;

            work.resize(lwork);

            // Check
            if (W->get_incx() != 1) {
                BufferVector<RemoveComplexT<AType>> temp(A->dim(0));
                blas::syev(jobz, 'u', n, A->data(), lda, temp.data(), work.data(), lwork);

                for (int i = 0; i < W->dim(0); i++) {
                    W->subscript(i) = temp[i];
                }
            } else {
                blas::syev(jobz, 'u', n, A->data(), lda, W->data(), work.data(), lwork);
            }
        } else {
            // We can't use LAPACK, so use our own version. Note that the eigenvectors may be off by some complex phase factor.
            // They are still eigenvectors, they just don't match when directly compared with the results of LAPACK.
            // Also, the sizes of the buffers are different.
            lwork = impl_syev_get_work_length(jobz, A, W);

            work.resize(lwork);
            impl_strided_syev(jobz, A, W, work.data());
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
    auto lda = A->impl().get_lda();
    auto ldb = B->impl().get_lda();

    auto nrhs = B->dim(0);

    int                      lwork = n;
    std::vector<blas::int_t> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

template <typename T>
void scale(T scale, einsums::detail::TensorImpl<T> *A) {
    einsums::detail::impl_scal(scale, *A);
}

template <CoreBasicTensorConcept AType>
void scale(typename AType::ValueType scale, AType *A) {
    detail::scale(scale, &A->impl());
}

template <typename T>
void scale_row(ptrdiff_t row, T scale, einsums::detail::TensorImpl<T> *A) {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The input to scale_row needs to be a rank-2 tensor!");
    }
    blas::scal(A->dim(1), scale, A->data(row, 0), A->stride(1));
}

template <CoreBasicTensorConcept AType>
    requires(MatrixConcept<AType>)
void scale_row(ptrdiff_t row, typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <typename T>
void scale_column(ptrdiff_t col, T scale, einsums::detail::TensorImpl<T> *A) {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The input to scale_column needs to be a rank-2 tensor!");
    }
    blas::scal(A->dim(1), scale, A->data(0, col), A->stride(1));
}

template <CoreBasicTensorConcept AType>
void scale_column(ptrdiff_t col, typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(0), scale, A->data(0, col), A->stride(0));
}

template <typename T, typename TOther>
BiggestTypeT<T, TOther> dot(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<TOther> const &B) {
    if (A.rank() != B.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The ranks of the tensors passed to dot must be the same!");
    }

    if (A.dims() != B.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of the tensors passed to dot must be the same!");
    }

    return impl_dot(A, B);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    return dot(A.impl(), B.impl());
}

template <typename T>
T true_dot(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &B) {
    if (A.rank() != B.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The ranks of the tensors passed to true_dot must be the same!");
    }

    if (A.dims() != B.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of the tensors passed to true_dot must be the same!");
    }

    return impl_true_dot(A, B);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    return true_dot(A.impl(), B.impl());
}

template <typename AType, typename BType, typename CType>
auto dot(einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B,
         einsums::detail::TensorImpl<CType> const &C) -> BiggestTypeT<AType, BType, CType> {
    return impl_dot(A, B, C);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameRank<AType, BType, CType>
auto dot(AType const &A, BType const &B, CType const &C)
    -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType> {

    return dot(A.impl(), B.impl(), C.impl());
}

template <typename T>
void axpy(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> *Y) {
    einsums::detail::impl_axpy(alpha, X, *Y);
}

template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {
    axpy(alpha, X.impl(), &Y->impl());
}

template <typename T>
void axpby(T alpha, einsums::detail::TensorImpl<T> const &X, T beta, einsums::detail::TensorImpl<T> *Y) {
    einsums::detail::impl_scal(beta, *Y);
    einsums::detail::impl_axpy(alpha, X, *Y);
}

template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpby(typename XType::ValueType alpha, XType const &X, typename YType::ValueType beta, YType *Y) {
    axpby(alpha, X.impl(), beta, &Y->impl());
}

template <typename T>

void ger(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> const &Y, einsums::detail::TensorImpl<T> *A) {
    impl_ger(alpha, X, Y, *A);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void ger(typename XYType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    ger(alpha, X.impl(), Y.impl(), &A->impl());
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

template <typename AType, typename BType, typename CType>
void direct_product(CType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B, CType beta,
                    einsums::detail::TensorImpl<CType> *C) {
    impl_direct_product(alpha, A, B, beta, C);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_product(typename AType::ValueType alpha, AType const &A, BType const &B, typename CType::ValueType beta, CType *C) {
    direct_product(alpha, A.impl(), B.impl(), beta, &C->impl());
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