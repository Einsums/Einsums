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
#include <Einsums/LinearAlgebra/Bases/triangular.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

#include <stdexcept>

#include "Einsums/LinearAlgebra/Bases/norm.hpp"

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

    if (A.dim(0) == 0 || A.dim(1) == 0 || B.dim(0) == 0 || B.dim(1) == 0 || C->dim(0) == 0 || C->dim(1) == 0) {
        return;
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
        if (A.dim(0) != Y->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The dimensions of the input matrix and output tensor do not match! Got {} and {}.", A.dim(1),
                                    Y->dim(0));
        }
        if (A.dim(1) != X.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the input matrix and input tensor do not match! Got {} and {}.",
                                    A.dim(0), X.dim(0));
        }
    } else {
        if (A.dim(1) != Y->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The dimensions of the input matrix and output tensor do not match! Got {} and {}.", A.dim(0),
                                    Y->dim(0));
        }
        if (A.dim(0) != X.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the input matrix and input tensor do not match! Got {} and {}.",
                                    A.dim(1), X.dim(0));
        }
    }

    if (A.dim(0) == 0 || A.dim(1) == 0 || X.dim(0) == 0 || Y->dim(0) == 0) {
        return;
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

            // We might need to transpose the eigenvectors.
            if (A->is_row_major()) {
                for (size_t i = 0; i < A->dim(0); i++) {
                    for (size_t j = i + 1; j < A->dim(1); j++) {
                        AType temp                  = A->subscript_no_check(i, j);
                        A->subscript_no_check(i, j) = A->subscript_no_check(j, i);
                        A->subscript_no_check(j, i) = temp;
                    }
                }
            }
        } else {
            // We can't use LAPACK, so use our own version. Note that the eigenvectors may be off by some complex phase factor.
            // They are still eigenvectors, they just don't match when directly compared with the results of LAPACK.
            // Also, the sizes of the buffers are different.

            if (jobz == 'v' && std::is_same_v<std::complex<float>, AType>) {
                EINSUMS_LOG_WARN("Computing eigenvectors of single-precision matrices with non-unit smallest stride is not fully stable. "
                                 "It works, but there may be some deviation from LAPACK. The stability is likely to improve in the future. "
                                 "Consider copying the data into a freshly constructed Tensor<std::complex<float>,2> instead.");
            }
            lwork = impl_heev_get_work_length(jobz, A, W);

            work.resize(lwork);
            BufferVector<RemoveComplexT<AType>> rwork(std::max((ptrdiff_t)1, 2 * (ptrdiff_t)n - 1));

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

            // We might need to transpose the eigenvectors.
            if (A->is_row_major()) {
                for (size_t i = 0; i < A->dim(0); i++) {
                    for (size_t j = i + 1; j < A->dim(1); j++) {
                        std::swap(A->subscript_no_check(i, j), A->subscript_no_check(j, i));
                    }
                }
            }
        } else {
            // We can't use LAPACK, so use our own version. Note that the eigenvectors may be off by some complex phase factor.
            // They are still eigenvectors, they just don't match when directly compared with the results of LAPACK.
            // Also, the sizes of the buffers are different.
            if (jobz == 'v' && std::is_same_v<float, AType>) {
                EINSUMS_LOG_WARN("Computing eigenvectors of single-precision matrices with non-unit smallest stride is not fully stable. "
                                 "It works, but there may be some deviation from LAPACK. The stability is likely to improve in the future. "
                                 "Consider copying the data into a freshly constructed Tensor<float,2> instead.");
            }
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

template <typename T>
void geev(einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<AddComplexT<T>> *W, einsums::detail::TensorImpl<T> *lvecs,
          einsums::detail::TensorImpl<T> *rvecs) {
    bool A_rank_fail    = A->rank() != 2;
    bool W_rank_fail    = W->rank() != 1;
    bool do_jobvl       = (lvecs != nullptr);
    bool do_jobvr       = (rvecs != nullptr);
    bool lvec_rank_fail = do_jobvl && lvecs->rank() != 2;
    bool rvec_rank_fail = do_jobvr && rvecs->rank() != 2;
    char jobvl = (do_jobvl) ? 'v' : 'n', jobvr = (do_jobvr) ? 'v' : 'n';
    if (A_rank_fail || W_rank_fail || lvec_rank_fail || rvec_rank_fail) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to geev do not have the correct ranks!");
    }

    if (A->dim(0) != A->dim(1) || (do_jobvl && lvecs->dim(0) != lvecs->dim(1)) || (do_jobvr && rvecs->dim(0) != rvecs->dim(1))) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The input tensor and eigenvector outputs need to be square!");
    }

    if (A->dim(0) != W->dim(0) || (do_jobvl && lvecs->dim(0) != A->dim(0)) || (do_jobvr && rvecs->dim(0) != A->dim(0))) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The tensors passed to geev need to have compatible dimensions!");
    }

    if (A->dim(0) == 0) {
        return;
    }

    T              *lvec_data = nullptr, *rvec_data = nullptr, *A_data = A->data();
    AddComplexT<T> *W_data = W->data();
    size_t          lda = A->get_lda(), ldvl = 1, ldvr = 1;

    Tensor<T, 2>              A_temp, lvecs_temp, rvecs_temp;
    Tensor<AddComplexT<T>, 1> W_temp;

    bool A_column_major = A->is_column_major();

    if (!A->is_gemmable()) {
        A_temp         = Tensor<T, 2>{"A temp tensor", A->dim(0), A->dim(1)};
        A_data         = A_temp.data();
        lda            = A_temp.impl().get_lda();
        A_column_major = A_temp.impl().is_column_major();
    }

    if (W->get_incx() != 1) {
        W_temp = Tensor<AddComplexT<T>, 1>{"W temp tensor", W->dim(0)};
        W_data = W_temp.data();
    }

    if (do_jobvl) {
        lvec_data = lvecs->data();
        ldvl      = lvecs->get_lda();
        if (!lvecs->is_gemmable()) {
            lvecs_temp = Tensor<T, 2>{"lvecs temp tensor", lvecs->dim(0), lvecs->dim(1)};
            lvec_data  = lvecs_temp.data();
            ldvl       = lvecs_temp.impl().get_lda();
        }
    }

    if (do_jobvr) {
        rvec_data = rvecs->data();
        ldvr      = rvecs->get_lda();

        if (!rvecs->is_gemmable()) {
            rvecs_temp = Tensor<T, 2>{"rvecs temp tensor", rvecs->dim(0), rvecs->dim(1)};
            rvec_data  = rvecs_temp.data();
            ldvr       = rvecs_temp.impl().get_lda();
        }
    }
    if (A_column_major) {

        auto info = blas::geev(jobvl, jobvr, A->dim(0), A_data, lda, W_data, lvec_data, ldvl, rvec_data, ldvr);

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "The {} argument to geev was invalid! 1 (jobvl): {}, 2 (jobvr): {}, 3 (n): {}, 5 (lda): {}, 8 (ldvl): {}, 10 (ldvr): {}",
                print::ordinal(-info), jobvl, jobvr, A->dim(0), lda, ldvl, ldvr);
        } else if (info > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
        }

        if (do_jobvl) {
            if (!lvecs->is_column_major() && lvec_data == lvecs->data()) {
                for (int i = 0; i < lvecs->dim(0); i++) {
                    for (int j = i + 1; j < lvecs->dim(1); j++) {
                        std::swap(lvecs->subscript_no_check(i, j), lvecs->subscript_no_check(j, i));
                    }
                }
            } else if (lvec_data != lvecs->data()) {
                for (int i = 0; i < lvecs->dim(0); i++) {
                    for (int j = i + 1; j < lvecs->dim(1); j++) {
                        lvecs->subscript_no_check(i, j) = lvecs_temp(j, i);
                    }
                }
            }
        }

        if (do_jobvr) {
            if (!rvecs->is_column_major() && rvec_data == rvecs->data()) {
                for (int i = 0; i < rvecs->dim(0); i++) {
                    for (int j = i + 1; j < rvecs->dim(1); j++) {
                        std::swap(rvecs->subscript_no_check(i, j), rvecs->subscript_no_check(j, i));
                    }
                }
            } else if (rvec_data != rvecs->data()) {
                for (int i = 0; i < rvecs->dim(0); i++) {
                    for (int j = i + 1; j < rvecs->dim(1); j++) {
                        rvecs->subscript_no_check(i, j) = rvecs_temp(j, i);
                    }
                }
            }
        }
    } else {
        auto info = blas::geev(jobvr, jobvl, A->dim(0), A_data, lda, W_data, rvec_data, ldvr, lvec_data, ldvl);

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "The {} argument to geev was invalid! 1 (jobvr): {}, 2 (jobvl): {}, 3 (n): {}, 5 (lda): {}, 8 (ldvr): {}, 10 (ldvl): {}",
                print::ordinal(-info), jobvr, jobvl, A->dim(0), lda, ldvr, ldvl);
        } else if (info > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
        }

        if (do_jobvl) {
            if (!lvecs->is_column_major() && lvec_data == lvecs->data()) {
                for (int i = 0; i < lvecs->dim(0); i++) {
                    for (int j = i + 1; j < lvecs->dim(1); j++) {
                        if constexpr (!IsComplexV<T>) {
                            std::swap(lvecs->subscript_no_check(i, j), lvecs->subscript_no_check(j, i));
                        } else {
                            T temp                          = std::conj(lvecs->subscript_no_check(i, j));
                            lvecs->subscript_no_check(i, j) = std::conj(lvecs->subscript_no_check(j, i));
                            lvecs->subscript_no_check(j, i) = temp;
                        }
                    }
                }

                if constexpr (!IsComplexV<T>) {
                    // Go through and conjugate as well.
                    for (int i = 0; i < lvecs->dim(0); i++) {
                        if (std::imag(W->subscript_no_check(i)) != RemoveComplexT<T>{0.0}) {
                            for (int j = 0; j < lvecs->dim(1); j++) {
                                lvecs->subscript_no_check(j, i + 1) = -lvecs->subscript_no_check(j, i + 1);
                            }
                            i++;
                        }
                    }
                }
            } else if (lvec_data != lvecs->data()) {
                einsums::detail::copy_to(lvecs_temp.impl(), *lvecs);
            }
        }

        if (do_jobvr) {
            if (!rvecs->is_column_major() && rvec_data == rvecs->data()) {
                for (int i = 0; i < rvecs->dim(0); i++) {
                    for (int j = i + 1; j < rvecs->dim(1); j++) {
                        if constexpr (!IsComplexV<T>) {
                            std::swap(rvecs->subscript_no_check(i, j), rvecs->subscript_no_check(j, i));
                        } else {
                            T temp                          = std::conj(rvecs->subscript_no_check(i, j));
                            rvecs->subscript_no_check(i, j) = std::conj(rvecs->subscript_no_check(j, i));
                            rvecs->subscript_no_check(j, i) = temp;
                        }
                    }
                }

                if constexpr (!IsComplexV<T>) {
                    // Go through and conjugate as well.
                    for (int i = 0; i < rvecs->dim(0); i++) {
                        if (std::imag(W->subscript_no_check(i)) != RemoveComplexT<T>{0.0}) {
                            for (int j = 0; j < rvecs->dim(1); j++) {
                                rvecs->subscript_no_check(j, i + 1) = -rvecs->subscript_no_check(j, i + 1);
                            }
                            i++;
                        }
                    }
                }
            } else if (rvec_data != rvecs->data()) {
                einsums::detail::copy_to(rvecs_temp.impl(), *rvecs);
            }
        }
    }

    // if (do_jobvl) {
    //     if (lvecs->is_row_major() && lvec_data == lvecs->data()) {
    //         for (int i = 0; i < lvecs->dim(0); i++) {
    //             for (int j = i + 1; j < lvecs->dim(1); j++) {
    //                 std::swap(lvecs->subscript_no_check(i, j), lvecs->subscript_no_check(j, i));
    //             }
    //         }
    //     } else if (lvec_data != lvecs->data()) {
    //         if (A_column_major) {
    //             for (int i = 0; i < lvecs->dim(0); i++) {
    //                 for (int j = i + 1; j < lvecs->dim(1); j++) {
    //                     lvecs->subscript_no_check(i, j) = lvecs_temp(j, i);
    //                 }
    //             }
    //         } else {
    //             einsums::detail::copy_to(lvecs_temp.impl(), *lvecs);
    //         }
    //     }
    // }
    // if (do_jobvr) {
    //     // == is xor
    //     if ((rvecs->is_column_major() != A_column_major) && rvec_data == rvecs->data()) {
    //         for (int i = 0; i < rvecs->dim(0); i++) {
    //             for (int j = i + 1; j < rvecs->dim(1); j++) {
    //                 std::swap(rvecs->subscript_no_check(i, j), rvecs->subscript_no_check(j, i));
    //             }
    //         }
    //     } else if (rvec_data != rvecs->data()) {
    //         if (A_column_major) {
    //             for (int i = 0; i < rvecs->dim(0); i++) {
    //                 for (int j = i + 1; j < rvecs->dim(1); j++) {
    //                     rvecs->subscript_no_check(i, j) = rvecs_temp(j, i);
    //                 }
    //             }
    //         } else {
    //             einsums::detail::copy_to(rvecs_temp.impl(), *rvecs);
    //         }
    //     }
    // }
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires std::is_same_v<AddComplexT<typename AType::ValueType>, typename WType::ValueType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<WType, 1>;
    }
void geev(AType *A, WType *W, AType *lvecs, AType *rvecs) {
    einsums::detail::TensorImpl<typename AType::ValueType> *lvec_ptr = nullptr, *rvec_ptr = nullptr;

    if (lvecs != nullptr) {
        lvec_ptr = &lvecs->impl();
    }

    if (rvecs != nullptr) {
        rvec_ptr = &rvecs->impl();
    }

    geev(&A->impl(), &W->impl(), lvec_ptr, rvec_ptr);
}

/**
 * @brief Convert the eigenvectors as created by geev into their actual complex forms.
 *
 * geev outputs eigenvectors in a packed real form. If the corresponding eigenvalue is completely real,
 * then the eigenvector is unchanged. However, if the eigenvalue is part of a complex conjugate pair,
 * then the eigenvector will be split across two columns. The first column is the real part of the eigenvector,
 * and the second is the imaginary part. The true eigenvectors for the two columns will then be the plus or minus
 * combinations of the two columns, giving a pair of conjugate vectors. This function converts these vectors from
 * geev and creates the actual complex eigenvectors. It is only needed for real matrices, since complex matrices
 * don't need to pack the eigenvectors in this way.
 */
template <NotComplex T>
void process_geev_vectors(einsums::detail::TensorImpl<AddComplexT<T>> const &evals, einsums::detail::TensorImpl<T> const *lvecs_in,
                          einsums::detail::TensorImpl<T> const *rvecs_in, einsums::detail::TensorImpl<AddComplexT<T>> *lvecs_out,
                          einsums::detail::TensorImpl<AddComplexT<T>> *rvecs_out) {

    if (lvecs_in != nullptr && lvecs_out == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The left output tensor should not be NULL if the left input is not NULL.");
    }

    if (rvecs_in != nullptr && rvecs_out == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The right output tensor should not be NULL if the right input is not NULL.");
    }
    if (lvecs_in == nullptr && rvecs_in == nullptr) {
        return;
    }

    if (evals.rank() != 1 || (lvecs_in != nullptr && lvecs_in->rank() != 2) || (rvecs_in != nullptr && rvecs_in->rank() != 2) ||
        (lvecs_out != nullptr && lvecs_out->rank() != 2) || (rvecs_out != nullptr && rvecs_out->rank() != 2)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The ranks of the tensors passed to process_geev_vectors are incorrect! the eigenvalues need "
                                            "to have rank 1, and the rest need rank 2.");
    }
    if (lvecs_in != nullptr && (evals.dim(0) != lvecs_in->dim(0) || evals.dim(0) != lvecs_in->dim(1))) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of left eigenvector input do not match the eigenvalues!");
    }
    if (rvecs_in != nullptr && (evals.dim(0) != rvecs_in->dim(0) || evals.dim(0) != rvecs_in->dim(1))) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of right eigenvector input do not match the eigenvalues!");
    }
    if (lvecs_out != nullptr && (evals.dim(0) != lvecs_out->dim(0) || evals.dim(0) != lvecs_out->dim(1))) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of left eigenvector output do not match the eigenvalues!");
    }
    if (rvecs_out != nullptr && (evals.dim(0) != rvecs_out->dim(0) || evals.dim(0) != rvecs_out->dim(1))) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of right eigenvector output do not match the eigenvalues!");
    }

    int i = 0;
    while (i < evals.dim(0)) {
        if (std::imag(evals.subscript_no_check(i)) != T{0.0}) {
            if (lvecs_out != nullptr) {
                for (int j = 0; j < evals.dim(0); j++) {
                    lvecs_out->subscript_no_check(j, i) =
                        AddComplexT<T>{lvecs_in->subscript_no_check(j, i), lvecs_in->subscript_no_check(j, i + 1)};
                    lvecs_out->subscript_no_check(j, i + 1) =
                        AddComplexT<T>{lvecs_in->subscript_no_check(j, i), -lvecs_in->subscript_no_check(j, i + 1)};
                }
            }
            if (rvecs_out != nullptr) {
                for (int j = 0; j < evals.dim(0); j++) {
                    rvecs_out->subscript_no_check(j, i) =
                        AddComplexT<T>{rvecs_in->subscript_no_check(j, i), rvecs_in->subscript_no_check(j, i + 1)};
                    rvecs_out->subscript_no_check(j, i + 1) =
                        AddComplexT<T>{rvecs_in->subscript_no_check(j, i), -rvecs_in->subscript_no_check(j, i + 1)};
                }
            }
            i += 2;
        } else {
            if (lvecs_out != nullptr) {
                for (int j = 0; j < evals.dim(0); j++) {
                    lvecs_out->subscript_no_check(j, i) = AddComplexT<T>{lvecs_in->subscript_no_check(j, i)};
                }
            }
            if (rvecs_out != nullptr) {
                for (int j = 0; j < evals.dim(0); j++) {
                    rvecs_out->subscript_no_check(j, i) = AddComplexT<T>{rvecs_in->subscript_no_check(j, i)};
                }
            }
            i++;
        }
    }
}

template <Complex T>
void process_geev_vectors(einsums::detail::TensorImpl<AddComplexT<T>> const &evals, einsums::detail::TensorImpl<T> const &lvecs_in,
                          einsums::detail::TensorImpl<T> const &rvecs_in, einsums::detail::TensorImpl<AddComplexT<T>> *lvecs_out,
                          einsums::detail::TensorImpl<AddComplexT<T>> *rvecs_out) {
    static_assert(false, "process_geev_vectors: Complex inputs to geev don't need to be processed. They already output their full "
                         "eigenvectors. Only real inputs need to be processed.");
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept WType, CoreBasicTensorConcept OutType>
    requires requires {
        requires std::is_same_v<AddComplexT<typename AType::ValueType>, typename WType::ValueType>;
        requires std::is_same_v<AddComplexT<typename AType::ValueType>, typename OutType::ValueType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<OutType, 2>;
        requires RankTensorConcept<WType, 1>;
    }
void process_geev_vectors(WType const &evals, AType const *lvecs_in, AType const *rvecs_in, OutType *lvecs_out, OutType *rvecs_out) {
    process_geev_vectors(evals.impl(), &lvecs_in->impl(), &rvecs_in->impl(), &lvecs_out->impl(), &rvecs_out->impl());
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

template <typename T>
auto gesv(einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<T> *B) -> int {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The coefficient matrix needs to be rank-2!");
    }
    if (B->rank() > 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The output matrix needs to be rank 1 or 2!");
    }

    if (A->dim(0) != A->dim(1) || A->dim(0) != B->dim(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                "The coefficient matrix needs to be square and the number of rows of the result matrix needs to match! A "
                                "dims: ({}, {}), B dim: {}.",
                                A->dim(0), A->dim(1), B->dim(0));
    }

    if (A->dim(0) == 0) {
        return 0;
    }

    if (A->dim(0) == 1) {
        if (B->rank() == 1) {
            B->subscript(0) /= A->subscript(0, 0);
        } else {
            for (int i = 0; i < B->dim(1); i++) {
                B->subscript(0, i) /= A->subscript(0, 0);
            }
        }
        return 0;
    }

    if (A->is_column_major() && B->is_column_major() && A->is_gemmable() && (B->rank() == 1 || B->is_gemmable())) {
        auto n   = A->dim(0);
        auto lda = A->get_lda();

        size_t nrhs, ldb;

        if (B->rank() == 1) {
            nrhs = 1;
            ldb  = n;
        } else {
            nrhs = B->dim(1);
            ldb  = B->get_ldb();
        }

        int                       lwork = n;
        BufferVector<blas::int_t> ipiv(lwork);

        int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
        return info;
    } else {
        BufferVector<blas::int_t> ipiv(A->dim(0));
        if (B->rank() == 2 && B->dim(1) == 0) {
            return impl_lu_decomp(*A, ipiv);
        }
        return impl_solve(*A, *B, ipiv);
    }
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires SameUnderlying<AType, BType>;
        requires MatrixConcept<AType>;
        requires MatrixConcept<BType> || VectorConcept<BType>;
    }
auto gesv(AType *A, BType *B) -> int {
    return gesv(&A->impl(), &B->impl());
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

template <typename T>
void gerc(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> const &Y, einsums::detail::TensorImpl<T> *A) {
    impl_gerc(alpha, X, Y, *A);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void gerc(typename XYType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    gerc(alpha, X.impl(), Y.impl(), &A->impl());
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

    if (a.dim(0) == 0) {
        return Tensor<typename AType::ValueType, 2>{"pow result", 0, 0};
    }

    if (a.dim(0) == 1) {
        Tensor<typename AType::ValueType, 2> out{"pow result", 1, 1};

        out(0, 0) = std::pow(a(0, 0), alpha);
        return out;
    }

    using T = typename AType::ValueType;

    size_t                               n      = a.dim(0);
    Tensor<typename AType::ValueType, 2> a1     = a;
    Tensor<typename AType::ValueType, 2> result = create_tensor_like(a);
    result.set_name("pow result");
    Tensor<RemoveComplexT<T>, 1> e{"e", n};
    result.zero();

    // Diagonalize
    if constexpr (IsComplexV<AType>) {
        hyev<true>(&a1, &e);
    } else {
        syev<true>(&a1, &e);
    }

    Tensor<typename AType::ValueType, 2> a2(a1);

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

        scale_column(i, e(i), &a2);
    }

    gemm('n', 'c', 1.0, a2, a1, 0.0, &result);

    return result;
}

template <typename T, ContiguousContainerOf<blas::int_t> Pivots>
auto getrf(einsums::detail::TensorImpl<T> *A, Pivots *pivot) -> int {
    LabeledSection0();

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only decompose rank-2 tensors!");
    }

    if (A->dim(0) == 0 || A->dim(1) == 0) {
        return 0;
    }

    if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
        // println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
        pivot->resize(std::min(A->dim(0), A->dim(1)));
    }

    int result;

    if (A->is_gemmable() && A->is_column_major()) {
        result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->get_lda(), pivot->data());
    } else {
        result = impl_lu_decomp(*A, *pivot);
    }

    if (result < 0) {
        EINSUMS_LOG_WARN("getrf: argument {} has an invalid value", -result);
    }

    return result;
}

/**
 * @brief Computes the LU factorization of a general m-by-n matrix.
 *
 * The routine computes the LU factorization of a general m-by-n matrix A as
 * \f[
 * A = P*L*U
 * \f]
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements and U is upper triangular. The routine uses
 * partial pivoting, with row interchanges.
 *
 * @tparam TensorType
 * @param A
 * @param pivot
 * @return
 */
template <MatrixConcept TensorType, ContiguousContainerOf<blas::int_t> Pivots>
    requires(CoreTensorConcept<TensorType>)
auto getrf(TensorType *A, Pivots *pivot) -> int {
    return getrf(&A->impl(), pivot);
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
 *
 * @tparam TensorType The type of the tensor
 * @param A The matrix to invert
 * @param pivot The pivot vector from getrf
 * @return int If 0, the execution is successful.
 */
template <typename T, ContiguousContainerOf<blas::int_t> Pivots>
auto getri(einsums::detail::TensorImpl<T> *A, Pivots const &pivot) -> int {
    LabeledSection0();

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only compute the inverses of matrices.");
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only compute the inverses of square matrices.");
    }

    int result;

    if (A->is_gemmable() && A->is_column_major()) {
        result = blas::getri(A->dim(0), A->data(), A->get_lda(), pivot.data());
    } else {
        BufferVector<T> work(A->dim(0));
        result = impl_invert_lu(*A, pivot, work.data());
    }

    if (result < 0) {
        EINSUMS_LOG_WARN("getri: argument {} has an invalid value", -result);
    }

    return result;
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
 *
 * @tparam TensorType The type of the tensor
 * @param A The matrix to invert
 * @param pivot The pivot vector from getrf
 * @return int If 0, the execution is successful.
 */
template <MatrixConcept TensorType, ContiguousContainerOf<blas::int_t> Pivots>
    requires(CoreTensorConcept<TensorType>)
auto getri(TensorType *A, Pivots const &pivot) -> int {
    return getri(&A->impl(), pivot);
}

template <typename T>
void invert(einsums::detail::TensorImpl<T> *A) {
    LabeledSection0();

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only compute the inverses of matrices.");
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only compute the inverses of square matrices.");
    }

    BufferVector<blas::int_t> pivot(A->dim(0));
    int                       result;

    if (A->is_gemmable()) {
        result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->get_lda(), pivot.data());
    } else {
        result = impl_lu_decomp(*A, pivot);
    }

    if (result < 0) {
        EINSUMS_LOG_WARN("getrf: argument {} has an invalid value", -result);
    }

    if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed",
                                result - 1, result - 1);
    }

    if (A->is_gemmable()) {
        result = blas::getri(A->dim(0), A->data(), A->get_lda(), pivot.data());
    } else {
        BufferVector<T> work(A->dim(0));
        result = impl_invert_lu(*A, pivot, work.data());
    }

    if (result < 0) {
        EINSUMS_LOG_WARN("getri: argument {} has an invalid value", -result);
    }

    if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed",
                                result - 1, result - 1);
    }
}

/**
 * @brief Inverts a matrix.
 *
 * Utilizes the LAPACK routines getrf and getri to invert a matrix.
 *
 * @tparam TensorType The type of the tensor
 * @param A Matrix to invert. On exit, the inverse of A.
 */
template <MatrixConcept TensorType>
    requires(CoreTensorConcept<TensorType>)
void invert(TensorType *A) {
    invert(&A->impl());
}

template <typename T>
auto svd(einsums::detail::TensorImpl<T> const &_A) -> std::tuple<Tensor<T, 2>, Tensor<RemoveComplexT<T>, 1>, Tensor<T, 2>> {
    LabeledSection0();

    if (_A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only decompose matrices!");
    }

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m   = A.dim(0);
    size_t n   = A.dim(1);
    size_t lda = A.impl().get_lda();

    // Test if it is absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    U.zero();
    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    Vt.zero();
    auto superb = create_tensor<T>("superb", std::min(m, n));
    superb.zero();

    //    int info{0};
    int info =
        blas::gesvd('A', 'A', m, n, A.data(), lda, S.data(), U.data(), U.impl().get_lda(), Vt.data(), Vt.impl().get_lda(), superb.data());

    if (info != 0) {
        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "svd: Argument {} has an invalid value.\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #6 (lda) = {}, #8 (m) = {}", -info, m, n,
                lda, U.impl().get_lda(), Vt.impl().get_lda());
        } else {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "svd: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto svd(AType const &A) -> std::tuple<Tensor<typename AType::ValueType, 2>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>,
                                       Tensor<typename AType::ValueType, 2>> {
    return svd(A.impl());
}

template <typename T>
auto norm(char norm_type, einsums::detail::TensorImpl<T> const &a) -> RemoveComplexT<T> {
    LabeledSection0();

    if (a.rank() > 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "A norm can only be taken on a matrix or vector!");
    }

    if constexpr (blas::IsBlasableV<T>) {
        switch (norm_type) {
        case 'm':
        case 'M':
            if (a.rank() == 1) {
                return impl_max_abs_norm(a);
            } else if (a.is_gemmable()) {
                return impl_max_abs_norm_gemmable(a);
            } else {
                return impl_max_abs_norm(a);
            }
        case '1':
        case 'o':
        case 'O':
            if (a.rank() == 1) {
                return blas::sum1(a.dim(0), a.data(), a.get_incx());
            } else if (a.is_gemmable()) {
                return impl_one_norm_gemmable(a);
            } else {
                return impl_one_norm(a);
            }
        case 'i':
        case 'I':
            if (a.rank() == 1) {
                return impl_max_abs_norm(a);
            } else if (a.is_gemmable()) {
                return impl_infinity_norm_gemmable(a);
            } else {
                return impl_infinity_norm(a);
            }
        case 'e':
        case 'f':
        case 'E':
        case 'F':
            if (a.rank() == 1) {
                return blas::nrm2(a.dim(0), a.data(), a.get_incx());
            } else if (a.is_gemmable()) {
                return impl_frobenius_norm_gemmable(a);
            } else {
                return impl_frobenius_norm(a);
            }
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The norm type passed to norm is not valid!");
        }
    } else {
        switch (norm_type) {
        case 'm':
        case 'M':
            return impl_max_abs_norm(a);
        case '1':
        case 'o':
        case 'O':
            return impl_one_norm(a);
        case 'i':
        case 'I':
            return impl_infinity_norm(a);
        case 'e':
        case 'f':
        case 'E':
        case 'F':
            return impl_frobenius_norm(a);
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The norm type passed to norm is not valid!");
        }
    }
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto norm(char norm_type, AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    return norm(norm_type, a.impl());
}

template <typename T>
auto vec_norm(einsums::detail::TensorImpl<T> const &a) -> RemoveComplexT<T> {
    RemoveComplexT<T> norm = 0.0, scale = 1.0;

    sum_square(a, &scale, &norm);

    return std::sqrt(norm) * scale;
}

template <TensorConcept AType>
auto vec_norm(AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    return vec_norm(a.impl());
}

} // namespace einsums::linear_algebra::detail