//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>
#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/hipBLAS.hpp>
#endif

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename AType, typename BType, typename CType>
void impl_gemm_noncontiguous(char transA, char transB, CType alpha, einsums::detail::TensorImpl<AType> const &A,
                             einsums::detail::TensorImpl<BType> const &B, CType beta, einsums::detail::TensorImpl<CType> *C) {
    LabeledSection0();

    char const   tA = std::tolower(transA), tB = std::tolower(transB);
    size_t const a_target_stride = (tA == 'n') ? A.stride(0) : A.stride(1), a_link_stride = (tA == 'n') ? A.stride(1) : A.stride(0);
    size_t const b_target_stride = (tB == 'n') ? B.stride(1) : B.stride(0), b_link_stride = (tB == 'n') ? B.stride(0) : B.stride(1);
    size_t const ca_stride = C->stride(0), cb_stride = C->stride(1);
    size_t const m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);

    AType const *A_data = A.data();
    BType const *B_data = B.data();
    CType       *C_data = C->data();

    constexpr CType zero = einsums::detail::convert<int, CType>(0);

    // Scale the output.
    if (beta == zero) {
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t ca = 0; ca < m; ca++) {
            for (size_t cb = 0; cb < n; cb++) {
                C_data[ca * ca_stride + cb * cb_stride] = zero;
            }
        }
    } else {
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t ca = 0; ca < m; ca++) {
            for (size_t cb = 0; cb < n; cb++) {
                C_data[ca * ca_stride + cb * cb_stride] *= beta;
            }
        }
    }

    // Do the matrix multiplication.
    if constexpr (IsComplexV<CType>) {
        if (tA == 'c' && tB == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for collapse(3))
            for (size_t link = 0; link < k; link++) {
                for (size_t ca = 0; ca < m; ca++) {
                    for (size_t cb = 0; cb < n; cb++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * std::conj(A_data[a_target_stride * ca + a_link_stride * link]) *
                                                                   std::conj(B_data[b_target_stride * cb + b_link_stride * link]);
                    }
                }
            }
        } else if (tA == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for collapse(3))
            for (size_t link = 0; link < k; link++) {
                for (size_t ca = 0; ca < m; ca++) {
                    for (size_t cb = 0; cb < n; cb++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * std::conj(A_data[a_target_stride * ca + a_link_stride * link]) *
                                                                   B_data[b_target_stride * cb + b_link_stride * link];
                    }
                }
            }
        } else if (tB == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for collapse(3))
            for (size_t link = 0; link < k; link++) {
                for (size_t ca = 0; ca < m; ca++) {
                    for (size_t cb = 0; cb < n; cb++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * A_data[a_target_stride * ca + a_link_stride * link] *
                                                                   std::conj(B_data[b_target_stride * cb + b_link_stride * link]);
                    }
                }
            }
        } else {
            EINSUMS_OMP_PRAGMA(parallel for collapse(3))
            for (size_t link = 0; link < k; link++) {
                for (size_t ca = 0; ca < m; ca++) {
                    for (size_t cb = 0; cb < n; cb++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * A_data[a_target_stride * ca + a_link_stride * link] *
                                                                   B_data[b_target_stride * cb + b_link_stride * link];
                    }
                }
            }
        }
    } else {
        EINSUMS_OMP_PRAGMA(parallel for collapse(3))
        for (size_t link = 0; link < k; link++) {
            for (size_t ca = 0; ca < m; ca++) {
                for (size_t cb = 0; cb < n; cb++) {
                    C_data[ca * ca_stride + cb * cb_stride] +=
                        alpha * A_data[a_target_stride * ca + a_link_stride * link] * B_data[b_target_stride * cb + b_link_stride * link];
                }
            }
        }
    }
}

template <typename T>
void impl_gemm_contiguous(char transA, char transB, T alpha, einsums::detail::TensorImpl<T> const &A,
                          einsums::detail::TensorImpl<T> const &B, T beta, einsums::detail::TensorImpl<T> *C) {
    char tA = std::tolower(transA), tB = std::tolower(transB);
    bool colA = A.is_column_major(), colB = B.is_column_major(), colC = C->is_column_major();
    bool rowA = A.is_row_major(), rowB = B.is_row_major(), rowC = C->is_row_major();

    // So many cases...
    if (colA && colB && colC) {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        blas::gemm(transA, transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
    } else if (rowA && colB && colC) {
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
                blas::gemm('n', transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tA == 't') {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
            blas::gemm('n', transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(1);
            blas::gemm('t', transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        }
    } else if (colA && rowB && colC) {
        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
                blas::gemm(transA, 'n', m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tB == 't') {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm(transA, 'n', m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm(transA, 't', m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        }
    } else if (rowA && rowB && colC) {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tA = 'n';
            }
        } else if (tA == 't') {
            tA = 'n';
        } else {
            tA = 't';
        }

        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tB = 'n';
            }
        } else if (tB == 't') {
            tB = 'n';
        } else {
            tB = 't';
        }

        blas::gemm(tA, tB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
    } else if (colA && colB && rowC) {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tA = 'n';
            }
        } else if (tA == 't') {
            tA = 'n';
        } else {
            tA = 't';
        }

        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tB = 'n';
            }
        } else if (tB == 't') {
            tB = 'n';
        } else {
            tB = 't';
        }

        blas::gemm(tB, tA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
    } else if (rowA && colB && rowC) {
        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
                blas::gemm('n', transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tB == 't') {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm('n', transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm('t', transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        }
    } else if (colA && rowB && rowC) {
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
                blas::gemm(transB, 'n', n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tA == 't') {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
            blas::gemm(transB, 'n', n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(1);
            blas::gemm(transB, 't', n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        }
    } else {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        blas::gemm(transB, transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
    }
}

template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void impl_gemm(char transA, char transB, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A,
               einsums::detail::TensorImpl<BType> const &B, BetaType beta, einsums::detail::TensorImpl<CType> *C);

#ifdef EINSUMS_COMPUTE_CODE
template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
bool impl_gemm_gpu(char transA, char transB, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A,
                   einsums::detail::TensorImpl<BType> const &B, BetaType beta, einsums::detail::TensorImpl<CType> *C) {
    if constexpr (std::is_same_v<AType, BType> && std::is_same_v<AType, CType> && blas::IsBlasableV<AType>) {
        using T = AType;

        if (A.size() >= 1024 && B.size() >= 1024 && C->size() >= 1024 && A.dim(0) <= 500 && A.dim(1) <= 500 && B.dim(0) <= 500 &&
            B.dim(1) <= 500 && C->dim(0) <= 500 && C->dim(1) <= 500) {
            try {
                char tA = std::tolower(transA), tB = std::tolower(transB);
                auto A_block = A.gpu_cache_tensor();
                auto B_block = B.gpu_cache_tensor();
                auto C_block = C->gpu_cache_tensor();

                if (A_block && B_block && C_block) {

                    auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);

                    blas::gpu::gemm(transA, transB, m, n, k, (T)alpha, A.get_gpu_pointer().get(), A.dim(0), B.get_gpu_pointer().get(),
                                    B.dim(0), (T)beta, C->get_gpu_pointer().get(), C->dim(0));
                    gpu::stream_wait();
                    C->increment_gpu_modify();

                    return true;
                }
                return false;
            } catch (std::runtime_error &e) {
                return false; // We couldn't allocate all the data, so don't do the GPU algorithm.
            }
        } else if (A.size() >= 1024 && B.size() >= 1024 && C->size() >= 1024) {
            bool tA = std::tolower(transA) != 'n', tB = std::tolower(transB) != 'n';
            int  min_dim = 500;

            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);

            int m_loops = m / min_dim;
            int n_loops = n / min_dim;
            int k_loops = k / min_dim;

            auto m_dim = std::min((int)m, min_dim);
            auto n_dim = std::min((int)n, min_dim);
            auto k_dim = std::min((int)k, min_dim);

            if (beta != BetaType{1.0}) {
                einsums::detail::impl_scal(beta, *C);
            }

            for (int i = 0; i < m_loops; i++) {
                for (int j = 0; j < n_loops; j++) {
                    auto C_view = C->subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim});
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                      B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        } else if (tA) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                      B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        } else if (tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        } else {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), BetaType{1.0}, &C_view);
                        } else if (tA) {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                B.subscript(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), BetaType{1.0}, &C_view);
                        } else if (tB) {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), BetaType{1.0}, &C_view);
                        } else {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                B.subscript(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), BetaType{1.0}, &C_view);
                        }
                    }
                    C_view.tensor_from_gpu();
                }
                if (n - n_loops * min_dim != 0) {
                    auto C_view = C->subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n});
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                      B.subscript(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), BetaType{1.0},
                                      &C_view);
                        } else if (tA) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                      B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), BetaType{1.0},
                                      &C_view);
                        } else if (tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      B.subscript(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), BetaType{1.0},
                                      &C_view);
                        } else {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), BetaType{1.0},
                                      &C_view);
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                      B.subscript(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), BetaType{1.0}, &C_view);
                        } else if (tA) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                      B.subscript(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), BetaType{1.0}, &C_view);
                        } else if (tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                      B.subscript(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), BetaType{1.0}, &C_view);
                        } else {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                      B.subscript(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), BetaType{1.0}, &C_view);
                        }
                    }
                    C_view.tensor_from_gpu();
                }
            }
            if (m - m_loops * min_dim != 0) {
                for (int j = 0; j < n_loops; j++) {
                    auto C_view = C->subscript(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim});
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                      B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        } else if (tA) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                      B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        } else if (tB) {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                      B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        } else {
                            impl_gemm(transA, transB, alpha,
                                      A.subscript(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                      B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}),
                                      BetaType{1.0}, &C_view);
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                      B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), BetaType{1.0},
                                      &C_view);
                        } else if (tA) {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                      B.subscript(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), BetaType{1.0},
                                      &C_view);
                        } else if (tB) {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                      B.subscript(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), BetaType{1.0},
                                      &C_view);
                        } else {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                      B.subscript(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), BetaType{1.0},
                                      &C_view);
                        }
                    }
                    C_view.tensor_from_gpu();
                }
                if (n - n_loops * min_dim != 0) {
                    auto C_view = C->subscript(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n});
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                B.subscript(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), BetaType{1.0}, &C_view);
                        } else if (tA) {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), BetaType{1.0}, &C_view);
                        } else if (tB) {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                B.subscript(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), BetaType{1.0}, &C_view);
                        } else {
                            impl_gemm(
                                transA, transB, alpha, A.subscript(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                B.subscript(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), BetaType{1.0}, &C_view);
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                      B.subscript(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), BetaType{1.0}, &C_view);
                        } else if (tA) {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                      B.subscript(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), BetaType{1.0}, &C_view);
                        } else if (tB) {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                      B.subscript(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), BetaType{1.0}, &C_view);
                        } else {
                            impl_gemm(transA, transB, alpha, A.subscript(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                      B.subscript(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), BetaType{1.0}, &C_view);
                        }
                    }
                    C_view.tensor_from_gpu();
                }
            }
            return true;
        }
        return false;
    } else {
        return false;
    }
}
#else
template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
constexpr bool impl_gemm_gpu(char transA, char transB, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A,
                             einsums::detail::TensorImpl<BType> const &B, BetaType beta, einsums::detail::TensorImpl<CType> *C) {
    return false;
}
#endif

template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void impl_gemm(char transA, char transB, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A,
               einsums::detail::TensorImpl<BType> const &B, BetaType beta, einsums::detail::TensorImpl<CType> *C) {
    bool did_gpu = impl_gemm_gpu(transA, transB, alpha, A, B, beta, C);

    if (!did_gpu) {
#ifdef EINSUMS_COMPUTE_CODE
        C->tensor_from_gpu();
        C->increment_core_modify();
#endif
        if constexpr (!std::is_same_v<AType, BType> || !std::is_same_v<AType, CType>) {
            impl_gemm_noncontiguous(transA, transB, einsums::detail::convert<AlphaType, CType>(alpha), A, B,
                                    einsums::detail::convert<AlphaType, CType>(beta), C);
        } else {
            if (A.is_gemmable() && B.is_gemmable() && C->is_gemmable()) {
                impl_gemm_contiguous(transA, transB, einsums::detail::convert<AlphaType, CType>(alpha), A, B,
                                     einsums::detail::convert<AlphaType, CType>(beta), C);
            } else {
                impl_gemm_noncontiguous(transA, transB, einsums::detail::convert<AlphaType, CType>(alpha), A, B,
                                        einsums::detail::convert<AlphaType, CType>(beta), C);
            }
        }
    }
}

#ifndef DOXYGEN

extern template EINSUMS_EXPORT void impl_gemm<float, float, float, float, float>(char transA, char transB, float alpha,
                                                                                 einsums::detail::TensorImpl<float> const &A,
                                                                                 einsums::detail::TensorImpl<float> const &B, float beta,
                                                                                 einsums::detail::TensorImpl<float> *C);
extern template EINSUMS_EXPORT void impl_gemm<double, double, double, double, double>(char transA, char transB, double alpha,
                                                                                      einsums::detail::TensorImpl<double> const &A,
                                                                                      einsums::detail::TensorImpl<double> const &B,
                                                                                      double beta, einsums::detail::TensorImpl<double> *C);
extern template EINSUMS_EXPORT void
impl_gemm<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>>(
    char transA, char transB, std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
extern template EINSUMS_EXPORT void
impl_gemm<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>>(
    char transA, char transB, std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);
#endif
} // namespace detail
} // namespace linear_algebra
} // namespace einsums