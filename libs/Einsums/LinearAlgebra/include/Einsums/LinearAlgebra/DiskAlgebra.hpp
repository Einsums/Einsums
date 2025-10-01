//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/LinearAlgebra/Bases/high_precision.hpp>

#include <thread>

#include "Einsums/Tensor/DiskTensor.hpp"

namespace einsums::linear_algebra::detail {

template <bool Conjugate, DiskTensorConcept AType, DiskTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
auto dot_base(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    constexpr size_t Rank = AType::Rank;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i)) {
            EINSUMS_THROW_EXCEPTION(einsums::tensor_compat_error, "The tensors passed into the dot product must have the same dimensions!");
        }
    }

    BufferAllocator<typename AType::ValueType> A_alloc;
    BufferAllocator<typename BType::ValueType> B_alloc;

    size_t buffer_size = std::min(A_alloc.work_buffer_size(), B_alloc.work_buffer_size());

    if (A.size() <= buffer_size) {
        bool A_reserved = A_alloc.reserve(A.size());
        bool B_reserved = B_alloc.reserve(A.size());

        while (!A_reserved) {
            std::this_thread::yield();
            A_reserved = A_alloc.reserve(A.size());
        }

        while (!B_reserved) {
            std::this_thread::yield();
            B_reserved = B_alloc.reserve(A.size());
        }

        auto A_view = std::apply(A, std::array<einsums::AllT, Rank>());
        auto B_view = std::apply(B, std::array<einsums::AllT, Rank>());

        BiggestTypeT<typename AType::ValueType, typename BType::ValueType> out;

        if constexpr (Conjugate && IsComplexV<typename AType::ValueType>) {
            out = true_dot(A_view.get(), B_view.get());
        } else {
            out = dot(A_view.get(), B_view.get());
        }

        A_alloc.release(A.size());
        B_alloc.release(B.size());

        return out;
    }

    bool A_reserved = A_alloc.reserve(buffer_size);
    bool B_reserved = B_alloc.reserve(buffer_size);

    while (!A_reserved) {
        std::this_thread::yield();
        A_reserved = A_alloc.reserve(buffer_size);
    }

    while (!B_reserved) {
        std::this_thread::yield();
        B_reserved = B_alloc.reserve(buffer_size);
    }

    // Calculate the things needed to loop over the tensors.
    size_t    loop_step = 1, loop_skip = 0;
    ptrdiff_t rank_step = -1, rank_skip = -1;
    size_t    view1_size = 1, remaining_size = 0, step_size = 1;
    bool      found_max = false;

    for (int i = Rank - 1; i >= 0; i--) {
        if (buffer_size > A.dim(i) * view1_size && !found_max) {
            view1_size *= A.dim(i);
        } else if (buffer_size <= A.dim(i) * view1_size && view1_size < buffer_size && !found_max) {
            size_t max_dim = buffer_size / view1_size;
            rank_skip      = i;
            view1_size *= max_dim;
            step_size      = max_dim;
            remaining_size = A.dim(i) % max_dim;
            loop_skip      = A.dim(i) / max_dim;
            found_max      = true;
        } else {
            loop_step *= A.dim(i);
            rank_step = std::max(rank_step, (ptrdiff_t)i);
        }
    }

    if (rank_skip < rank_step) {
        rank_skip = rank_step;
    }

    // Loop over and add.
    einsums::BiggestTypeT<typename AType::ValueType, typename BType::ValueType> big_sum{0.0}, medium_sum{0.0}, small_sum{0.0};

    bool not_big_re = true, not_big_im = true;

    // Set up the indices for the view.
    std::array<Range, Rank> view_indices;

    for (int i = rank_skip; i < Rank; i++) {
        view_indices[i] = Range{0, A.dim(i)};
    }

    for (size_t i = 0; i < loop_step; i++) {
        size_t temp = i;
        for (int k = rank_step; k >= 0; k--) {
            view_indices[k] = Range{temp % A.dim(k), temp % A.dim(k) + 1};
            temp /= A.dim(k);
        }
        for (size_t j = 0; j < loop_skip; j++) {
            // Generate the view.
            view_indices[rank_skip] = Range{j * step_size, (j + 1) * step_size};

            // Find the view.
            auto A_view = std::apply(A, view_indices);
            auto B_view = std::apply(B, view_indices);

            if constexpr (Conjugate && IsComplexV<typename AType::ValueType>) {
                add_scale(true_dot(A_view.get(), B_view.get()), big_sum, medium_sum, small_sum, not_big_re, not_big_im);
            } else {
                add_scale(dot(A_view.get(), B_view.get()), big_sum, medium_sum, small_sum, not_big_re, not_big_im);
            }
        }

        // Handle the remainder.
        if (remaining_size != 0) {
            view_indices[rank_skip] = Range{loop_skip * step_size, loop_skip * step_size + remaining_size};
            auto A_view             = std::apply(A, view_indices);
            auto B_view             = std::apply(B, view_indices);

            if constexpr (Conjugate && IsComplexV<typename AType::ValueType>) {
                add_scale(true_dot(A_view.get(), B_view.get()), big_sum, medium_sum, small_sum, not_big_re, not_big_im);
            } else {
                add_scale(dot(A_view.get(), B_view.get()), big_sum, medium_sum, small_sum, not_big_re, not_big_im);
            }
        }
    }

    A_alloc.release(buffer_size);
    B_alloc.release(buffer_size);
    return combine_accum(big_sum, medium_sum, small_sum);
}

template <DiskTensorConcept AType, DiskTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    return dot_base<false>(A, B);
}

template <DiskTensorConcept AType, DiskTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    return dot_base<true>(A, B);
}

template <DiskTensorConcept AType, DiskTensorConcept BType, DiskTensorConcept CType, typename U>
    requires requires { requires SameRank<AType, BType>; }
void direct_product(U alpha, AType const &A, BType const &B, U beta, CType *C) {
    constexpr size_t Rank = AType::Rank;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i) || A.dim(i) != C->dim(i)) {
            EINSUMS_THROW_EXCEPTION(einsums::tensor_compat_error,
                                    "The tensors passed into the direct product must have the same dimensions!");
        }
    }

    BufferAllocator<typename AType::ValueType> A_alloc;
    BufferAllocator<typename BType::ValueType> B_alloc;
    BufferAllocator<typename CType::ValueType> C_alloc;

    size_t buffer_size = std::min(A_alloc.work_buffer_size(), std::min(B_alloc.work_buffer_size(), C_alloc.work_buffer_size()));

    if (A.size() <= buffer_size) {
        bool A_reserved = A_alloc.reserve(A.size());
        bool B_reserved = B_alloc.reserve(A.size());
        bool C_reserved = C_alloc.reserve(A.size());

        while (!A_reserved) {
            std::this_thread::yield();
            A_reserved = A_alloc.reserve(A.size());
        }

        while (!B_reserved) {
            std::this_thread::yield();
            B_reserved = B_alloc.reserve(A.size());
        }

        while (!C_reserved) {
            std::this_thread::yield();
            C_reserved = C_alloc.reserve(A.size());
        }

        auto A_view = std::apply(A, std::array<einsums::AllT, Rank>());
        auto B_view = std::apply(B, std::array<einsums::AllT, Rank>());
        auto C_view = std::apply(*C, std::array<einsums::AllT, Rank>());

        direct_product(alpha, A_view.get(), B_view.get(), beta, &C_view.get());

        A_alloc.release(A.size());
        B_alloc.release(A.size());
        C_alloc.release(A.size());
        return;
    }

    // Calculate the things needed to loop over the tensors.
    size_t    loop_step = 1, loop_skip = 0;
    ptrdiff_t rank_step = -1, rank_skip = -1;
    size_t    view1_size = 1, remaining_size = 0, step_size = 1;
    bool      found_max = false;

    bool A_reserved = A_alloc.reserve(buffer_size);
    bool B_reserved = B_alloc.reserve(buffer_size);
    bool C_reserved = C_alloc.reserve(buffer_size);

    while (!A_reserved) {
        std::this_thread::yield();
        A_reserved = A_alloc.reserve(buffer_size);
    }

    while (!B_reserved) {
        std::this_thread::yield();
        B_reserved = B_alloc.reserve(buffer_size);
    }

    while (!C_reserved) {
        std::this_thread::yield();
        C_reserved = C_alloc.reserve(buffer_size);
    }

    for (int i = Rank - 1; i >= 0; i--) {
        if (buffer_size > A.dim(i) * view1_size && !found_max) {
            view1_size *= A.dim(i);
        } else if (buffer_size <= A.dim(i) * view1_size && view1_size < buffer_size && !found_max) {
            size_t max_dim = buffer_size / view1_size;
            rank_skip      = i;
            view1_size *= max_dim;
            step_size      = max_dim;
            remaining_size = A.dim(i) % max_dim;
            loop_skip      = A.dim(i) / max_dim;
            found_max      = true;
        } else {
            loop_step *= A.dim(i);
            rank_step = std::max(rank_step, (ptrdiff_t)i);
        }
    }

    if (rank_skip < rank_step) {
        rank_skip = rank_step;
    }

    // Set up the indices for the view.
    std::array<Range, Rank> view_indices;

    for (int i = rank_skip; i < Rank; i++) {
        view_indices[i] = Range{0, A.dim(i)};
    }

    for (size_t i = 0; i < loop_step; i++) {
        size_t temp = i;
        for (int k = rank_step; k >= 0; k--) {
            view_indices[k] = Range{temp % A.dim(k), temp % A.dim(k) + 1};
            temp /= A.dim(k);
        }
        for (size_t j = 0; j < loop_skip; j++) {
            // Generate the view.
            view_indices[rank_skip] = Range{j * step_size, (j + 1) * step_size};

            // Find the view.
            auto A_view = std::apply(A, view_indices);
            auto B_view = std::apply(B, view_indices);
            auto C_view = std::apply(*C, view_indices);

            direct_product(alpha, A_view.get(), B_view.get(), beta, &C_view.get());
        }

        // Handle the remainder.
        if (remaining_size != 0) {
            view_indices[rank_skip] = Range{loop_skip * step_size, loop_skip * step_size + remaining_size};
            // Find the view.
            auto A_view = std::apply(A, view_indices);
            auto B_view = std::apply(B, view_indices);
            auto C_view = std::apply(*C, view_indices);

            direct_product(alpha, A_view.get(), B_view.get(), beta, &C_view.get());
        }
    }

    A_alloc.release(buffer_size);
    B_alloc.release(buffer_size);
    C_alloc.release(buffer_size);
}

template <DiskTensorConcept AType, DiskTensorConcept BType, DiskTensorConcept CType, typename U>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires MatrixConcept<AType>;
    }
void gemm(char transA, char transB, U alpha, AType const &A, BType const &B, U beta, CType *C) {
    // Strassen's algorithm.

    bool tA = (std::tolower(transA) == 'n') ? false : true;
    bool tB = (std::tolower(transB) == 'n') ? false : true;

    // Calculate the parameters.
    size_t C_m = C->dim(0);
    size_t C_n = C->dim(1);

    size_t A_m = (tA) ? A.dim(1) : A.dim(0);
    size_t A_k = (tA) ? A.dim(0) : A.dim(1);

    size_t B_k = (tB) ? B.dim(1) : B.dim(0);
    size_t B_n = (tB) ? B.dim(0) : B.dim(1);

    // We are assuming that we have done some Strassen iterations before, so we need to find the least of these.
    size_t m = std::min(C_m, A_m), n = std::min(C_n, B_n), k = std::min(A_k, B_k);

    // If all parameters are less than 500, then perform the normal matrix multiplication.
    if (m < 500 && n < 500 && k < 500) {
        if (tA && tB) {
            detail::gemm(transA, transB, alpha, A(Range{0, k}, Range{0, m}), B(Range{0, n}, Range{0, k}), beta,
                         &(*C)(Range{0, m}, Range{0, n}));
        } else if (tA) {
            detail::gemm(transA, transB, alpha, A(Range{0, k}, Range{0, m}), B(Range{0, k}, Range{0, n}), beta,
                         &(*C)(Range{0, m}, Range{0, n}));
        } else if (tB) {
            detail::gemm(transA, transB, alpha, A(Range{0, m}, Range{0, k}), B(Range{0, n}, Range{0, k}), beta,
                         &(*C)(Range{0, m}, Range{0, n}));
        } else {
            detail::gemm(transA, transB, alpha, A(Range{0, m}, Range{0, k}), B(Range{0, k}, Range{0, n}), beta,
                         &(*C)(Range{0, m}, Range{0, n}));
        }
    } else {
        // We need to do a Strassen iteration.
        // Start by scaling C.
        if (beta != U{1.0}) {
            scale(beta, C);
        }

        // Next, check for skinny matrices.
        auto max_dim = std::max(m, std::max(n, k));
        auto min_dim = std::min(m, std::min(n, k));

        // If the max is more than double the min, then we have a skinny case.
        if (max_dim > 2 * min_dim) {
            int m_loops = m / min_dim;
            int n_loops = n / min_dim;
            int k_loops = k / min_dim;

            for (int i = 0; i < m_loops; i++) {
                for (int j = 0; j < n_loops; j++) {
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                 B(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}));
                        }
                    }
                }
                if (n - n_loops * min_dim != 0) {
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{i * min_dim, (i + 1) * min_dim}),
                                 B(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                 B(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{i * min_dim, (i + 1) * min_dim}, Range{k_loops * min_dim, k}),
                                 B(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{i * min_dim, (i + 1) * min_dim}, Range{n_loops * min_dim, n}));
                        }
                    }
                }
            }
            if (m - m_loops * min_dim != 0) {
                for (int j = 0; j < n_loops; j++) {
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                 B(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                 B(Range{j * min_dim, (j + 1) * min_dim}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                 B(Range{k_loops * min_dim, k}, Range{j * min_dim, (j + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{j * min_dim, (j + 1) * min_dim}));
                        }
                    }
                }
                if (n - n_loops * min_dim != 0) {
                    for (int l = 0; l < k_loops; l++) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                 B(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{l * min_dim, (l + 1) * min_dim}, Range{m - m_loops * min_dim, m}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{n_loops * min_dim, n}, Range{l * min_dim, (l + 1) * min_dim}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{l * min_dim, (l + 1) * min_dim}),
                                 B(Range{l * min_dim, (l + 1) * min_dim}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        }
                    }

                    if (k - k_loops * min_dim != 0) {
                        if (tA && tB) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                 B(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        } else if (tA) {
                            gemm(transA, transB, alpha, A(Range{k_loops * min_dim, k}, Range{m - m_loops * min_dim, m}),
                                 B(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        } else if (tB) {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                 B(Range{n_loops * min_dim, n}, Range{k_loops * min_dim, k}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        } else {
                            gemm(transA, transB, alpha, A(Range{m - m_loops * min_dim, m}, Range{k_loops * min_dim, k}),
                                 B(Range{k_loops * min_dim, k}, Range{n_loops * min_dim, n}), U{1.0},
                                 &(*C)(Range{m - m_loops * min_dim, m}, Range{n_loops * min_dim, n}));
                        }
                    }
                }
            }
        } else {
            einsums::DiskView<typename AType::ValueType, 2> A_11, A_12, A_21, A_22, A_temp11, A_temp12, A_temp21, A_temp22;
            einsums::DiskView<typename BType::ValueType, 2> B_11, B_12, B_21, B_22, B_temp11, B_temp12, B_temp21, B_temp22;
            einsums::DiskView<typename CType::ValueType, 2> C_11, C_12, C_21, C_22;

            einsums::DiskTensor<typename CType::ValueType, 2> M(C_m - m / 2, C_n - n / 2);
        }
    }
}

} // namespace einsums::linear_algebra::detail