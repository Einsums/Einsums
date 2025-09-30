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
        auto A_view = std::apply(A, std::array<einsums::AllT, Rank>());
        auto B_view = std::apply(B, std::array<einsums::AllT, Rank>());

        if constexpr (Conjugate && IsComplexV<typename AType::ValueType>) {
            return true_dot(A_view.get(), B_view.get());
        } else {
            return dot(A_view.get(), B_view.get());
        }
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

} // namespace einsums::linear_algebra::detail