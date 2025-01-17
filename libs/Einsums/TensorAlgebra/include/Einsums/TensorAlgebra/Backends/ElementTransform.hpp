//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace einsums::tensor_algebra {

namespace detail {}

template <template <typename, size_t> typename CType, size_t CRank, typename UnaryOperator, typename T>
    requires requires {
        requires TensorConcept<CType<T, CRank>>;
        requires !BlockTensorConcept<CType<T, CRank>>;
    }
auto element_transform(CType<T, CRank> *C, UnaryOperator unary_opt) -> void {
    LabeledSection0();

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    // EINSUMS_OMP_PARALLEL_FOR
    for (auto it = view.begin(); it != view.end(); it++) {
        T &target_value = std::apply(*C, *it);
        target_value    = unary_opt(target_value);
    }
}

template <BlockTensorConcept CType, typename UnaryOperator>
auto element_transform(CType *C, UnaryOperator unary_opt) -> void {
    for (int i = 0; i < C->num_blocks(); i++) {
        element_transform(&(C->block(i)), unary_opt);
    }
}

template <BlockTensorConcept CType, BlockTensorConcept... MultiTensors, size_t Rank, typename MultiOperator, typename T>
    requires(IsIncoreRankBlockTensorV<MultiTensors, Rank, T> && ... && IsIncoreRankBlockTensorV<CType, Rank, T>)
auto element(MultiOperator multi_opt, CType *C, MultiTensors &...tensors) {
    if (((C->num_blocks() != tensors.num_blocks()) || ...)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "element: All tensors need to have the same number of blocks.");
    }
    for (int i = 0; i < C->num_blocks; i++) {
        if (((C->block_dim(i) != tensors.block_dim(i)) || ...)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "element: All tensor blocks need to have the same size.");
        }
    }

    for (int i = 0; i < C->num_blocks; i++) {
        element(multi_opt, &(C->block(i)), tensors.block(i)...);
    }
}

template <template <typename, size_t> typename CType, template <typename, size_t> typename... MultiTensors, size_t Rank,
          typename MultiOperator, typename T>
    requires(!(IsIncoreRankBlockTensorV<MultiTensors<T, Rank>, Rank, T> && ... && IsIncoreRankBlockTensorV<CType<T, Rank>, Rank, T>))
auto element(MultiOperator multi_opt, CType<T, Rank> *C, MultiTensors<T, Rank> &...tensors) {
    LabeledSection0();

    auto target_dims = get_dim_ranges<Rank>(*C);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    // Ensure the various tensors passed in are the same dimensionality
    if (((C->dims() != tensors.dims()) || ...)) {
        println_abort("element: at least one tensor does not have same dimensionality as destination");
    }

    // EINSUMS_OMP_PARALLEL_FOR
    for (auto it = view.begin(); it != view.end(); it++) {
        T target_value      = std::apply(*C, *it);
        std::apply(*C, *it) = multi_opt(target_value, std::apply(tensors, *it)...);
    }
}
} // namespace einsums::tensor_algebra