//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace einsums::tensor_algebra {

namespace detail {}

template <CoreTensorConcept CType, typename UnaryOperator>
    requires requires {
        requires BasicTensorConcept<CType>;
        requires RankTensorConcept<CType>;
    }
auto element_transform(CType *C, UnaryOperator unary_opt) -> void {
    LabeledSection0();
    using T               = typename CType::ValueType;
    constexpr size_t Rank = CType::Rank;

    Stride<Rank> index_strides;
    size_t       elements = dims_to_strides(C->dims(), index_strides);

    // EINSUMS_OMP_PARALLEL_FOR
    for (size_t item = 0; item < elements; item++) {
        size_t offset;
        sentinel_to_sentinels(item, index_strides, C->strides(), offset);

        T &target = C->data()[offset];

        target = unary_opt(target);
    }
}

template <BlockTensorConcept CType, typename UnaryOperator>
auto element_transform(CType *C, UnaryOperator unary_opt) -> void {
    for (int i = 0; i < C->num_blocks(); i++) {
        element_transform(&(C->block(i)), unary_opt);
    }
}

template <BlockTensorConcept CType, typename MultiOperator, BlockTensorConcept... MultiTensors>
    requires requires {
        requires(IsIncoreBlockTensorV<MultiTensors> && ... && IsIncoreBlockTensorV<CType>);
        requires(SameUnderlyingAndRank<CType, MultiTensors> && ...);
    }
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
    requires requires {
        requires(CoreRankTensor<MultiTensors<T, Rank>, Rank, T> && ... && CoreRankTensor<CType<T, Rank>, Rank, T>);
        requires(!CollectedTensorConcept<MultiTensors<T, Rank>> && ... && !CollectedTensorConcept<CType<T, Rank>>);
        requires BasicTensorConcept<CType<T, Rank>>;
    }
auto element(MultiOperator multi_opt, CType<T, Rank> *C, MultiTensors<T, Rank> &...tensors) {
    LabeledSection0();

    // Ensure the various tensors passed in are the same dimensionality
    if (((C->dims() != tensors.dims()) || ...)) {
        println_abort("element: at least one tensor does not have same dimensionality as destination");
    }

    Stride<Rank> index_strides;
    size_t       elements = dims_to_strides(C->dims(), index_strides);

    // EINSUMS_OMP_PARALLEL_FOR
    for (size_t item = 0; item < elements; item++) {
        thread_local std::array<int64_t, Rank> index;

        sentinel_to_indices(item, index_strides, index);

        T &target_value = subscript_tensor(*C, index);

        target_value = multi_opt(target_value, subscript_tensor(tensors, index)...);
    }
}
} // namespace einsums::tensor_algebra