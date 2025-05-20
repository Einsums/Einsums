//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

namespace einsums {

/**
 * Computes the RMSD between two tensors of arbitrary dimension
 */
template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType>;
        requires InSamePlace<AType, BType>;
    }
auto rmsd(AType const &tensor1, BType const &tensor2) -> ValueTypeT<AType> {
    using TType            = ValueTypeT<AType>;
    constexpr size_t TRank = TensorRank<AType>;
    EINSUMS_PROFILE_SCOPE("TensorUtilities");

    TType diff = 0.0;

    std::array<size_t, TRank> index_strides;
    size_t                    elements = dims_to_strides(tensor1.dims(), index_strides);

    // #pragma omp parallel for reduction(+ : diff)
    for (size_t item = 0; item < elements; item++) {
        thread_local std::array<size_t, TRank> target_combination;
        sentinel_to_indices(item, index_strides, target_combination);

        TType target1 = subscript_tensor(tensor1, target_combination), target2 = subscript_tensor(tensor2, target_combination);

        diff += (target1 - target2) * (target1 - target2);
    }

    return std::sqrt(diff / elements);
}
} // namespace einsums