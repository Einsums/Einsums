//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

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
    LabeledSection0();

    TType diff = 0.0;

    size_t nelem = 1;
    for_sequence<TRank>([&](auto i) { nelem *= tensor1.dim(i); });

    auto target_dims = get_dim_ranges<TRank>(tensor1);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for reduction(+ : diff)
    for (auto it = view.begin(); it < view.end(); it++) {
        auto  target_combination = *it;
        TType target1            = std::apply(tensor1, target_combination);
        TType target2            = std::apply(tensor2, target_combination);
        diff += (target1 - target2) * (target1 - target2);
    }

    return std::sqrt(diff / nelem);
}
} // namespace einsums