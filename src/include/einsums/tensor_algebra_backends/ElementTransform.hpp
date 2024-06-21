#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_Index.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Timer.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsums::tensor_algebra {
template <template <typename, size_t> typename CType, size_t CRank, typename UnaryOperator, typename T>
    requires std::derived_from<CType<T, CRank>, ::einsums::detail::TensorBase<T, CRank>>
auto element_transform(CType<T, CRank> *C, UnaryOperator unary_opt) -> void {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<CType<T, CRank>, CRank, T>) {
        for (int i = 0; i < C->num_blocks(); i++) {
            element_transform(&(C->block(i)), unary_opt);
        }
        return;
    }

    LabeledSection0();

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    EINSUMS_OMP_PARALLEL_FOR
    for (auto it = view.begin(); it != view.end(); it++) {
        T &target_value = std::apply(*C, *it);
        target_value    = unary_opt(target_value);
    }
}

template <template <typename, size_t> typename CType, template <typename, size_t> typename... MultiTensors, size_t Rank,
          typename MultiOperator, typename T>
auto element(MultiOperator multi_opt, CType<T, Rank> *C, MultiTensors<T, Rank> &...tensors) {
    if constexpr ((einsums::detail::IsIncoreRankBlockTensorV<MultiTensors<T, Rank>, Rank, T> && ... &&
                   einsums::detail::IsIncoreRankBlockTensorV<CType<T, Rank>, Rank, T>)) {

        if (((C->num_blocks() != tensors.num_blocks()) || ...)) {
            throw std::runtime_error("element: All tensors need to have the same number of blocks.");
        }
        for (int i = 0; i < C->num_blocks; i++) {
            if (((C->block_dim(i) != tensors.block_dim(i)) || ...)) {
                throw std::runtime_error("element: All tensor blocks need to have the same size.");
            }
        }

        for (int i = 0; i < C->num_blocks; i++) {
            element(multi_opt, &(C->block(i)), tensors.block(i)...);
        }
        return;
    }

    LabeledSection0();

    auto target_dims = get_dim_ranges<Rank>(*C);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    // Ensure the various tensors passed in are the same dimensionality
    if (((C->dims() != tensors.dims()) || ...)) {
        println_abort("element: at least one tensor does not have same dimensionality as destination");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (auto it = view.begin(); it != view.end(); it++) {
        T target_value      = std::apply(*C, *it);
        std::apply(*C, *it) = multi_opt(target_value, std::apply(tensors, *it)...);
    }
}
} // namespace einsums::tensor_algebra