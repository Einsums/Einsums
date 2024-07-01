#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Index.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

namespace einsums::tensor_algebra {
namespace detail {
template <int Remaining, typename Skip, typename Head, typename... Args>
constexpr auto _get_n_skip() {
    // They are the same, skip it.
    if constexpr (std::is_same_v<std::decay_t<Skip>, std::decay_t<Head>> && Remaining > 0) {
        return _get_n_skip<Remaining, Skip, Args...>();
    } else if constexpr (Remaining > 0) {
        // They are not the same, add it.
        return std::tuple_cat(std::make_tuple(Head()), _get_n_skip<Remaining - 1, Skip, Args...>());
    } else {
        return std::tuple{};
    }
}

template <unsigned int N, typename Skip, typename... List>
constexpr auto get_n_skip(const Skip &, const std::tuple<List...> &) {
    return _get_n_skip<N, Skip, List...>();
}

template <int Remaining, typename Head, typename... Args>
constexpr auto _get_n() {
    if constexpr (Remaining > 0) {
        // They are not the same, add it.
        return std::tuple_cat(std::make_tuple(Head()), _get_n<Remaining - 1, Args...>());
    } else {
        return std::tuple{};
    }
}
} // namespace detail

template <unsigned int N, typename... List>
constexpr auto get_n(const std::tuple<List...> &) {
    return detail::_get_n<N, List...>();
}

/**
 * Returns the mode-`mode` unfolding of `tensor` with modes startng at `0`
 *
 * @returns unfolded_tensor of shape ``(tensor.dim(mode), -1)``
 */
template <unsigned int mode, template <typename, size_t> typename CType, size_t CRank, typename T>
auto unfold(const CType<T, CRank> &source) -> Tensor<T, 2>
    requires(std::is_same_v<Tensor<T, CRank>, CType<T, CRank>>)
{
    LabeledSection1(fmt::format("mode-{} unfold", mode));

    Dim<2> target_dims;
    target_dims[0] = source.dim(mode);
    target_dims[1] = 1;
    for (int i = 0; i < CRank; i++) {
        if (i == mode)
            continue;
        target_dims[1] *= source.dim(i);
    }

    auto target         = Tensor{fmt::format("mode-{} unfolding of {}", mode, source.name()), target_dims[0], target_dims[1]};
    auto target_indices = std::make_tuple(std::get<mode>(index::list), index::Z);
    auto source_indices = get_n<CRank>(index::list);

    // Use similar logic found in einsums:
    auto link        = intersect_t<decltype(target_indices), decltype(source_indices)>();
    auto target_only = difference_t<decltype(target_indices), decltype(link)>();
    auto source_only = difference_t<decltype(source_indices), decltype(link)>();

    auto source_position_in_source = detail::find_type_with_position(source_only, source_indices);
    auto link_position_in_source   = detail::find_type_with_position(link, source_indices);

    auto link_dims   = detail::get_dim_ranges_for(target, detail::find_type_with_position(link, target_indices));
    auto source_dims = detail::get_dim_ranges_for(source, source_position_in_source);

    auto link_view   = std::apply(ranges::views::cartesian_product, link_dims);
    auto source_view = std::apply(ranges::views::cartesian_product, source_dims);

#pragma omp parallel for
    for (auto link_it = link_view.begin(); link_it < link_view.end(); link_it++) {
        size_t Z{0};
        for (auto source_it = source_view.begin(); source_it < source_view.end(); source_it++) {

            auto target_order = std::make_tuple(std::get<0>(*link_it), Z);

            auto source_order =
                detail::construct_indices(source_indices, *source_it, source_position_in_source, *link_it, link_position_in_source);

            T &target_value = std::apply(target, target_order);
            T  source_value = std::apply(source, source_order);

            target_value = source_value;

            Z++;
        }
    }

    return target;
}

} // namespace einsums::tensor_algebra