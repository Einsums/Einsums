/*
 * Copyright (c) 2022 Justin Turney
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <tuple>
#include <utility>

#include "einsums/STL.hpp"
#include "einsums/_Common.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

namespace detail {

// Finds the order of the indices.
template <size_t Rank, typename... Args, std::size_t... I>
auto order_indices(const std::tuple<Args...> &combination,
                   const std::array<size_t, Rank> &order,
                   std::index_sequence<I...>) {
    return std::tuple{get_from_tuple<size_t>(combination, order[I])...};
}

}  // namespace detail

// Finds the order of the indices.
template <size_t Rank, typename... Args>
auto order_indices(const std::tuple<Args...> &combination,
                   const std::array<size_t, Rank> &order) {
    return detail::order_indices(combination, order,
                                 std::make_index_sequence<Rank>{});
}

namespace detail {

template <typename T, int Position>
constexpr auto _find_type_with_position() {
    return std::make_tuple();
}

// TODO: find what this does.
template <typename T, int Position, typename Head, typename... Args>
constexpr auto _find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position),
                        _find_type_with_position<T, Position + 1, Args...>());
    } else {
        return _find_type_with_position<T, Position + 1, Args...>();
    }
}

// TODO: find what this does.
template <typename T, int Position>
constexpr auto _unique_type_with_position() {
    return std::make_tuple();
}

// TODO: find what this does.
template <typename T, int Position, typename Head, typename... Args>
constexpr auto _unique_find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position));
    } else {
        return _unique_find_type_with_position<T, Position + 1, Args...>();
    }
}

// TODO: find what this does.
template <template <typename, size_t> typename TensorType,
          size_t Rank, typename... Args, std::size_t... I, typename T = double>
auto get_dim_ranges_for(const TensorType<T, Rank> &tensor,
                        const std::tuple<Args...> &args,
                        std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0,
                   static_cast<int>(tensor.dim(std::get<2 * I + 1>(args))))...};
}

// TODO: find what this does.
template <template <typename, size_t> typename TensorType,
          size_t Rank, typename... Args,
          std::size_t... I, typename T = double>
auto get_dim_for(const TensorType<T, Rank> &tensor,
                 const std::tuple<Args...> &args, std::index_sequence<I...>) {
    return std::tuple{tensor.dim(std::get<2 * I + 1>(args))...};
}

// Base case for the recursive function.
template <typename T, int Position>
constexpr auto find_position() {
    return -1;
}

// Finds the position of the type T in the argument list.
template <typename T, int Position, typename Head, typename... Args>
constexpr auto find_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        // Found it
        return Position;
    } else {
        return find_position<T, Position + 1, Args...>();
    }
}

// Finds the position of the type AIndex in the argument list.
template <typename AIndex, typename... Args>
constexpr auto find_position() {
    return find_position<AIndex, 0, Args...>();
}

// TODO: find what makes this different.
template <typename AIndex, typename... TargetCombination>
constexpr auto find_position(const std::tuple<TargetCombination...> &) {
    return detail::find_position<AIndex, TargetCombination...>();
}

// TODO: find what this does.
template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(detail::_find_type_with_position<
                          std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

// TODO: find what this does.
template <typename... Ts, typename... Us>
constexpr auto find_type_with_position(const std::tuple<Ts...> &,
                                       const std::tuple<Us...> &) {
    return _find_type_with_position<std::tuple<Ts...>, Us...>(
                                std::make_index_sequence<sizeof...(Ts)>{});
}

// TODO: find what this does.
template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _unique_find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(detail::_unique_find_type_with_position<
                             std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

// TODO: find what this does.
template <typename... Ts, typename... Us>
constexpr auto unique_find_type_with_position(const std::tuple<Ts...> &,
                                              const std::tuple<Us...> &) {
    return _unique_find_type_with_position<std::tuple<Ts...>, Us...>(
                        std::make_index_sequence<sizeof...(Ts)>{});
}

// TODO: find what this does.
template <template <typename, size_t> typename TensorType,
          size_t Rank, typename... Args, typename T = double>
auto get_dim_ranges_for(const TensorType<T, Rank> &tensor,
                        const std::tuple<Args...> &args) {
    return detail::get_dim_ranges_for(tensor, args,
                std::make_index_sequence<sizeof...(Args) / 2>{});
}

// TODO: find what this does.
template <template <typename, size_t> typename TensorType,
          size_t Rank, typename... Args, typename T = double>
auto get_dim_for(const TensorType<T, Rank> &tensor,
                 const std::tuple<Args...> &args) {
    return detail::get_dim_for(tensor, args,
                    std::make_index_sequence<sizeof...(Args) / 2>{});
}

// TODO: find what this does.
template <typename AIndex, typename... TargetCombination,
          typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
auto construct_index(const std::tuple<TargetCombination...> &target_combination,
                     const std::tuple<TargetPositionInC...> &,
                     const std::tuple<LinkCombination...>   &link_combination,
                     const std::tuple<LinkPositionInLink...> &) {
    constexpr auto IsAIndexInC    = detail::find_position<AIndex,
                                                        TargetPositionInC...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex,
                                                       LinkPositionInLink...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1,
            "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC / 2>(target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink / 2>(link_combination);
    } else {
        return -1;
    }
}

// TODO: find what this does.
template <typename... AIndices, typename... TargetCombination,
          typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto
construct_indices(const std::tuple<TargetCombination...> &target_combination,
                  const std::tuple<TargetPositionInC...> &target_position_in_C,
                  const std::tuple<LinkCombination...> &link_combination,
                  const std::tuple<LinkPositionInLink...> &link_position_in_link
                 ) {
    return std::make_tuple(construct_index<AIndices>(target_combination,
                target_position_in_C, link_combination,
                link_position_in_link)...);
}

// TODO: find what this does.
template <typename AIndex, typename... UniqueTargetIndices,
          typename... UniqueTargetCombination, typename... TargetPositionInC,
          typename... UniqueLinkIndices, typename... UniqueLinkCombination,
          typename... LinkPositionInLink>
auto construct_index_from_unique_target_combination(
        const std::tuple<UniqueTargetIndices...> & /*unique_target_indices*/,
        const std::tuple<UniqueTargetCombination...> &unique_target_combination,
        const std::tuple<TargetPositionInC...> &,
        const std::tuple<UniqueLinkIndices...> & /*unique_link_indices*/,
        const std::tuple<UniqueLinkCombination...> &unique_link_combination,
        const std::tuple<LinkPositionInLink...> &) {
    constexpr auto IsAIndexInC = detail::find_position<AIndex,
                                                UniqueTargetIndices...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex,
                                                UniqueLinkIndices...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1,
        "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC>(unique_target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink>(unique_link_combination);
    } else {
        return -1;
    }
}

// TODO: find what this does.
template <typename... AIndices, typename... UniqueTargetIndices,
          typename... UniqueTargetCombination, typename... TargetPositionInC,
          typename... UniqueLinkIndices, typename... UniqueLinkCombination,
          typename... LinkPositionInLink>
constexpr auto construct_indices_from_unique_combination(
        const std::tuple<UniqueTargetIndices...>     &unique_target_indices,
        const std::tuple<UniqueTargetCombination...> &unique_target_combination,
        const std::tuple<TargetPositionInC...>       &target_position_in_C,
        const std::tuple<UniqueLinkIndices...>       &unique_link_indices,
        const std::tuple<UniqueLinkCombination...>   &unique_link_combination,
        const std::tuple<LinkPositionInLink...>      &link_position_in_link) {
    return std::make_tuple(
            construct_index_from_unique_target_combination<AIndices>(
                unique_target_indices, unique_target_combination,
                target_position_in_C, unique_link_indices,
                unique_link_combination, link_position_in_link)...);
}

// TODO: find what this does.
template <typename... AIndices, typename... TargetCombination,
          typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto construct_indices(const std::tuple<AIndices...> &,
            const std::tuple<TargetCombination...> &target_combination,
            const std::tuple<TargetPositionInC...>  &target_position_in_C,
            const std::tuple<LinkCombination...>    &link_combination,
            const std::tuple<LinkPositionInLink...> &link_position_in_link) {
    return construct_indices<AIndices...>(target_combination,
                                          target_position_in_C,
                                          link_combination,
                                          link_position_in_link);
}

// TODO: find what this does.
template <typename... PositionsInX, std::size_t... I>
constexpr auto _contiguous_positions(const std::tuple<PositionsInX...> &x,
                                     std::index_sequence<I...>) -> bool {
    return ((std::get<2 * I + 1>(x) == std::get<2 * I + 3>(x) - 1) && ...
                                                                   && true);
}

// TODO: find what this does.
template <typename... PositionsInX>
constexpr auto contiguous_positions(const std::tuple<PositionsInX...> &x)
-> bool {
    if constexpr (sizeof...(PositionsInX) <= 2) {
        return true;
    } else {
        return _contiguous_positions(x, std::make_index_sequence<
                                        sizeof...(PositionsInX) / 2 - 1>{});
    }
}

// Checks to see if the two lists have the same ordering.
template <typename... PositionsInX, typename... PositionsInY, std::size_t... I>
constexpr auto _is_same_ordering(
                const std::tuple<PositionsInX...> &positions_in_x,
                const std::tuple<PositionsInY...> &positions_in_y,
                std::index_sequence<I...>) {
    return (std::is_same_v<decltype(std::get<2 * I>(positions_in_x)),
                            decltype(std::get<2 * I>(positions_in_y))> && ...);
}

// Checks to see if the two lists have the same ordering.
template <typename... PositionsInX, typename... PositionsInY>
constexpr auto is_same_ordering(
                        const std::tuple<PositionsInX...> &positions_in_x,
                        const std::tuple<PositionsInY...> &positions_in_y) {
    if constexpr (sizeof...(PositionsInX) == 0 || sizeof...(PositionsInY) == 0)
        return false; // NOLINT
    else if constexpr (sizeof...(PositionsInX) != sizeof...(PositionsInY))
        return false;
    else
        return _is_same_ordering(positions_in_x, positions_in_y,
                    std::make_index_sequence<sizeof...(PositionsInX) / 2>{});
}

// TODO: find what this does.
template <template <typename, size_t> typename XType,
          size_t XRank, typename... PositionsInX,
          std::size_t... I, typename T = double>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices,
                            const XType<T, XRank> &X,
                            std::index_sequence<I...>) -> size_t {
    return (X.dim(std::get<2 * I + 1>(indices)) * ... * 1);
}

// TODO: find what this does.
template <template <typename, size_t> typename XType, size_t XRank,
          typename... PositionsInX, std::size_t... I, typename T = double>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices,
                            const XType<T, XRank> &X,
                             std::index_sequence<I...>) -> bool {
    return ((X.dim(std::get<1>(indices)) ==
             X.dim(std::get<2 * I + 1>(indices))) && ... && 1);
}

// Checks to see if the two arguments have the same indices.
template <typename LHS, typename RHS, std::size_t... I>
constexpr auto same_indices(std::index_sequence<I...>) {
    return (std::is_same_v<std::tuple_element_t<I, LHS>,
                           std::tuple_element_t<I, RHS>> && ...);
}

// TODO: find what this does.
template <template <typename, size_t> typename XType, size_t XRank,
          typename... PositionsInX, typename T = double>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices,
                            const XType<T, XRank> &X) -> size_t {
    return detail::product_dims(indices, X,
                    std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

// TODO: find what this does.
template <template <typename, size_t> typename XType, size_t XRank,
          typename... PositionsInX, typename T = double>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices,
                            const XType<T, XRank> &X) -> size_t {
    return detail::is_same_dims(indices, X,
                     std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

// Returns the last stride of X.
template <template <typename, size_t> typename XType, size_t XRank,
          typename... PositionsInX, typename T = double>
constexpr auto last_stride(const std::tuple<PositionsInX...> &indices,
                           const XType<T, XRank> &X) -> size_t {
    return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
}

// Checks to see if the two arguments have the same indices.
template <typename LHS, typename RHS>
constexpr auto same_indices() {
    if constexpr (std::tuple_size_v<LHS> != std::tuple_size_v<RHS>)
        return false;
    else
        return detail::same_indices<LHS, RHS>(
                    std::make_index_sequence<std::tuple_size_v<LHS>>());
}

}  // namespace detail

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)
