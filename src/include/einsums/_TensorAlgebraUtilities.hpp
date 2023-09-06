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

/**
 * @file _TensorAlgebraUtilities.hpp
 *
 * Contains utilities for tensor algebra.
 */

#pragma once

#include "einsums/STL.hpp"
#include "einsums/_Common.hpp"

#include <tuple>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

namespace detail {

/**
 * Reorders a set of objects with the given new ordering. Should not be called
 * on its own. Rather, it is the worker for another definition of order_indices.
 *
 * @param combination The current set of objects.
 * @param order The new order for the objects.
 * @param A list of indices.
 * 
 * @return A tuple with the same objects as was passed in, but in a different
 * order.
 */
template <size_t Rank, typename... Args, std::size_t... I>
auto order_indices(const std::tuple<Args...> &combination, const std::array<size_t, Rank> &order, std::index_sequence<I...>) {
    return std::tuple{get_from_tuple<size_t>(combination, order[I])...};
}

} // namespace detail

/**
 * Reorders a set of objects with the given new ordering.
 *
 * @param combination A tuple containing objects to reorder.
 * @param order A list of new indices for the new ordering.
 *
 * @return The tuple of objects in a new order.
 */
template <size_t Rank, typename... Args>
auto order_indices(const std::tuple<Args...> &combination, const std::array<size_t, Rank> &order) {
    return detail::order_indices(combination, order, std::make_index_sequence<Rank>{});
}

namespace detail {

/**
 * Base case for einsums::tensor_algebra::_find_type_with_position.
 *
 * @return An empty tuple.
 */
template <typename T, int Position>
constexpr auto _find_type_with_position() {
    return std::make_tuple();
}

/**
 * Finds all positions where the type in the argument matches T.
 *
 * @todo Find what this may do.
 * 
 * @return A tuple of pairs which contain the matching type and the position.
 */
template <typename T, int Position, typename Head, typename... Args>
constexpr auto _find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position), _find_type_with_position<T, Position + 1, Args...>());
    } else {
        return _find_type_with_position<T, Position + 1, Args...>();
    }
}

/**
 * Base case for the more compicated version.
 *
 * @return An empty tuple.
 */
template <typename T, int Position>
constexpr auto _unique_type_with_position() {
    return std::make_tuple();
}

/**
 * Finds the first position where the value in the list matches T.
 *
 * @todo Find what this may do.
 *
 * @return A tuple containing a pair containing the type and the position.
 */
template <typename T, int Position, typename Head, typename... Args>
constexpr auto _unique_find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position));
    } else {
        return _unique_find_type_with_position<T, Position + 1, Args...>();
    }
}

/**
 * @todo I have no idea what this does.
 *
 * @return A tuple of some sort. @todo WTF does this return?
 */
template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, std::size_t... I, typename T = double>
auto get_dim_ranges_for(const TensorType<T, Rank> &tensor, const std::tuple<Args...> &args, std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(std::get<2 * I + 1>(args)))...};
}

/**
 * Does something with dimensions.
 * @todo I have no clue what this does.
 *
 * @return A tuple of some sort. @todo C++ is too opaque for this.
 */
template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, std::size_t... I, typename T = double>
auto get_dim_for(const TensorType<T, Rank> &tensor, const std::tuple<Args...> &args, std::index_sequence<I...>) {
    return std::tuple{tensor.dim(std::get<2 * I + 1>(args))...};
}

/**
 * Base case for the find_position function indicating that the item was
 * not found.
 *
 * @return -1 because it could not find what it was looking for.
 */
template <typename T, int Position>
constexpr auto find_position() {
    return -1;
}

/**
 * Finds the position of type T in the argument list. Do not call. Should
 * be called using the simpler find_position.
 *
 * @return The position of the argument, or -1 if not found.
 */
template <typename T, int Position, typename Head, typename... Args>
constexpr auto find_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        // Found it
        return Position;
    } else {
        return find_position<T, Position + 1, Args...>();
    }
}

/**
 * Finds the position of the type AIndex within the argument list.
 *
 * @return The position of the argument, or -1 if not found.
 */
template <typename AIndex, typename... Args>
constexpr auto find_position() {
    return find_position<AIndex, 0, Args...>();
}

/**
 * Find the position of the element with type AIndex within the tuple.
 *
 * @param The tuple to search.
 *
 * @return The index of the item, or -1 if not found.
 */
template <typename AIndex, typename... TargetCombination>
constexpr auto find_position(const std::tuple<TargetCombination...> &) {
    return detail::find_position<AIndex, TargetCombination...>();
}

/**
 * @todo This makes no sense.
 *
 * @return Some sort of tuple. @todo Not much to go on here.
 */
template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(detail::_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

/**
 * @todo This does something. I think only God knows now.
 *
 * @return Some sort of tuple. @todo What does this mean?
 */
template <typename... Ts, typename... Us>
constexpr auto find_type_with_position(const std::tuple<Ts...> &, const std::tuple<Us...> &) {
    return _find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

/**
 * @todo No idea what this is supposed to do.
 * 
 * @return A tuple. @todo Man, there are a whole bunch of these tuples.
 */
template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _unique_find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(detail::_unique_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

/**
 * @todo Still no idea.
 *
 * @return Probably some sort of tuple. @todo Another tuple? Really?
 */
template <typename... Ts, typename... Us>
constexpr auto unique_find_type_with_position(const std::tuple<Ts...> &, const std::tuple<Us...> &) {
    return _unique_find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

/**
 * This does something related to other get_dim_ranges_for.
 * @todo This might do something, or it might not. No clue.
 *
 * @return Something.
 */
template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, typename T = double>
auto get_dim_ranges_for(const TensorType<T, Rank> &tensor, const std::tuple<Args...> &args) {
    return detail::get_dim_ranges_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}
  
/**
 * This does something related to other get_dim_for functions.
 * @todo Get a description of this function.
 *
 * @return Something.
 */
template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, typename T = double>
auto get_dim_for(const TensorType<T, Rank> &tensor, const std::tuple<Args...> &args) {
    return detail::get_dim_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

/**
 * @todo This looks important. It would help if I knew what it did.
 *
 * @return -1 on error. No clue what it does normally.
 */
template <typename AIndex, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
auto construct_index(const std::tuple<TargetCombination...> &target_combination, const std::tuple<TargetPositionInC...> &,
                     const std::tuple<LinkCombination...>   &link_combination, const std::tuple<LinkPositionInLink...> &) {

    constexpr auto IsAIndexInC    = detail::find_position<AIndex, TargetPositionInC...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex, LinkPositionInLink...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1, "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC / 2>(target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink / 2>(link_combination);
    } else {
        return -1;
    }
}

/**
 * @todo This looks important. It would be a shame if its purpose was lost to
 * the blowing winds of time.
 *
 * @return Some sort of tuple. I've been writing this a lot.
 */
template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto
construct_indices(const std::tuple<TargetCombination...> &target_combination, const std::tuple<TargetPositionInC...> &target_position_in_C,
                  const std::tuple<LinkCombination...> &link_combination, const std::tuple<LinkPositionInLink...> &link_position_in_link) {
    return std::make_tuple(construct_index<AIndices>(target_combination, target_position_in_C, link_combination, link_position_in_link)...);
}

/**
 * @todo This seems like a descriptive name of some sort. Unfortunately,
 * I don't speak self-documenting code.
 *
 * @return -1 on error. No idea otherwise.
 */
template <typename AIndex, typename... UniqueTargetIndices, typename... UniqueTargetCombination, typename... TargetPositionInC,
          typename... UniqueLinkIndices, typename... UniqueLinkCombination, typename... LinkPositionInLink>
auto construct_index_from_unique_target_combination(const std::tuple<UniqueTargetIndices...> & /*unique_target_indices*/,
                                                    const std::tuple<UniqueTargetCombination...> &unique_target_combination,
                                                    const std::tuple<TargetPositionInC...> &,
                                                    const std::tuple<UniqueLinkIndices...> & /*unique_link_indices*/,
                                                    const std::tuple<UniqueLinkCombination...> &unique_link_combination,
                                                    const std::tuple<LinkPositionInLink...> &) {
    constexpr auto IsAIndexInC    = detail::find_position<AIndex, UniqueTargetIndices...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex, UniqueLinkIndices...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1, "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC>(unique_target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink>(unique_link_combination);
    } else {
        return -1;
    }
}

/**
 * @todo Looks similar to the last one.
 * 
 * @return Some sort of tuple.
 */
template <typename... AIndices, typename... UniqueTargetIndices, typename... UniqueTargetCombination, typename... TargetPositionInC,
          typename... UniqueLinkIndices, typename... UniqueLinkCombination, typename... LinkPositionInLink>
constexpr auto construct_indices_from_unique_combination(const std::tuple<UniqueTargetIndices...>     &unique_target_indices,
                                                         const std::tuple<UniqueTargetCombination...> &unique_target_combination,
                                                         const std::tuple<TargetPositionInC...>       &target_position_in_C,
                                                         const std::tuple<UniqueLinkIndices...>       &unique_link_indices,
                                                         const std::tuple<UniqueLinkCombination...>   &unique_link_combination,
                                                         const std::tuple<LinkPositionInLink...>      &link_position_in_link) {
    return std::make_tuple(construct_index_from_unique_target_combination<AIndices>(unique_target_indices, unique_target_combination,
                                                                                    target_position_in_C, unique_link_indices,
                                                                                    unique_link_combination, link_position_in_link)...);
}

/**
 * @todo This constructs indices. But what does it actually do?
 *
 * @return Probably a tuple.
 */
template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto construct_indices(const std::tuple<AIndices...> &, const std::tuple<TargetCombination...> &target_combination,
                                 const std::tuple<TargetPositionInC...>  &target_position_in_C,
                                 const std::tuple<LinkCombination...>    &link_combination,
                                 const std::tuple<LinkPositionInLink...> &link_position_in_link) {
    return construct_indices<AIndices...>(target_combination, target_position_in_C, link_combination, link_position_in_link);
}

/**
 * @todo This means nothing to me. What is it for?
 *
 * @return Some sort of boolean value.
 */
template <typename... PositionsInX, std::size_t... I>
constexpr auto _contiguous_positions(const std::tuple<PositionsInX...> &x, std::index_sequence<I...>) -> bool {
    return ((std::get<2 * I + 1>(x) == std::get<2 * I + 3>(x) - 1) && ... && true);
}

/**
 * @todo What does it do???
 *
 * @return A boolean.
 */
template <typename... PositionsInX>
constexpr auto contiguous_positions(const std::tuple<PositionsInX...> &x) -> bool {
    if constexpr (sizeof...(PositionsInX) <= 2) {
        return true;
    } else {
        return _contiguous_positions(x, std::make_index_sequence<sizeof...(PositionsInX) / 2 - 1>{});
    }
}

/**
 * Worker that determines if two tuples have the same ordering.
 * @todo Write a better argument description.
 * 
 * @param positions_in_x the values in the x positions.
 * @param positions_in_y The values in the y positions.
 *
 * @return Whether or not the ordering is the same.
 */
template <typename... PositionsInX, typename... PositionsInY, std::size_t... I>
constexpr auto _is_same_ordering(const std::tuple<PositionsInX...> &positions_in_x, const std::tuple<PositionsInY...> &positions_in_y,
                                 std::index_sequence<I...>) {
    return (std::is_same_v<decltype(std::get<2 * I>(positions_in_x)), decltype(std::get<2 * I>(positions_in_y))> && ...);
}

/**
 * Checks to see if two lists have the same ordering.
 * @todo Find out what this means.
 *
 * @param positions_in_x Positions in the x array.
 * @param positions_in_y Positions in the y array.
 *
 * @return A boolean.
 */
template <typename... PositionsInX, typename... PositionsInY>
constexpr auto is_same_ordering(const std::tuple<PositionsInX...> &positions_in_x, const std::tuple<PositionsInY...> &positions_in_y) {
    // static_assert(sizeof...(PositionsInX) == sizeof...(PositionsInY) && sizeof...(PositionsInX) > 0);
    if constexpr (sizeof...(PositionsInX) == 0 || sizeof...(PositionsInY) == 0)
        return false; // NOLINT
    else if constexpr (sizeof...(PositionsInX) != sizeof...(PositionsInY))
        return false;
    else
        return _is_same_ordering(positions_in_x, positions_in_y, std::make_index_sequence<sizeof...(PositionsInX) / 2>{});
}

/**
 * Finds the product of the dimensions. Worker function. Do not call.
 *
 * @param indices The list of indices.
 * @param X The tensor? @todo What is this?
 * 
 * @return A product? Idk.
 */
template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, std::size_t... I, typename T = double>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType<T, XRank> &X, std::index_sequence<I...>) -> size_t {
    return (X.dim(std::get<2 * I + 1>(indices)) * ... * 1);
}

/**
 * Checks to see if the indices match the dimensions of the tensor.
 * @todo Verify this description.
 *
 * @param indices The indices.
 * @param X The tensor to compare to.
 *
 * @return Whether the dimensions are the same.
 */
template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, std::size_t... I, typename T = double>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType<T, XRank> &X, std::index_sequence<I...>) -> bool {
    return ((X.dim(std::get<1>(indices)) == X.dim(std::get<2 * I + 1>(indices))) && ... && 1);
}

/**
 * Checks to see if the arguments have the same indices. Worker function?
 * @todo Figure out the usage of this function.
 *
 * @param An index sequence.
 *
 * @return Boolean?
 */
template <typename LHS, typename RHS, std::size_t... I>
constexpr auto same_indices(std::index_sequence<I...>) {
    return (std::is_same_v<std::tuple_element_t<I, LHS>, std::tuple_element_t<I, RHS>> && ...);
}

/**
 * @todo This does something. That's a fact.
 *
 * @param indices Indices of some sort.
 * @param X A tensor.
 */
template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType<T, XRank> &X) -> size_t {
    return detail::product_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

/**
 * Checks whether the dimensions are the same as in the tensor.
 * @todo Double check this behavior.
 *
 * @param indices Indices of some sort.
 * @param X A tensor.
 *
 * @return Whether the dimensions are the same.
 */
template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType<T, XRank> &X) -> size_t {
    return detail::is_same_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

/**
 * Returns the last stride of X. @todo What could it all mean?
 *
 * @param indices Indices of some sort.
 * @param X A tensor.
 *
 * @return The last stride of X?
 */
template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto last_stride(const std::tuple<PositionsInX...> &indices, const XType<T, XRank> &X) -> size_t {
    return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
}

/**
 * Checks to see if two indices are the same?
 * @todo This has meaning to someone.
 *
 * @return A boolean.
 */
template <typename LHS, typename RHS>
constexpr auto same_indices() {
    if constexpr (std::tuple_size_v<LHS> != std::tuple_size_v<RHS>)
        return false;
    else
        return detail::same_indices<LHS, RHS>(std::make_index_sequence<std::tuple_size_v<LHS>>());
}

} // namespace detail

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)
