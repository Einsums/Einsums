//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include "einsums/STL.hpp"
#include "range/v3/view/iota.hpp"

#include <tuple>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {

template <size_t Rank, typename... Args, std::size_t... I>
auto order_indices(const std::tuple<Args...> &combination, const std::array<size_t, Rank> &order, std::index_sequence<I...> /*seq*/) {
    return std::tuple{get_from_tuple<size_t>(combination, order[I])...};
}

} // namespace detail
#endif

template <size_t Rank, typename... Args>
auto order_indices(const std::tuple<Args...> &combination, const std::array<size_t, Rank> &order) {
    return detail::order_indices(combination, order, std::make_index_sequence<Rank>{});
}

namespace detail {

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <typename T, int Position>
constexpr auto _find_type_with_position() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto _find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position), _find_type_with_position<T, Position + 1, Args...>());
    } else {
        return _find_type_with_position<T, Position + 1, Args...>();
    }
}

template <typename T, int Position>
constexpr auto _unique_type_with_position() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto _unique_find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position));
    } else {
        return _unique_find_type_with_position<T, Position + 1, Args...>();
    }
}
#endif

template<TensorConcept TensorType, typename... Args, size_t... I>
auto get_dim_ranges_for(const TensorType &tensor, const std::tuple<Args...> &args, std::index_sequence<I...> /*seq*/) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(std::get<2 * I + 1>(args)))...};
}

template<TensorConcept TensorType, typename... Args, size_t... I>
auto get_dim_for(const TensorType &tensor, const std::tuple<Args...> &args, std::index_sequence<I...> /*seq*/) {
    return std::tuple{tensor.dim(std::get<2 * I + 1>(args))...};
}

template <typename T, int Position>
constexpr auto find_position() {
    return -1;
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto find_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        // Found it
        return Position;
    } else {
        return find_position<T, Position + 1, Args...>();
    }
}

template <typename AIndex, typename... Args>
constexpr auto find_position() {
    return find_position<AIndex, 0, Args...>();
}

template <typename AIndex, typename... TargetCombination>
constexpr auto find_position(const std::tuple<TargetCombination...> & /*indices*/) {
    return detail::find_position<AIndex, TargetCombination...>();
}

template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _find_type_with_position(std::index_sequence<Is...> /*seq*/) {
    return std::tuple_cat(detail::_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

template <typename... Ts, typename... Us>
constexpr auto find_type_with_position(const std::tuple<Ts...> & /*unused*/, const std::tuple<Us...> & /*unused*/) {
    return _find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _unique_find_type_with_position(std::index_sequence<Is...> /*seq*/) {
    return std::tuple_cat(detail::_unique_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

template <typename... Ts, typename... Us>
constexpr auto unique_find_type_with_position(const std::tuple<Ts...> & /*unused*/, const std::tuple<Us...> & /*unused*/) {
    return _unique_find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

template<TensorConcept TensorType, typename... Args>
auto get_dim_ranges_for(const TensorType &tensor, const std::tuple<Args...> &args) {
    return detail::get_dim_ranges_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

template<TensorConcept TensorType, typename... Args>
auto get_dim_for(const TensorType &tensor, const std::tuple<Args...> &args) {
    return detail::get_dim_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

template<typename ScalarType>
requires(!TensorConcept<ScalarType>)
auto get_dim_ranges_for(const ScalarType &tensor, const std::tuple<> &args) {
    return std::tuple{};
}

template<typename ScalarType>
auto get_dim_for(const ScalarType &tensor, const std::tuple<> &args) {
    return std::tuple{};
}

template <typename AIndex, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
auto construct_index(const std::tuple<TargetCombination...> &target_combination, const std::tuple<TargetPositionInC...> & /*unused*/,
                     const std::tuple<LinkCombination...>   &link_combination, const std::tuple<LinkPositionInLink...>   &/*unused*/) {

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

template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto
construct_indices(const std::tuple<TargetCombination...> &target_combination, const std::tuple<TargetPositionInC...> &target_position_in_C,
                  const std::tuple<LinkCombination...> &link_combination, const std::tuple<LinkPositionInLink...> &link_position_in_link) {
    return std::make_tuple(construct_index<AIndices>(target_combination, target_position_in_C, link_combination, link_position_in_link)...);
}

template <typename AIndex, typename... UniqueTargetIndices, typename... UniqueTargetCombination, typename... TargetPositionInC,
          typename... UniqueLinkIndices, typename... UniqueLinkCombination, typename... LinkPositionInLink>
auto construct_index_from_unique_target_combination(const std::tuple<UniqueTargetIndices...> & /*unique_target_indices*/,
                                                    const std::tuple<UniqueTargetCombination...> &unique_target_combination,
                                                    const std::tuple<TargetPositionInC...> & /*unused*/,
                                                    const std::tuple<UniqueLinkIndices...> & /*unique_link_indices*/,
                                                    const std::tuple<UniqueLinkCombination...> &unique_link_combination,
                                                    const std::tuple<LinkPositionInLink...> & /*unused*/) {

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

template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto construct_indices(const std::tuple<AIndices...> & /*unused*/, const std::tuple<TargetCombination...> &target_combination,
                                 const std::tuple<TargetPositionInC...>  &target_position_in_C,
                                 const std::tuple<LinkCombination...>    &link_combination,
                                 const std::tuple<LinkPositionInLink...> &link_position_in_link) {
    return construct_indices<AIndices...>(target_combination, target_position_in_C, link_combination, link_position_in_link);
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <typename... PositionsInX, std::size_t... I>
constexpr auto _contiguous_positions(const std::tuple<PositionsInX...> &x, std::index_sequence<I...> /*unused*/) -> bool {
    return ((std::get<2 * I + 1>(x) == std::get<2 * I + 3>(x) - 1) && ... && true);
}
#endif

/**
 * @brief Determines in the indices are contiguous.
 *
 * The tuple that is passed in resembles the following:
 *
 * @verbatim
 * {i, 0, j, 1, k, 2}
 * @endverbatim
 *
 * or
 *
 * @verbatim
 * {i, 0, j, 2, k 1}
 * @endverbatim
 *
 * In the first case, the function will return true because the indices of the labels are contiguous.
 * And in the second case, the function will return false because the indices of the labels are not contiguous.
 *
 * @tparam PositionsInX
 * @tparam I
 * @param x
 * @return true
 * @return false
 */
template <typename... PositionsInX>
constexpr auto contiguous_positions(const std::tuple<PositionsInX...> &x) -> bool {
    if constexpr (sizeof...(PositionsInX) <= 2) {
        return true;
    } else {
        return _contiguous_positions(x, std::make_index_sequence<sizeof...(PositionsInX) / 2 - 1>{});
    }
}

template <typename... PositionsInX, typename... PositionsInY, std::size_t... I>
constexpr auto _is_same_ordering(const std::tuple<PositionsInX...> &positions_in_x, const std::tuple<PositionsInY...> &positions_in_y,
                                 std::index_sequence<I...> /*unused*/) {
    return (std::is_same_v<decltype(std::get<2 * I>(positions_in_x)), decltype(std::get<2 * I>(positions_in_y))> && ...);
}

template <typename... PositionsInX, typename... PositionsInY>
constexpr auto is_same_ordering(const std::tuple<PositionsInX...> &positions_in_x, const std::tuple<PositionsInY...> &positions_in_y) {
    // static_assert(sizeof...(PositionsInX) == sizeof...(PositionsInY) && sizeof...(PositionsInX) > 0);
    if constexpr (sizeof...(PositionsInX) == 0 || sizeof...(PositionsInY) == 0) {
        return false; // NOLINT
    } else if constexpr (sizeof...(PositionsInX) != sizeof...(PositionsInY)) {
        return false;
    } else {
        return _is_same_ordering(positions_in_x, positions_in_y, std::make_index_sequence<sizeof...(PositionsInX) / 2>{});
    }
}

template<TensorConcept XType, typename... PositionsInX, size_t... I>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType &X,
                            std::index_sequence<I...> /*unused*/) -> size_t {
    return (X.dim(std::get<2 * I + 1>(indices)) * ... * 1);
}

template<TensorConcept XType, typename... PositionsInX, size_t... I>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType &X,
                            std::index_sequence<I...> /*unused*/) -> bool {
    return ((X.dim(std::get<1>(indices)) == X.dim(std::get<2 * I + 1>(indices))) && ... && 1);
}

template <typename LHS, typename RHS, std::size_t... I>
constexpr auto same_indices(std::index_sequence<I...> /*unused*/) {
    return (std::is_same_v<std::tuple_element_t<I, LHS>, std::tuple_element_t<I, RHS>> && ...);
}

template<TensorConcept XType, typename... PositionsInX>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType &X) -> size_t {
    return detail::product_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template<typename XType, typename... PositionsInX>
requires(!TensorConcept<XType>)
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType &X) -> size_t {
    return 0UL;
}

template<TensorConcept XType, typename... PositionsInX>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType &X) -> size_t {
    return detail::is_same_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template<typename XType, typename... PositionsInX>
requires(!TensorConcept<XType>)
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType &X) -> size_t {
    return true;
}

template<TensorConcept XType, typename... PositionsInX>
constexpr auto last_stride(const std::tuple<PositionsInX...> &indices, const XType &X) -> size_t {
    return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
}

template<typename XType, typename... PositionsInX>
requires(!TensorConcept<XType>)
constexpr auto last_stride(const std::tuple<PositionsInX...> &indices, const XType &X) -> size_t {
    return 0UL;
}

template <typename LHS, typename RHS>
constexpr auto same_indices() {
    if constexpr (std::tuple_size_v<LHS> != std::tuple_size_v<RHS>) {
        return false;
    } else {
        return detail::same_indices<LHS, RHS>(std::make_index_sequence<std::tuple_size_v<LHS>>());
    }
}

template <typename UniqueIndex, int BDim, TensorConcept BType>
inline size_t get_grid_ranges_for_many_b(const BType &B, const ::std::tuple<> &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, TensorConcept BType, typename BHead>
inline auto get_grid_ranges_for_many_b(const BType &B, const ::std::tuple<BHead> &B_indices)
    -> ::std::enable_if<::std::is_same_v<BHead, UniqueIndex>, size_t> {
    if constexpr (einsums::detail::IsTiledTensorV<BType>) {
        return B.grid_size(BDim);
    } else if constexpr (einsums::detail::IsBlockTensorV<BType>) {
        return B.num_blocks();
    } else {
        return 1;
    }
}

template <typename UniqueIndex, int BDim, TensorConcept BType, typename BHead,
          typename... BIndices>
inline size_t get_grid_ranges_for_many_b(const BType &B, const ::std::tuple<BHead, BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<BHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<BType>) {
            return B.grid_size(BDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<BType>) {
            return B.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, BDim + 1>(B, ::std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, TensorConcept AType, 
          TensorConcept BType, typename... BIndices>
inline size_t get_grid_ranges_for_many_a(const AType &A, const ::std::tuple<> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, TensorConcept AType,
          TensorConcept BType, typename AHead, typename... BIndices>
inline size_t get_grid_ranges_for_many_a(const AType &A, const ::std::tuple<AHead> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<AType>) {
            return A.grid_size(ADim);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, TensorConcept AType,
          TensorConcept BType, typename AHead, typename... AIndices,
          typename... BIndices>
inline auto get_grid_ranges_for_many_a(const AType&A, const ::std::tuple<AHead, AIndices...> &A_indices,
                                       const BType   &B,
                                       const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<AType>) {
            return A.grid_size(ADim);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, ADim + 1>(A, ::std::tuple<AIndices...>(), B, B_indices);
    }
}

// In these functions, leave CType as typename to allow for scalar types and tensor types.
template <typename UniqueIndex, int CDim, typename CType,
          TensorConcept AType, TensorConcept BType,
          typename... AIndices, typename... BIndices>
inline size_t get_grid_ranges_for_many_c(const CType &C, const ::std::tuple<> &C_indices,
                                         const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim,  typename CType,
          TensorConcept AType, TensorConcept BType,
          typename CHead, typename... AIndices, typename... BIndices>
inline size_t get_grid_ranges_for_many_c(const CType &C, const ::std::tuple<CHead> &C_indices,
                                         const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<CType>) {
            return C.grid_size(CDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<CType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim,  typename CType,
          TensorConcept AType, TensorConcept BType,
          typename CHead, typename... CIndices, typename... AIndices, typename... BIndices>
inline auto get_grid_ranges_for_many_c(const CType &C, const ::std::tuple<CHead, CIndices...> &C_indices,
                                       const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                       const BType   &B,
                                       const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<CType>) {
            return C.grid_size(CDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<CType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_c<UniqueIndex, CDim + 1>(C, ::std::tuple<CIndices...>(), A, A_indices, B, B_indices);
    }
}

/**
 * @brief Finds the tile grid dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param B The B tensor.
 * @param B_indices The indices for the B tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, TensorConcept AType,
          TensorConcept BType,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... AllUniqueIndices>
inline auto get_grid_ranges_for_many(const CType &C, const ::std::tuple<CIndices...> &C_indices,
                                     const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                     const BType &B, const ::std::tuple<BIndices...> &B_indices,
                                     const ::std::tuple<AllUniqueIndices...> &All_unique_indices) {
    return ::std::array{get_grid_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

} // namespace detail

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)