//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace einsums::tensor_algebra::detail {

template <size_t __I, typename T, bool ConjA, bool ConjB, typename... LinkDims, typename... LinkUnique, typename... LinkPositionInA,
          typename... LinkPositionInB, CoreTensorConcept AType, CoreTensorConcept BType>
    requires(!BasicTensorConcept<AType> || !BasicTensorConcept<BType>)
std::remove_cvref_t<T> einsums_generic_link_loop(std::tuple<LinkDims...> const &link_dims, std::tuple<LinkUnique...> const &link_unique,
                                                 std::tuple<LinkPositionInA...> const &link_position_in_A,
                                                 std::tuple<LinkPositionInB...> const &link_position_in_B,
                                                 std::array<size_t, AType::Rank> &A_indices, std::array<size_t, BType::Rank> &B_indices,
                                                 AType const &A, BType const &B) {
    if constexpr (sizeof...(LinkDims) == __I) {
        auto A_val = subscript_tensor(A, A_indices);
        auto B_val = subscript_tensor(B, B_indices);

        if constexpr (IsComplexV<std::remove_cvref_t<decltype(A_val)>> && ConjA) {
            A_val = std::conj(A_val);
        }

        if constexpr (IsComplexV<std::remove_cvref_t<decltype(B_val)>> && ConjB) {
            B_val = std::conj(B_val);
        }
        return A_val * B_val;
    } else {
        size_t const curr_dim = std::get<__I>(link_dims);

        T sum{0.0};

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < curr_dim; i++) {
            for_sequence<sizeof...(LinkPositionInA) / 2>([&](auto n) {
                if constexpr (std::is_same_v<std::remove_cvref_t<std::tuple_element_t<2 * decltype(n)::value,
                                                                                      std::remove_cvref_t<decltype(link_position_in_A)>>>,
                                             std::remove_cvref_t<std::tuple_element_t<__I, std::remove_cvref_t<decltype(link_unique)>>>>) {
                    A_indices[std::get<2 * decltype(n)::value + 1>(link_position_in_A)] = i;
                }
            });
            for_sequence<sizeof...(LinkPositionInB) / 2>([&](auto n) {
                if constexpr (std::is_same_v<std::remove_cvref_t<std::tuple_element_t<2 * decltype(n)::value,
                                                                                      std::remove_cvref_t<decltype(link_position_in_B)>>>,
                                             std::remove_cvref_t<std::tuple_element_t<__I, std::remove_cvref_t<decltype(link_unique)>>>>) {
                    B_indices[std::get<2 * decltype(n)::value + 1>(link_position_in_B)] = i;
                }
            });
            sum += einsums_generic_link_loop<__I + 1, T, ConjA, ConjB>(link_dims, link_unique, link_position_in_A, link_position_in_B,
                                                                     A_indices, B_indices, A, B);
        }
    }
}

template <size_t __I, bool ConjA, bool ConjB, typename... TargetDims, typename... LinkDims, typename... CUnique, typename... LinkUnique,
          typename... TargetPositionInC, typename... TargetPositionInA, typename... TargetPositionInB, typename... LinkPositionInA,
          typename... LinkPositionInB, CoreTensorConcept CType, CoreTensorConcept AType, CoreTensorConcept BType, typename T>
    requires(!BasicTensorConcept<AType> || !BasicTensorConcept<BType>)
void einsums_generic_target_loop(std::tuple<TargetDims...> const &target_dims, std::tuple<LinkDims...> const &link_dims,
                                 std::tuple<CUnique...> const &C_unique, std::tuple<LinkUnique...> const &link_unique,
                                 std::tuple<TargetPositionInC...> const &target_position_in_C,
                                 std::tuple<TargetPositionInA...> const &target_position_in_A,
                                 std::tuple<TargetPositionInB...> const &target_position_in_B,
                                 std::tuple<LinkPositionInA...> const   &link_position_in_A,
                                 std::tuple<LinkPositionInB...> const &link_position_in_B, std::array<size_t, CType::Rank> &C_indices,
                                 std::array<size_t, AType::Rank> &A_indices, std::array<size_t, BType::Rank> &B_indices, T &&C_prefactor,
                                 CType *C, T &&AB_prefactor, AType const &A, BType const &B) {
    if constexpr (sizeof...(TargetDims) == __I) {
        subscript_tensor(*C, C_indices) +=
            AB_prefactor * einsums_generic_link_loop<0, T, ConjA, ConjB>(link_dims, link_unique, link_position_in_A, link_position_in_B,
                                                                         A_indices, B_indices, A, B);
    } else {
        size_t const curr_dim = std::get<__I>(target_dims);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < curr_dim; i++) {
            for_sequence<sizeof...(TargetPositionInC) / 2>([&](auto n) {
                if constexpr (std::is_same_v<std::remove_cvref_t<std::tuple_element_t<2 * decltype(n)::value,
                                                                                      std::remove_cvref_t<decltype(target_position_in_C)>>>,
                                             std::remove_cvref_t<std::tuple_element_t<__I, std::remove_cvref_t<decltype(C_unique)>>>>) {
                    C_indices[std::get<2 * decltype(n)::value + 1>(target_position_in_C)] = i;
                }
            });
            for_sequence<sizeof...(TargetPositionInA) / 2>([&](auto n) {
                if constexpr (std::is_same_v<std::remove_cvref_t<std::tuple_element_t<2 * decltype(n)::value,
                                                                                      std::remove_cvref_t<decltype(target_position_in_A)>>>,
                                             std::remove_cvref_t<std::tuple_element_t<__I, std::remove_cvref_t<decltype(C_unique)>>>>) {
                    A_indices[std::get<2 * decltype(n)::value + 1>(target_position_in_A)] = i;
                }
            });
            for_sequence<sizeof...(TargetPositionInB) / 2>([&](auto n) {
                if constexpr (std::is_same_v<std::remove_cvref_t<std::tuple_element_t<2 * decltype(n)::value,
                                                                                      std::remove_cvref_t<decltype(target_position_in_B)>>>,
                                             std::remove_cvref_t<std::tuple_element_t<__I, std::remove_cvref_t<decltype(C_unique)>>>>) {
                    B_indices[std::get<2 * decltype(n)::value + 1>(target_position_in_B)] = i;
                }
            });

            einsums_generic_target_loop<__I + 1, ConjA, ConjB>(target_dims, link_dims, C_unique, link_unique, target_position_in_C,
                                                             target_position_in_A, target_position_in_B, link_position_in_A,
                                                             link_position_in_B, C_indices, A_indices, B_indices,
                                                             std::forward<T>(C_prefactor), C, std::forward<T>(AB_prefactor), A, B);
        }
    }
}

template <bool ConjA, bool ConjB, typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices,
          typename... LinkUniqueIndices, typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims,
          typename... LinkDims, typename... TargetPositionInC, typename... LinkPositionInLink, typename CType, CoreTensorConcept AType,
          CoreTensorConcept BType>
    requires requires {
        requires CoreBasicTensorConcept<CType> || (!TensorConcept<CType> && sizeof...(CIndices) == 0);
        requires !BasicTensorConcept<AType> || !BasicTensorConcept<BType>;
    }
void einsum_generic_algorithm(std::tuple<CUniqueIndices...> const &C_unique, std::tuple<AUniqueIndices...> const & /*A_unique*/,
                              std::tuple<BUniqueIndices...> const & /*B_unique*/, std::tuple<LinkUniqueIndices...> const &link_unique,
                              std::tuple<CIndices...> const &C_indices, std::tuple<AIndices...> const &A_indices,
                              std::tuple<BIndices...> const &B_indices, std::tuple<TargetDims...> const &target_dims,
                              std::tuple<LinkDims...> const &link_dims, std::tuple<TargetPositionInC...> const &target_position_in_C,
                              std::tuple<LinkPositionInLink...> const &link_position_in_link, ValueTypeT<CType> const C_prefactor, CType *C,
                              std::conditional_t<(sizeof(typename AType::ValueType) > sizeof(typename BType::ValueType)),
                                                 typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                              AType const &A, BType const &B) {
    LabeledSection0();

    using ADataType        = typename AType::ValueType;
    using BDataType        = typename BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    auto const target_position_in_A = find_type_with_position(C_unique, A_indices);
    auto const target_position_in_B = find_type_with_position(C_unique, B_indices);
    auto const link_position_in_A   = find_type_with_position(link_unique, A_indices);
    auto const link_position_in_B   = find_type_with_position(link_unique, B_indices);

    std::array<size_t, ARank> A_index;
    std::array<size_t, BRank> B_index;

    A_index.fill(0);
    B_index.fill(0);

    if constexpr (sizeof...(CIndices) == 0 && sizeof...(LinkDims) != 0) {

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }

        *C += AB_prefactor * einsums_generic_link_loop<0, ConjA, ConjB>(link_dims, link_unique, link_position_in_A, link_position_in_B,
                                                                        A_index, B_index, A, B);
    } else {
        std::array<size_t, CRank> C_index;

        if (C_prefactor == CDataType{0.0}) {
            C->zero();
        } else {
            *C *= C_prefactor;
        }

        C_index.fill(0);
        einsums_generic_target_loop<0, ConjA, ConjB>(target_dims, link_dims, C_unique, link_unique, target_position_in_C,
                                                     target_position_in_A, target_position_in_B, link_position_in_A, link_position_in_B,
                                                     C_index, A_index, B_index, C_prefactor, C, AB_prefactor, A, B);
    }
}
} // namespace einsums::tensor_algebra::detail