//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

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

template <typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices, typename... LinkUniqueIndices,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims, typename... LinkDims,
          typename... TargetPositionInC, typename... LinkPositionInLink, typename CType, CoreTensorConcept AType, CoreTensorConcept BType>
    requires(CoreTensorConcept<CType> || (!TensorConcept<CType> && sizeof...(CIndices) == 0))
void einsum_generic_algorithm(std::tuple<CUniqueIndices...> const &C_unique, std::tuple<AUniqueIndices...> const & /*A_unique*/,
                              std::tuple<BUniqueIndices...> const & /*B_unique*/, std::tuple<LinkUniqueIndices...> const &link_unique,
                              std::tuple<CIndices...> const &C_indices, std::tuple<AIndices...> const &A_indices,
                              std::tuple<BIndices...> const &B_indices, std::tuple<TargetDims...> const &target_dims,
                              std::tuple<LinkDims...> const &link_dims, std::tuple<TargetPositionInC...> const &target_position_in_C,
                              std::tuple<LinkPositionInLink...> const &link_position_in_link, ValueTypeT<CType> const C_prefactor, CType *C,
                              std::conditional_t<(sizeof(typename AType::ValueType) > sizeof(typename BType::ValueType)),
                                                 typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                              AType const &A, BType const &B) {
    EINSUMS_PROFILE_SCOPE("TensorAlgebra");

    using ADataType        = typename AType::ValueType;
    using BDataType        = typename BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    if constexpr (sizeof...(CIndices) == 0 && sizeof...(LinkDims) != 0) {
        CDataType sum{0.0};

        Stride<sizeof...(LinkDims)> link_index_strides;
        size_t                      link_elements = dims_to_strides(link_dims, link_index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < link_elements; item++) {
            thread_local std::array<int64_t, sizeof...(LinkDims)> link_combination;
            sentinel_to_indices(item, link_index_strides, link_combination);

            // Print::Indent _indent;

            // Construct the tuples that will be used to access the tensor elements of A and B
            auto A_order = detail::construct_indices_from_unique_combination(A_indices, C_unique, std::tuple<>{}, {}, link_unique,
                                                                             link_combination, link_position_in_link);
            auto B_order = detail::construct_indices_from_unique_combination(B_indices, C_unique, std::tuple<>{}, {}, link_unique,
                                                                             link_combination, link_position_in_link);

            // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
            ADataType A_value = subscript_tensor(A, A_order);

            BDataType B_value = subscript_tensor(B, B_order);

            sum += AB_prefactor * A_value * B_value;
        }

        auto &target_value = static_cast<CDataType &>(*C);
        if (C_prefactor == CDataType{0.0})
            target_value = CDataType{0.0};
        target_value *= C_prefactor;
        target_value += sum;
    } else if constexpr (sizeof...(LinkDims) != 0) {
        Stride<sizeof...(TargetDims)> target_index_strides;
        size_t                        target_elements = dims_to_strides(target_dims, target_index_strides);

        Stride<sizeof...(LinkDims)> link_index_strides;
        size_t                      link_elements = dims_to_strides(link_dims, link_index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < target_elements; item++) {
            thread_local std::array<int64_t, sizeof...(TargetDims)> target_combination;
            sentinel_to_indices(item, target_index_strides, target_combination);

            // println("target_combination: {}", print_tuple_no_type(target_combination));
            auto C_order = detail::construct_indices_from_unique_combination(C_indices, C_unique, target_combination, target_position_in_C,
                                                                             std::tuple<>(), std::tuple<>(), target_position_in_C);
            // println("C_order: {}", print_tuple_no_type(C_order));

            // This is the generic case.
            CDataType                                             sum{0};
            thread_local std::array<int64_t, sizeof...(LinkDims)> link_combination;
            for (size_t item2 = 0; item2 < link_elements; item2++) {
                // Print::Indent _indent;

                sentinel_to_indices(item2, link_index_strides, link_combination);

                // Construct the tuples that will be used to access the tensor elements of A and B
                auto A_order = detail::construct_indices_from_unique_combination(
                    A_indices, C_unique, target_combination, target_position_in_C, link_unique, link_combination, link_position_in_link);
                auto B_order = detail::construct_indices_from_unique_combination(
                    B_indices, C_unique, target_combination, target_position_in_C, link_unique, link_combination, link_position_in_link);

                // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
                ADataType A_value = subscript_tensor(A, A_order);
                BDataType B_value = subscript_tensor(B, B_order);

                sum += AB_prefactor * A_value * B_value;
            }

            if constexpr (IsFastSubscriptableV<CType>) {
                CDataType &target_value = C->subscript(C_order);
                if (C_prefactor == CDataType{0.0})
                    target_value = CDataType{0.0};
                target_value *= C_prefactor;
                target_value += sum;
            } else {

                CDataType &target_value = subscript_tensor(*C, C_order);
                if (C_prefactor == CDataType{0.0})
                    target_value = CDataType{0.0};
                target_value *= C_prefactor;
                target_value += sum;
            }
        }
    } else {
        // println("beginning contraction");
        // println("target_dims {}", target_dims);
        // println("target_position_in_C {}", print_tuple_no_type(target_position_in_C));
        // println("AUniqueIndices... {}", print_tuple_no_type(A_unique));
        // println("BUniqueIndices... {}", print_tuple_no_type(B_unique));
        // println("CUniqueIndices... {}", print_tuple_no_type(C_unique));
        // println("LinkUniqueIndices... {}", print_tuple_no_type(link_unique));
        // println("AIndices... {}", print_tuple_no_type(A_indices));
        // println("BIndices... {}", print_tuple_no_type(B_indices));
        // println("CIndices... {}", print_tuple_no_type(C_indices));

        Stride<sizeof...(TargetDims)> target_index_strides;
        size_t                        target_elements = dims_to_strides(target_dims, target_index_strides);

        for (size_t item = 0; item < target_elements; item++) {

            thread_local std::array<int64_t, sizeof...(TargetDims)> target_combination;
            sentinel_to_indices(item, target_index_strides, target_combination);

            // Construct the tuples that will be used to access the tensor elements of A and B
            auto A_order = detail::construct_indices_from_unique_combination(A_indices, C_unique, target_combination, target_position_in_C,
                                                                             std::tuple<>(), std::tuple<>(), target_position_in_C);
            auto B_order = detail::construct_indices_from_unique_combination(B_indices, C_unique, target_combination, target_position_in_C,
                                                                             std::tuple<>(), std::tuple<>(), target_position_in_C);
            auto C_order = detail::construct_indices_from_unique_combination(C_indices, C_unique, target_combination, target_position_in_C,
                                                                             std::tuple<>(), std::tuple<>(), target_position_in_C);

            // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
            ADataType A_value = subscript_tensor(A, A_order);
            BDataType B_value = subscript_tensor(B, B_order);

            CDataType sum = AB_prefactor * A_value * B_value;

            if constexpr (IsFastSubscriptableV<CType>) {
                CDataType &target_value = C->subscript(C_order);
                if (C_prefactor == CDataType{0.0})
                    target_value = CDataType{0.0};
                target_value *= C_prefactor;
                target_value += sum;
            } else {

                CDataType &target_value = subscript_tensor(*C, C_order);
                if (C_prefactor == CDataType{0.0})
                    target_value = CDataType{0.0};
                target_value *= C_prefactor;
                target_value += sum;
            }
        }
    }
}
} // namespace einsums::tensor_algebra::detail