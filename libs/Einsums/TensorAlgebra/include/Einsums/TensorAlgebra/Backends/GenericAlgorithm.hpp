//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <range/v3/view/cartesian_product.hpp>

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
                              std::tuple<CIndices...> const & /*C_indices*/, std::tuple<AIndices...> const & /*A_indices*/,
                              std::tuple<BIndices...> const & /*B_indices*/, std::tuple<TargetDims...> const &target_dims,
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

    auto view = std::apply(ranges::views::cartesian_product, target_dims);

    if constexpr (sizeof...(CIndices) == 0 && sizeof...(LinkDims) != 0) {
        CDataType sum{0.0};
        for (auto link_combination : std::apply(ranges::views::cartesian_product, link_dims)) {
            // Print::Indent _indent;

            // Construct the tuples that will be used to access the tensor elements of A and B
            auto A_order = detail::construct_indices_from_unique_combination<AIndices...>(C_unique, {}, {}, link_unique, link_combination,
                                                                                          link_position_in_link);
            auto B_order = detail::construct_indices_from_unique_combination<BIndices...>(C_unique, {}, {}, link_unique, link_combination,
                                                                                          link_position_in_link);

            // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
            ADataType A_value = std::apply(A, A_order);
            BDataType B_value = std::apply(B, B_order);

            sum += AB_prefactor * A_value * B_value;
        }

        auto &target_value = static_cast<CDataType &>(*C);
        if (C_prefactor == CDataType{0.0})
            target_value = CDataType{0.0};
        target_value *= C_prefactor;
        target_value += sum;
    } else if constexpr (sizeof...(LinkDims) != 0) {
        EINSUMS_OMP_PARALLEL_FOR
        for (auto it = view.begin(); it < view.end(); it++) {
            // println("target_combination: {}", print_tuple_no_type(target_combination));
            auto C_order = detail::construct_indices_from_unique_combination<CIndices...>(
                C_unique, *it, target_position_in_C, std::tuple<>(), std::tuple<>(), target_position_in_C);
            // println("C_order: {}", print_tuple_no_type(C_order));

            // This is the generic case.
            CDataType sum{0};
            for (auto link_combination : std::apply(ranges::views::cartesian_product, link_dims)) {
                // Print::Indent _indent;

                // Construct the tuples that will be used to access the tensor elements of A and B
                auto A_order = detail::construct_indices_from_unique_combination<AIndices...>(
                    C_unique, *it, target_position_in_C, link_unique, link_combination, link_position_in_link);
                auto B_order = detail::construct_indices_from_unique_combination<BIndices...>(
                    C_unique, *it, target_position_in_C, link_unique, link_combination, link_position_in_link);

                // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
                ADataType A_value = std::apply(A, A_order);
                BDataType B_value = std::apply(B, B_order);

                sum += AB_prefactor * A_value * B_value;
            }

            CDataType &target_value = std::apply(*C, C_order);
            if (C_prefactor == CDataType{0.0})
                target_value = CDataType{0.0};
            target_value *= C_prefactor;
            target_value += sum;
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

        // EINSUMS_OMP_PARALLEL_FOR
        for (auto it = view.begin(); it < view.end(); it++) {

            // Construct the tuples that will be used to access the tensor elements of A and B
            auto A_order = detail::construct_indices_from_unique_combination<AIndices...>(
                C_unique, *it, target_position_in_C, std::tuple<>(), std::tuple<>(), target_position_in_C);
            auto B_order = detail::construct_indices_from_unique_combination<BIndices...>(
                C_unique, *it, target_position_in_C, std::tuple<>(), std::tuple<>(), target_position_in_C);
            auto C_order = detail::construct_indices_from_unique_combination<CIndices...>(
                C_unique, *it, target_position_in_C, std::tuple<>(), std::tuple<>(), target_position_in_C);

            // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
            ADataType A_value = std::apply(A, A_order);
            BDataType B_value = std::apply(B, B_order);

            CDataType sum = AB_prefactor * A_value * B_value;

            CDataType &target_value = std::apply(*C, C_order);
            if (C_prefactor == CDataType{0.0})
                target_value = CDataType{0.0};
            target_value *= C_prefactor;
            target_value += sum;
        }
    }
}
} // namespace einsums::tensor_algebra::detail