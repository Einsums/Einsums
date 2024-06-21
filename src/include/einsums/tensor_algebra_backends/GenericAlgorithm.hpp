#pragma once

#include "einsums/_Compiler.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/Section.hpp"

#include <cmath>
#include <cstddef>
#include <range/v3/view/cartesian_product.hpp>
#include <tuple>
#include <type_traits>

namespace einsums::tensor_algebra::detail {

template <typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices, typename... LinkUniqueIndices,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims, typename... LinkDims,
          typename... TargetPositionInC, typename... LinkPositionInLink, template <typename, size_t> typename CType, typename CDataType,
          size_t CRank, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank>
#ifdef __HIP__
    requires requires {
        requires !DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires !DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires !DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
#endif
void einsum_generic_algorithm(const std::tuple<CUniqueIndices...> &C_unique, const std::tuple<AUniqueIndices...> & /*A_unique*/,
                              const std::tuple<BUniqueIndices...> & /*B_unique*/, const std::tuple<LinkUniqueIndices...> &link_unique,
                              const std::tuple<CIndices...> & /*C_indices*/, const std::tuple<AIndices...> & /*A_indices*/,
                              const std::tuple<BIndices...> & /*B_indices*/, const std::tuple<TargetDims...> &target_dims,
                              const std::tuple<LinkDims...> &link_dims, const std::tuple<TargetPositionInC...> &target_position_in_C,
                              const std::tuple<LinkPositionInLink...> &link_position_in_link, const CDataType C_prefactor,
                              CType<CDataType, CRank>                                                                *C,
                              const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                              const AType<ADataType, ARank> &A, const BType<BDataType, BRank> &B) {
    LabeledSection0();

    auto view = std::apply(ranges::views::cartesian_product, target_dims);

    if constexpr (sizeof...(CIndices) == 0 && sizeof...(LinkDims) != 0) {
        CDataType sum{0};
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

        EINSUMS_OMP_PARALLEL_FOR
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