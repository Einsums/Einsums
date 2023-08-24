#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_Index.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

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

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

namespace detail {

template <typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices, typename... LinkUniqueIndices,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims, typename... LinkDims,
          typename... TargetPositionInC, typename... LinkPositionInLink, template <typename, size_t> typename CType, typename CDataType,
          size_t CRank, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank>
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

        CDataType &target_value = *C;
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

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const CDataType C_prefactor, const std::tuple<CIndices...> & /*Cs*/, CType<CDataType, CRank> *C,
            const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
            const std::tuple<AIndices...> & /*As*/, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> & /*Bs*/,
            const BType<BDataType, BRank> &B)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>>> {
    print::Indent _indent;

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();
    using ABDataType         = std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    // 1. Ensure the ranks are correct. (Compile-time check.)
    static_assert(sizeof...(CIndices) == CRank, "Rank of C does not match Indices given for C.");
    static_assert(sizeof...(AIndices) == ARank, "Rank of A does not match Indices given for A.");
    static_assert(sizeof...(BIndices) == BRank, "Rank of B does not match Indices given for B.");

    // 2. Determine the links from AIndices and BIndices
    constexpr auto linksAB = intersect_t<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    // 2a. Remove any links that appear in the target
    constexpr auto links = difference_t<decltype(linksAB), std::tuple<CIndices...>>();

    // 3. Determine the links between CIndices and AIndices
    constexpr auto CAlinks = intersect_t<std::tuple<CIndices...>, std::tuple<AIndices...>>();

    // 4. Determine the links between CIndices and BIndices
    constexpr auto CBlinks = intersect_t<std::tuple<CIndices...>, std::tuple<BIndices...>>();

    // Remove anything from A that exists in C
    constexpr auto CminusA = difference_t<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto CminusB = difference_t<std::tuple<CIndices...>, std::tuple<BIndices...>>();

    constexpr bool have_remaining_indices_in_CminusA = std::tuple_size_v<decltype(CminusA)>;
    constexpr bool have_remaining_indices_in_CminusB = std::tuple_size_v<decltype(CminusB)>;

    // Determine unique indices in A
    constexpr auto A_only = difference_t<std::tuple<AIndices...>, decltype(links)>();
    constexpr auto B_only = difference_t<std::tuple<BIndices...>, decltype(links)>();

    constexpr auto A_unique    = unique_t<std::tuple<AIndices...>>();
    constexpr auto B_unique    = unique_t<std::tuple<BIndices...>>();
    constexpr auto C_unique    = unique_t<std::tuple<CIndices...>>();
    constexpr auto link_unique = c_unique_t<decltype(links)>();

    constexpr bool A_hadamard_found = std::tuple_size_v<std::tuple<AIndices...>> != std::tuple_size_v<decltype(A_unique)>;
    constexpr bool B_hadamard_found = std::tuple_size_v<std::tuple<BIndices...>> != std::tuple_size_v<decltype(B_unique)>;
    constexpr bool C_hadamard_found = std::tuple_size_v<std::tuple<CIndices...>> != std::tuple_size_v<decltype(C_unique)>;

    constexpr auto link_position_in_A    = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B    = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto link_position_in_link = detail::find_type_with_position(link_unique, links);

    constexpr auto target_position_in_A = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto target_position_in_C = detail::find_type_with_position(C_unique, C_indices);

    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = detail::find_type_with_position(B_indices, C_indices);

    auto unique_target_dims = detail::get_dim_ranges_for(*C, detail::unique_find_type_with_position(C_unique, C_indices));
    auto unique_link_dims   = detail::get_dim_ranges_for(A, link_position_in_A);

    constexpr auto contiguous_link_position_in_A = detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B = detail::contiguous_positions(link_position_in_B);

    constexpr auto contiguous_target_position_in_A = detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B = detail::contiguous_positions(target_position_in_B);

    constexpr auto contiguous_A_targets_in_C = detail::contiguous_positions(A_target_position_in_C);
    constexpr auto contiguous_B_targets_in_C = detail::contiguous_positions(B_target_position_in_C);

    constexpr auto same_ordering_link_position_in_AB   = detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA = detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB = detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr auto C_exactly_matches_A =
        sizeof...(CIndices) == sizeof...(AIndices) && same_indices<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto C_exactly_matches_B =
        sizeof...(CIndices) == sizeof...(BIndices) && same_indices<std::tuple<CIndices...>, std::tuple<BIndices...>>();
    constexpr auto A_exactly_matches_B = same_indices<std::tuple<AIndices...>, std::tuple<BIndices...>>();

    constexpr auto is_gemm_possible = have_remaining_indices_in_CminusA && have_remaining_indices_in_CminusB &&
                                      contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                                      contiguous_target_position_in_B && contiguous_A_targets_in_C && contiguous_B_targets_in_C &&
                                      same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                                      same_ordering_target_position_in_CB && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    constexpr auto is_gemv_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                                      same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                                      !same_ordering_target_position_in_CB && std::tuple_size_v<decltype(B_target_position_in_C)> == 0 &&
                                      !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    constexpr auto element_wise_multiplication =
        C_exactly_matches_A && C_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    constexpr auto dot_product =
        sizeof...(CIndices) == 0 && A_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    constexpr auto outer_product = std::tuple_size_v<decltype(linksAB)> == 0 && contiguous_target_position_in_A &&
                                   contiguous_target_position_in_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    // println("A_indices {}", print_tuple_no_type(A_indices));
    // println("B_indices {}", print_tuple_no_type(B_indices));
    // println("C_indices {}", print_tuple_no_type(C_indices));
    // println("A_unique {}", print_tuple_no_type(A_unique));
    // println("B_unique {}", print_tuple_no_type(B_unique));
    // println("C_unique {}", print_tuple_no_type(C_unique));
    // println("target_position_in_A {}", print_tuple_no_type(target_position_in_A));
    // println("target_position_in_B {}", print_tuple_no_type(target_position_in_B));
    // println("target_position_in_C {}", print_tuple_no_type(target_position_in_C));
    // println("link_position_in_A {}", print_tuple_no_type(link_position_in_A));
    // println("link_position_in_B {}", print_tuple_no_type(link_position_in_B));
    // println("contiguous_link_position_in_A {}", contiguous_link_position_in_A);
    // println("contiguous_link_position_in_B {}", contiguous_link_position_in_B);
    // println("contiguous_target_position_in_A {}", contiguous_target_position_in_A);
    // println("same_ordering_link_position_in_AB {}", same_ordering_link_position_in_AB);
    // println("same_ordering_target_position_in_CA {}", same_ordering_target_position_in_CA);
    // println("same_ordering_target_position_in_CB {}", same_ordering_target_position_in_CB);
    // println("std::tuple_size_v<decltype(B_target_position_in_C)> == 0 {}", std::tuple_size_v<decltype(B_target_position_in_C)> == 0);
    // println("A_hadamard_found {}", A_hadamard_found);
    // println("B_hadamard_found {}", B_hadamard_found);
    // println("C_hadamard_found {}", C_hadamard_found);

    // println("is_gemv_possible {}", is_gemv_possible);
    // println("is_gemm_possible {}", is_gemm_possible);
    // println("dot_product {}", dot_product);

    // Runtime check of sizes
#if defined(EINSUMS_RUNTIME_INDICES_CHECK)
    bool runtime_indices_abort{false};
    for_sequence<ARank>([&](auto a) {
        size_t dimA = A.dim(a);
        for_sequence<BRank>([&](auto b) {
            size_t dimB = B.dim(b);
            if (std::get<a>(A_indices).letter == std::get<b>(B_indices).letter) {
                if (dimA != dimB) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(),
                            print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));
                    runtime_indices_abort = true;
                }
            }
        });
        for_sequence<CRank>([&](auto c) {
            size_t dimC = C->dim(c);
            if (std::get<a>(A_indices).letter == std::get<c>(C_indices).letter) {
                if (dimA != dimC) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(),
                            print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));
                    runtime_indices_abort = true;
                }
            }
        });
    });
    for_sequence<BRank>([&](auto b) {
        size_t dimB = B.dim(b);
        for_sequence<CRank>([&](auto c) {
            size_t dimC = C->dim(c);
            if (std::get<b>(B_indices).letter == std::get<c>(C_indices).letter) {
                if (dimB != dimC) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(),
                            print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));
                    runtime_indices_abort = true;
                }
            }
        });
    });

    if (runtime_indices_abort) {
        throw std::runtime_error("einsum: Inconsistent dimensions found!");
    }
#endif

    if constexpr (!std::is_same_v<CDataType, ADataType> || !std::is_same_v<CDataType, BDataType>) {
        // Mixed datatypes go directly to the generic algorithm.
        einsum_generic_algorithm(C_unique, A_unique, B_unique, link_unique, C_indices, A_indices, B_indices, unique_target_dims,
                                 unique_link_dims, target_position_in_C, link_position_in_link, C_prefactor, C, AB_prefactor, A, B);
        return;
    } else if constexpr (dot_product) {
        CDataType temp = linear_algebra::dot(A, B);
        (*C) *= C_prefactor;
        (*C) += AB_prefactor * temp;

        return;
    } else if constexpr (element_wise_multiplication) {
        timer::Timer element_wise_multiplication{"element-wise multiplication"};

        auto target_dims = get_dim_ranges<CRank>(*C);
        auto view        = std::apply(ranges::views::cartesian_product, target_dims);

        // Ensure the various tensors passed in are the same dimensionality
        if (((C->dims() != A.dims()) || C->dims() != B.dims())) {
            println_abort("einsum: at least one tensor does not have same dimensionality as destination");
        }

        // Horrible hack. For some reason, in the for loop below, the result could be
        // NAN if the target_value is initially a trash value.
        if (C_prefactor == CDataType(0)) {
            C->zero();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (auto it = view.begin(); it != view.end(); it++) {
            CDataType &target_value = std::apply(*C, *it);
            ABDataType AB_product   = std::apply(A, *it) * std::apply(B, *it);
            target_value            = C_prefactor * target_value + AB_prefactor * AB_product;
        }

        return;
    } else if constexpr (outer_product) {
        do { // do {} while (false) trick to allow us to use a break below to "break" out of the loop.
            constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;

            Dim<2> dC;
            dC[0] = product_dims(A_target_position_in_C, *C);
            dC[1] = product_dims(B_target_position_in_C, *C);
            if constexpr (swap_AB)
                std::swap(dC[0], dC[1]);

            TensorView<CDataType, 2> tC{*C, dC};

            if (C_prefactor != CDataType{1.0})
                linear_algebra::scale(C_prefactor, C);

            try {
                if constexpr (swap_AB) {
                    linear_algebra::ger(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
                } else {
                    linear_algebra::ger(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
                }
            } catch (std::runtime_error &e) {
#if defined(EINSUMS_SHOW_WARNING)
                println(
                    bg(fmt::color::yellow) | fg(fmt::color::black),
                    "Optimized outer product failed. Likely from a non-contiguous TensorView. Attempting to perform generic algorithm.");
#endif
                if (C_prefactor == CDataType{0.0}) {
#if defined(EINSUMS_SHOW_WARNING)
                    println(bg(fmt::color::red) | fg(fmt::color::white),
                            "WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor, C->name());
#endif
                } else {
                    linear_algebra::scale(1.0 / C_prefactor, C);
                }
                break; // out of the do {} while(false) loop.
            }
            // If we got to this position, assume we successfully called ger.
            return;
        } while (false);
    } else if constexpr (!OnlyUseGenericAlgorithm) {
        do { // do {} while (false) trick to allow us to use a break below to "break" out of the loop.
            if constexpr (is_gemv_possible) {

                if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                    // Fall through to generic algorithm.
                    break;
                }

                constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;

                Dim<2>    dA;
                Dim<1>    dB, dC;
                Stride<2> sA;
                Stride<1> sB, sC;

                dA[0] = product_dims(A_target_position_in_C, *C);
                dA[1] = product_dims(link_position_in_A, A);
                sA[0] = last_stride(target_position_in_A, A);
                sA[1] = last_stride(link_position_in_A, A);
                if constexpr (transpose_A) {
                    std::swap(dA[0], dA[1]);
                    std::swap(sA[0], sA[1]);
                }

                dB[0] = product_dims(link_position_in_B, B);
                sB[0] = last_stride(link_position_in_B, B);

                dC[0] = product_dims(A_target_position_in_C, *C);
                sC[0] = last_stride(A_target_position_in_C, *C);

                const TensorView<ADataType, 2> tA{const_cast<AType<ADataType, ARank> &>(A), dA, sA};
                const TensorView<BDataType, 1> tB{const_cast<BType<BDataType, BRank> &>(B), dB, sB};
                TensorView<CDataType, 1>       tC{*C, dC, sC};

                // println(*C);
                // println(tC);
                // println(A);
                // println(tA);
                // println(B);
                // println(tB);

                if constexpr (transpose_A) {
                    linear_algebra::gemv<true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                } else {
                    linear_algebra::gemv<false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                }

                return;
            }
            // To use a gemm the input tensors need to be at least rank 2
            else if constexpr (CRank >= 2 && ARank >= 2 && BRank >= 2) {
                if constexpr (!A_hadamard_found && !B_hadamard_found && !C_hadamard_found) {
                    if constexpr (is_gemm_possible) {

                        if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                            // Fall through to generic algorithm.
                            break;
                        }

                        constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;
                        constexpr bool transpose_B = std::get<1>(link_position_in_B) != 0;
                        constexpr bool transpose_C = std::get<1>(A_target_position_in_C) != 0;

                        Dim<2>    dA, dB, dC;
                        Stride<2> sA, sB, sC;

                        dA[0] = product_dims(A_target_position_in_C, *C);
                        dA[1] = product_dims(link_position_in_A, A);
                        sA[0] = last_stride(target_position_in_A, A);
                        sA[1] = last_stride(link_position_in_A, A);
                        if constexpr (transpose_A) {
                            std::swap(dA[0], dA[1]);
                            std::swap(sA[0], sA[1]);
                        }

                        dB[0] = product_dims(link_position_in_B, B);
                        dB[1] = product_dims(B_target_position_in_C, *C);
                        sB[0] = last_stride(link_position_in_B, B);
                        sB[1] = last_stride(target_position_in_B, B);
                        if constexpr (transpose_B) {
                            std::swap(dB[0], dB[1]);
                            std::swap(sB[0], sB[1]);
                        }

                        dC[0] = product_dims(A_target_position_in_C, *C);
                        dC[1] = product_dims(B_target_position_in_C, *C);
                        sC[0] = last_stride(A_target_position_in_C, *C);
                        sC[1] = last_stride(B_target_position_in_C, *C);
                        if constexpr (transpose_C) {
                            std::swap(dC[0], dC[1]);
                            std::swap(sC[0], sC[1]);
                        }

                        TensorView<CDataType, 2>       tC{*C, dC, sC};
                        const TensorView<ADataType, 2> tA{const_cast<AType<ADataType, ARank> &>(A), dA, sA};
                        const TensorView<BDataType, 2> tB{const_cast<BType<BDataType, BRank> &>(B), dB, sB};

                        // println("--------------------");
                        // println(*C);
                        // println(tC);
                        // println("--------------------");
                        // println(A);
                        // println(tA);
                        // println("--------------------");
                        // println(B);
                        // println(tB);
                        // println("--------------------");

                        if constexpr (!transpose_C && !transpose_A && !transpose_B) {
                            linear_algebra::gemm<false, false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && !transpose_A && transpose_B) {
                            linear_algebra::gemm<false, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && transpose_A && !transpose_B) {
                            linear_algebra::gemm<true, false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && transpose_A && transpose_B) {
                            linear_algebra::gemm<true, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && !transpose_A && !transpose_B) {
                            linear_algebra::gemm<true, true>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && !transpose_A && transpose_B) {
                            linear_algebra::gemm<false, true>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && transpose_A && !transpose_B) {
                            linear_algebra::gemm<true, false>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && transpose_A && transpose_B) {
                            linear_algebra::gemm<false, false>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else {
                            println("This GEMM case is not programmed: transpose_C {}, transpose_A {}, transpose_B {}", transpose_C,
                                    transpose_A, transpose_B);
                            std::abort();
                        }
                    }
                }
            }
            // If we make it here, then none of our algorithms for this last block could be used.
            // Fall through to the generic algorithm below.
        } while (false);
    }

    // If we somehow make it here, then none of our algorithms above could be used. Attempt to use
    // the generic algorithm instead.
    einsum_generic_algorithm(C_unique, A_unique, B_unique, link_unique, C_indices, A_indices, B_indices, unique_target_dims,
                             unique_link_dims, target_position_in_C, link_position_in_link, C_prefactor, C, AB_prefactor, A, B);
}

} // namespace detail

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename U>
auto einsum(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C, const U UAB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>> &&
                        std::is_arithmetic_v<U>> {
    using ABDataType = std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    LabeledSection1(FP_ZERO != std::fpclassify(UC_prefactor)
                        ? fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices),
                                      UAB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices),
                                      UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{})", C->name(), print_tuple_no_type(C_indices), UAB_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices)));

    const CDataType  C_prefactor  = UC_prefactor;
    const ABDataType AB_prefactor = UAB_prefactor;

#if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    // Clone C into a new tensor
    Tensor<CDataType, CRank> testC{C->dims()};
    testC = *C;

    // Perform the einsum using only the generic algorithm
    timer::push("testing");
    detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, B);
    timer::pop();
#endif

    // Perform the actual einsum
    detail::einsum<false>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);

#if defined(EINSUMS_TEST_NANS)
    if constexpr (CRank != 0) {
        auto target_dims = get_dim_ranges<CRank>(*C);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            CDataType Cvalue{std::apply(*C, target_combination)};
            if constexpr (!is_complex_v<CDataType>) {
                if (std::isnan(Cvalue)) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "NaN DETECTED!");
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor,
                            C->name(), print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    throw std::runtime_error("NAN detected in resulting tensor.");
                }

                if (std::isinf(Cvalue)) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "Infinity DETECTED!");
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor,
                            C->name(), print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    throw std::runtime_error("Infinity detected in resulting tensor.");
                }

                if (std::abs(Cvalue) > 100000000) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "Large value DETECTED!");
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor,
                            C->name(), print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    throw std::runtime_error("Large value detected in resulting tensor.");
                }
            }
        }
    }
#endif

#if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    if constexpr (CRank != 0) {
        // Need to walk through the entire C and testC comparing values and reporting differences.
        auto target_dims = get_dim_ranges<CRank>(*C);
        bool print_info_and_abort{false};

        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            CDataType Cvalue{std::apply(*C, target_combination)};
            CDataType Ctest{std::apply(testC, target_combination)};

            if constexpr (!is_complex_v<CDataType>) {
                if (std::isnan(Cvalue) || std::isnan(Ctest)) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "NAN DETECTED!");
                    println("Source tensors");
                    println(A);
                    println(B);
                    if (std::isnan(Cvalue)) {
                        println("NAN detected in C");
                        println("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(*C);
                    }
                    if (std::isnan(Ctest)) {
                        println("NAN detected in reference Ctest");
                        println("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(testC);
                    }

                    print_info_and_abort = true;
                }
            }

#    if defined(EINSUMS_USE_CATCH2)
            if constexpr (!is_complex_v<CDataType>) {
                REQUIRE_THAT(Cvalue,
                             Catch::Matchers::WithinRel(Ctest, static_cast<CDataType>(0.001)) || Catch::Matchers::WithinAbs(0, 0.0001));
                CHECK(print_info_and_abort == false);
            }
#    endif

            if (std::abs(Cvalue - Ctest) > 1.0E-6) {
                print_info_and_abort = true;
            }

            if (print_info_and_abort) {
                println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "    !!! EINSUM ERROR !!!");
                if constexpr (is_complex_v<CDataType>) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(),
                            Cvalue.imag());
                } else {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
                }
                println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ({:})", print_tuple_no_type(target_combination));
                std::string C_prefactor_string;
                if constexpr (is_complex_v<CDataType>) {
                    C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
                } else {
                    C_prefactor_string = fmt::format("{:f}", C_prefactor);
                }
                std::string AB_prefactor_string;
                if constexpr (is_complex_v<ABDataType>) {
                    AB_prefactor_string = fmt::format("({:f} + {:f}i)", AB_prefactor.real(), AB_prefactor.imag());
                } else {
                    AB_prefactor_string = fmt::format("{:f}", AB_prefactor);
                }
                println(bg(fmt::color::red) | fg(fmt::color::white), "    {} C({:}) += {:f} A({:}) * B({:})", C_prefactor_string,
                        print_tuple_no_type(C_indices), AB_prefactor_string, print_tuple_no_type(A_indices),
                        print_tuple_no_type(B_indices));

                println("Expected:");
                println(testC);
                println("Obtained");
                println(*C);
                println(A);
                println(B);
#    if defined(EINSUMS_TEST_EINSUM_ABORT)
                std::abort();
#    endif
            }
        }
    } else {
        const CDataType Cvalue = *C;
        const CDataType Ctest  = testC;

        // testC could be a Tensor<CDataType, 0> type. Cast it to the underlying data type.
        if (std::abs(Cvalue - (CDataType)testC) > 1.0E-6) {
            println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "!!! EINSUM ERROR !!!");
            if constexpr (is_complex_v<CDataType>) {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(), Cvalue.imag());
            } else {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
            }

            println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ()");
            std::string C_prefactor_string;
            if constexpr (is_complex_v<CDataType>) {
                C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
            } else {
                C_prefactor_string = fmt::format("{:f}", C_prefactor);
            }
            std::string AB_prefactor_string;
            if constexpr (is_complex_v<ABDataType>) {
                AB_prefactor_string = fmt::format("({:f} + {:f}i)", AB_prefactor.real(), AB_prefactor.imag());
            } else {
                AB_prefactor_string = fmt::format("{:f}", AB_prefactor);
            }
            println(bg(fmt::color::red) | fg(fmt::color::white), "    {} C() += {} A({:}) * B({:})", C_prefactor_string,
                    AB_prefactor_string, print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

            println("Expected:");
            println(testC);
            println("Obtained");
            println(*C);
            println(A);
            println(B);

#    if defined(EINSUMS_TEST_EINSUM_ABORT)
            std::abort();
#    endif
        }
    }
#endif
}

// Einsums with provided prefactors.
// 1. C n A n B n is defined above as the base implementation.

// 2. C n A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, *B);
}

// 3. C n A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && ~is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, B);
}

// 4. C n A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, *B);
}

// 5. C y A n B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, B);
}

// 6. C y A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, *B);
}

// 7. C y A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, B);
}

// 8. C y A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, *B);
}

//
// Einsums with default prefactors.
//

// 1. C n A n B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, B);
}

// 2. C n A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, *B);
}

// 3. C n A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, B);
}

// 4. C n A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, *B);
}

// 5. C y A n B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, B);
}

// 6. C y A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, *B);
}

// 7. C y A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, B);
}

// 8. C y A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, *B);
}

//
// Element Transform
///

template <template <typename, size_t> typename CType, size_t CRank, typename UnaryOperator, typename T = double>
auto element_transform(CType<T, CRank> *C, UnaryOperator unary_opt)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, CRank>, CType<T, CRank>>> {
    LabeledSection0();

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    EINSUMS_OMP_PARALLEL_FOR
    for (auto it = view.begin(); it != view.end(); it++) {
        T &target_value = std::apply(*C, *it);
        target_value    = unary_opt(target_value);
    }
}

template <typename SmartPtr, typename UnaryOperator>
auto element_transform(SmartPtr *C, UnaryOperator unary_opt) -> std::enable_if_t<is_smart_pointer_v<SmartPtr>> {
    element_transform(C->get(), unary_opt);
}

template <template <typename, size_t> typename CType, template <typename, size_t> typename... MultiTensors, size_t Rank,
          typename MultiOperator, typename T = double>
auto element(MultiOperator multi_opt, CType<T, Rank> *C, MultiTensors<T, Rank> &...tensors) {
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

template <unsigned int N, typename... List>
constexpr auto get_n(const std::tuple<List...> &) {
    return _get_n<N, List...>();
}

/**
 * Returns the mode-`mode` unfolding of `tensor` with modes startng at `0`
 *
 * @returns unfolded_tensor of shape ``(tensor.dim(mode), -1)``
 */
template <unsigned int mode, template <typename, size_t> typename CType, size_t CRank, typename T = double>
auto unfold(const CType<T, CRank> &source) -> std::enable_if_t<std::is_same_v<Tensor<T, CRank>, CType<T, CRank>>, Tensor<T, 2>> {
    LabeledSection1(fmt::format("mode-{} unfold on {} threads", mode, omp_get_max_threads()));

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

/** Computes the Khatri-Rao product of tensors A and B.
 *
 * Example:
 *    Tensor<2> result = khatri_rao(Indices{I, r}, A, Indices{J, r}, B);
 *
 * Result is described as {(I,J), r}. If multiple common indices are provided they will be collapsed into a single index in the result.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank,
          typename... AIndices, typename... BIndices, typename T = double>
auto khatri_rao(const std::tuple<AIndices...> &, const AType<T, ARank> &A, const std::tuple<BIndices...> &, const BType<T, BRank> &B)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, ARank>, AType<T, ARank>> &&
                            std::is_base_of_v<::einsums::detail::TensorBase<T, BRank>, BType<T, BRank>>,
                        Tensor<T, 2>> {
    LabeledSection0();

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();

    // Determine the common indices between A and B
    constexpr auto common = intersect_t<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    // Determine unique indices in A
    constexpr auto A_only = difference_t<std::tuple<AIndices...>, decltype(common)>();
    // Determine unique indices in B
    constexpr auto B_only = difference_t<std::tuple<BIndices...>, decltype(common)>();

    // Record the positions of each types.
    constexpr auto A_common_position = detail::find_type_with_position(common, A_indices);
    constexpr auto B_common_position = detail::find_type_with_position(common, B_indices);
    constexpr auto A_only_position   = detail::find_type_with_position(A_only, A_indices);
    constexpr auto B_only_position   = detail::find_type_with_position(B_only, B_indices);

    // Obtain dimensions of the indices discovered above
    auto A_common_dims = detail::get_dim_for(A, A_common_position);
    auto B_common_dims = detail::get_dim_for(B, B_common_position);
    auto A_only_dims   = detail::get_dim_for(A, A_only_position);
    auto B_only_dims   = detail::get_dim_for(B, B_only_position);

    // Sanity check - ensure the common dims between A and B are the same size.
    for_sequence<std::tuple_size_v<decltype(common)>>([&](auto i) {
        if (std::get<i>(A_common_dims) != std::get<i>(B_common_dims)) {
            throw std::runtime_error(fmt::format("Common dimensions for index {} of A and B do not match.", std::get<i>(common)));
        }
    });

    auto result_dims = std::tuple_cat(std::make_tuple("KR product"), A_only_dims, B_only_dims, A_common_dims);

    // Construct resulting tensor
    auto result = std::make_from_tuple<Tensor<T, std::tuple_size_v<decltype(result_dims)> - 1>>(result_dims);

    // Perform the actual Khatri-Rao product using our einsum routine.
    einsum(std::tuple_cat(A_only, B_only, common), &result, std::tuple_cat(A_only, common), A, std::tuple_cat(B_only, common), B);

    // Return a reconstruction of the result tensor ... this can be considered as a simple reshape of the tensor.
    return Tensor<T, 2>{std::move(result), "KR product", -1, detail::product_dims(A_common_position, A)};
}

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)