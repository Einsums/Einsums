#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/BlockTensor.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TiledTensor.hpp"
#include "einsums/Timer.hpp"
#include "einsums/tensor_algebra_backends/BlockAlgebra.hpp"
#include "einsums/tensor_algebra_backends/BlockTileAlgebra.hpp"
#include "einsums/tensor_algebra_backends/TileAlgebra.hpp"
#ifdef __HIP__
#    include "einsums/tensor_algebra_backends/GPUTensorAlgebra.hpp"
#endif
#include "einsums/tensor_algebra_backends/GenericAlgorithm.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

namespace einsums::tensor_algebra {
namespace detail {

template <bool OnlyUseGenericAlgorithm, typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
auto einsum(const typename CType::data_type C_prefactor, const std::tuple<CIndices...> & /*Cs*/, CType *C,
            const std::conditional_t<(sizeof(typename AType::data_type) > sizeof(typename BType::data_type)), typename AType::data_type,
                                     typename BType::data_type>
                AB_prefactor,
            const std::tuple<AIndices...> & /*As*/, const AType &A, const std::tuple<BIndices...> & /*Bs*/, const BType &B) -> void {
    print::Indent const _indent;

    using ADataType        = AType::data_type;
    using BDataType        = BType::data_type;
    using CDataType        = CType::data_type;
    constexpr size_t ARank = AType::rank;
    constexpr size_t BRank = BType::rank;
    constexpr size_t CRank = CType::rank;

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
        goto generic_default;
    } else if constexpr (dot_product) {
        CDataType temp = linear_algebra::dot(A, B);
        (*C) *= C_prefactor;
        (*C) += AB_prefactor * temp;

        return;
    } else if constexpr (element_wise_multiplication) {
        timer::Timer const element_wise_multiplication_timer{"element-wise multiplication"};

        linear_algebra::direct_product(AB_prefactor, A, B, C_prefactor, C);

        return;
    } else if constexpr (!einsums::detail::IsBasicTensorV<AType> || !einsums::detail::IsBasicTensorV<BType> ||
                         !einsums::detail::IsBasicTensorV<CType>) {
        goto generic_default;
    } else if constexpr (outer_product) {
        if (!A.full_view_of_underlying() || !B.full_view_of_underlying()) {
            goto generic_default;
        }
        constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;

        Dim<2> dC;
        dC[0] = product_dims(A_target_position_in_C, *C);
        dC[1] = product_dims(B_target_position_in_C, *C);
        if constexpr (swap_AB)
            std::swap(dC[0], dC[1]);

#ifdef __HIP__
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<CType, CRank, CDataType>, TensorView<CDataType, 2>,
                           DeviceTensorView<CDataType, 2>>
            tC{*C, dC};
#else
        TensorView<CDataType, 2> tC{*C, dC};
#endif
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
            println(bg(fmt::color::yellow) | fg(fmt::color::black), "Optimized outer product failed. Likely from a non-contiguous "
                                                                    "TensorView. Attempting to perform generic algorithm.");
#endif
            if (C_prefactor == CDataType{0.0}) {
#if defined(EINSUMS_SHOW_WARNING)
                println(bg(fmt::color::red) | fg(fmt::color::white),
                        "WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor, C->name());
#endif
            } else {
                linear_algebra::scale(1.0 / C_prefactor, C);
            }
            goto generic_default; // Go to the generic algorithm.
        }
        // If we got to this position, assume we successfully called ger.
        return;
    } else if constexpr (!OnlyUseGenericAlgorithm) {
        if constexpr (is_gemv_possible) {
            if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                // Fall through to generic algorithm.
                goto generic_default;
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

#ifdef __HIP__
            std::conditional_t<einsums::detail::IsIncoreRankTensorV<AType, ARank, ADataType>, const TensorView<ADataType, 2>,
                               const DeviceTensorView<ADataType, 2>>
                tA{const_cast<AType<ADataType, ARank> &>(A), dA, sA};
            std::conditional_t<einsums::detail::IsIncoreRankTensorV<BType, BRank, BDataType>, const TensorView<BDataType, 1>,
                               const DeviceTensorView<BDataType, 1>>
                tB{const_cast<BType<BDataType, BRank> &>(B), dB, sB};
            std::conditional_t<einsums::detail::IsIncoreRankTensorV<CType, CRank, CDataType>, TensorView<CDataType, 1>,
                               DeviceTensorView<CDataType, 1>>
                tC{*C, dC, sC};
#else
            const TensorView<ADataType, 2> tA{const_cast<AType &>(A), dA, sA};
            const TensorView<BDataType, 1> tB{const_cast<BType &>(B), dB, sB};
            TensorView<CDataType, 1>       tC{*C, dC, sC};
#endif

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
                    if constexpr (einsums::detail::IsBlockTensorV<AType> && einsums::detail::IsBlockTensorV<BType> &&
                                  einsums::detail::IsBlockTensorV<CType>) {
                        EINSUMS_OMP_PARALLEL_FOR
                        for (int i = 0; i < A.num_blocks(); i++) {
                            if (A.block_dim(i) == 0) {
                                continue;
                            }
                            einsum<false>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, A.block(i), B_indices,
                                          B.block(i));
                        }
                    } else if constexpr (!einsums::detail::IsBasicTensorV<AType> || !einsums::detail::IsBasicTensorV<BType> ||
                                         !einsums::detail::IsBasicTensorV<CType>) {
                        goto generic_default; // Use generic algorithm.
                    } else {

                        if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                            // Fall through to generic algorithm.
                            goto generic_default;
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

#ifdef __HIP__
                        std::conditional_t<einsums::detail::IsIncoreRankTensorV<AType, ARank, ADataType>, const TensorView<ADataType, 2>,
                                           const DeviceTensorView<ADataType, 2>>
                            tA{const_cast<AType &>(A), dA, sA};
                        std::conditional_t<einsums::detail::IsIncoreRankTensorV<BType, BRank, BDataType>, const TensorView<BDataType, 2>,
                                           const DeviceTensorView<BDataType, 2>>
                            tB{const_cast<BType &>(B), dB, sB};
                        std::conditional_t<einsums::detail::IsIncoreRankTensorV<CType, CRank, CDataType>, TensorView<CDataType, 2>,
                                           DeviceTensorView<CDataType, 2>>
                            tC{*C, dC, sC};
#else
                        const TensorView<ADataType, 2> tA{const_cast<AType &>(A), dA, sA};
                        const TensorView<BDataType, 2> tB{const_cast<BType &>(B), dB, sB};
                        TensorView<CDataType, 2>       tC{*C, dC, sC};
#endif

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
                        } else if constexpr (!transpose_C && !transpose_A) {
                            linear_algebra::gemm<false, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && !transpose_B) {
                            linear_algebra::gemm<true, false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C) {
                            linear_algebra::gemm<true, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_A && !transpose_B) {
                            linear_algebra::gemm<true, true>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_A && transpose_B) {
                            linear_algebra::gemm<false, true>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_A && !transpose_B) {
                            linear_algebra::gemm<true, false>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_A && transpose_B) {
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
        }
    }

// Label to jump to when the generic algorithm is needed.
generic_default:;
    // If we somehow make it here, then none of our algorithms above could be used. Attempt to use
    // the generic algorithm instead.
    if constexpr (!einsums::detail::IsBasicTensorV<AType> || !einsums::detail::IsBasicTensorV<BType> ||
                  !einsums::detail::IsBasicTensorV<CType>) {
        einsum_special_dispatch<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
    } else {
        einsum_generic_algorithm(C_unique, A_unique, B_unique, link_unique, C_indices, A_indices, B_indices, unique_target_dims,
                                 unique_link_dims, target_position_in_C, link_position_in_link, C_prefactor, C, AB_prefactor, A, B);
    }
}
} // namespace detail

template <typename AType, typename BType, typename CType, typename U, typename... CIndices, typename... AIndices, typename... BIndices>
    requires InSamePlace<AType, BType, CType>
auto einsum(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const U UAB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) -> void {
    using ADataType        = AType::data_type;
    using BDataType        = BType::data_type;
    using CDataType        = CType::data_type;
    constexpr size_t ARank = AType::rank;
    constexpr size_t BRank = BType::rank;
    constexpr size_t CRank = CType::rank;

    using ABDataType = std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    LabeledSection1((std::fabs(UC_prefactor) > EINSUMS_ZERO)
                        ? fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices),
                                      UAB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices),
                                      UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{})", C->name(), print_tuple_no_type(C_indices), UAB_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices)));

    const CDataType  C_prefactor  = UC_prefactor;
    const ABDataType AB_prefactor = UAB_prefactor;

#if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    // Clone C into a new tensor
    auto testC = Tensor<CDataType, CRank>(*C);
    timer::push("testing");
#    ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceRankTensorV<CType, CRank, CDataType>) {
        auto testA = Tensor<ADataType, ARank>(A);
        auto testB = Tensor<BDataType, BRank>(B);
        // Perform the einsum using only the generic algorithm
        // #pragma omp task depend(in: A, B) depend(inout: testC)
        { detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, testB); }
        // #pragma omp taskwait depend(in: testC)
    } else {
#    endif
        if constexpr (!einsums::detail::IsBasicTensorV<AType> &&
                      !einsums::detail::IsBasicTensorV<BType>) {
            auto testA = Tensor<ADataType, ARank>(A);
            auto testB = Tensor<BDataType, BRank>(B);
            { detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, testB); }
        } else if constexpr (!einsums::detail::IsBasicTensorV<AType>) {
            auto testA = Tensor<ADataType, ARank>(A);
            { detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, B); }
        } else if constexpr (!einsums::detail::IsBasicTensorV<BType>) {
            auto testB = Tensor<BDataType, BRank>(B);
            { detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, testB); }
        } else {
            // Perform the einsum using only the generic algorithm
            // #pragma omp task depend(in: A, B) depend(inout: testC)
            { detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, B); }
            // #pragma omp taskwait depend(in: testC)
        }
#    ifdef __HIP__
    }
#    endif
    timer::pop();
#endif

    // Default einsums.
    detail::einsum<false>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);

#if defined(EINSUMS_TEST_NANS)
    // The tests need a wait.
    // #pragma omp taskwait depend(in: *C, testC)
    if constexpr (CRank != 0) {
        auto target_dims = get_dim_ranges<CRank>(*C);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            CDataType Cvalue{std::apply(*C, target_combination)};
            if constexpr (!IsComplexV<CDataType>) {
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

            if constexpr (!IsComplexV<CDataType>) {
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
            if constexpr (!IsComplexV<CDataType>) {
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
                if constexpr (IsComplexV<CDataType>) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(),
                            Cvalue.imag());
                } else {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
                }
                println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ({:})", print_tuple_no_type(target_combination));
                std::string C_prefactor_string;
                if constexpr (IsComplexV<CDataType>) {
                    C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
                } else {
                    C_prefactor_string = fmt::format("{:f}", C_prefactor);
                }
                std::string AB_prefactor_string;
                if constexpr (IsComplexV<ABDataType>) {
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
        const CDataType Cvalue = static_cast<const CDataType>(*C);
        const CDataType Ctest  = static_cast<const CDataType>(testC);

        // testC could be a Tensor<CDataType, 0> type. Cast it to the underlying data type.
        if (std::abs(Cvalue - (CDataType)testC) > 1.0E-6) {
            println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "!!! EINSUM ERROR !!!");
            if constexpr (IsComplexV<CDataType>) {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(), Cvalue.imag());
            } else {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
            }

            println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ()");
            std::string C_prefactor_string;
            if constexpr (IsComplexV<CDataType>) {
                C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
            } else {
                C_prefactor_string = fmt::format("{:f}", C_prefactor);
            }
            std::string AB_prefactor_string;
            if constexpr (IsComplexV<ABDataType>) {
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
} // namespace einsums::tensor_algebra