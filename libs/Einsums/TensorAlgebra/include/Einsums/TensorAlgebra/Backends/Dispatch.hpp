//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#ifndef DOXYGEN

#    include <Einsums/Config.hpp>

#    include <Einsums/Concepts/SubscriptChooser.hpp>
#    include <Einsums/Concepts/TensorConcepts.hpp>
#    include <Einsums/Errors/Error.hpp>
#    include <Einsums/LinearAlgebra.hpp>
#    include <Einsums/Logging.hpp>
#    include <Einsums/Print.hpp>
#    include <Einsums/Tensor/BlockTensor.hpp>
#    include <Einsums/Tensor/Tensor.hpp>
#    include <Einsums/TensorAlgebra/Backends/BaseAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Backends/BlockAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Backends/BlockTileAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Backends/GenericAlgorithm.hpp>
#    include <Einsums/TensorAlgebra/Backends/TileAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#    include <Einsums/TensorBase/Common.hpp>
#    if defined(EINSUMS_COMPUTE_CODE)
#        include <Einsums/TensorAlgebra/Backends/GPUTensorAlgebra.hpp>
#    endif
#    include <Einsums/Profile.hpp>

#    include <algorithm>
#    include <cmath>
#    include <cstddef>
#    include <memory>
#    include <stdexcept>
#    include <string>
#    include <tuple>
#    include <type_traits>
#    include <utility>

#    if defined(EINSUMS_USE_CATCH2)
#        include <catch2/catch_all.hpp>
#    endif

namespace einsums::tensor_algebra {
namespace detail {

template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
AlgorithmChoice einsum(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                       BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                       std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B);

/**
 * @brief Perform runtime checks for the tensor dimensions.
 */
template <bool OnlyUseGenericAlgorithm, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
void einsum_runtime_check(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                          BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                          std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                          BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;
    bool             runtime_indices_abort{false};

    for_sequence<ARank>([&](auto a) {
        size_t dimA = A.dim(a);
        for_sequence<BRank>([&](auto b) {
            size_t dimB = B.dim(b);
            if (std::get<a>(A_indices).letter == std::get<b>(B_indices).letter) {
                if (dimA != dimB) {
#    if !defined(EINSUMS_IS_TESTING)
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));
#    endif
                    runtime_indices_abort = true;
                }
            }
        });
        for_sequence<CRank>([&](auto c) {
            size_t dimC;
            if constexpr (IsTensorV<CType>) {
                dimC = C->dim(c);
            } else {
                dimC = 0;
            }
            if (std::get<a>(A_indices).letter == std::get<c>(C_indices).letter) {
                if (dimA != dimC) {
#    if !defined(EINSUMS_IS_TESTING)
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));
#    endif
                    runtime_indices_abort = true;
                }
            }
        });
    });
    for_sequence<BRank>([&](auto b) {
        size_t dimB = B.dim(b);
        for_sequence<CRank>([&](auto c) {
            size_t dimC;
            if constexpr (IsTensorV<CType>) {
                dimC = C->dim(c);
            } else {
                dimC = 0;
            }
            if (std::get<b>(B_indices).letter == std::get<c>(C_indices).letter) {
                if (dimB != dimC) {
#    if !defined(EINSUMS_IS_TESTING)
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));
#    endif
                    runtime_indices_abort = true;
                }
            }
        });
    });

    if (runtime_indices_abort) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Inconsistent dimensions found!");
    }
}

/**
 * @brief Perform the generic fallback algorithm.
 *
 * This will either call einsum_generic_algorithm or einsum_special_dispatch, depending on whether the tensors
 * have special dispatching. The template argument will be passed onto einsum_special_dispatch.
 */
template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
AlgorithmChoice einsum_generic_default(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                                       BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                                       std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                                       BType const &B) {
    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();

    if constexpr (IsAlgebraTensorV<AType> && IsAlgebraTensorV<BType> && (IsAlgebraTensorV<CType> || !IsTensorV<CType>) &&
                  (!IsBasicTensorV<AType> || !IsBasicTensorV<BType> || (!IsBasicTensorV<CType> && IsTensorV<CType>))) {
        if constexpr (!DryRun) {
            einsum_special_dispatch<OnlyUseGenericAlgorithm, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices,
                                                                           B);
        }
        Tensor<typename AType::ValueType, AType::Rank> dry_A;
        Tensor<typename BType::ValueType, BType::Rank> dry_B;
        if constexpr (TensorConcept<CType>) {
            Tensor<typename CType::ValueType, CType::Rank> dry_C;
            return einsum<OnlyUseGenericAlgorithm, true, ConjA, ConjB>(C_prefactor, C_indices, &dry_C, AB_prefactor, A_indices, dry_A,
                                                                       B_indices, dry_B);
        } else {
            return einsum<OnlyUseGenericAlgorithm, true, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, dry_A, B_indices,
                                                                       dry_B);
        }
    } else {
        constexpr auto A_unique              = UniqueT<std::tuple<AIndices...>>();
        constexpr auto B_unique              = UniqueT<std::tuple<BIndices...>>();
        constexpr auto C_unique              = UniqueT<std::tuple<CIndices...>>();
        constexpr auto linksAB               = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
        constexpr auto links                 = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
        constexpr auto link_unique           = CUniqueT<decltype(links)>();
        constexpr auto link_position_in_A    = detail::find_type_with_position(link_unique, A_indices);
        constexpr auto link_position_in_link = detail::find_type_with_position(link_unique, links);
        constexpr auto target_position_in_C  = detail::find_type_with_position(C_unique, C_indices);
        auto           unique_target_dims    = detail::get_dim_for(*C, detail::unique_find_type_with_position(C_unique, C_indices));
        auto           unique_link_dims      = detail::get_dim_for(A, link_position_in_A);

        EINSUMS_LOG_TRACE("Performing the generic algorithm.");

        if constexpr (!DryRun) {
            einsum_generic_algorithm<ConjA, ConjB>(C_unique, A_unique, B_unique, link_unique, C_indices, A_indices, B_indices,
                                                   unique_target_dims, unique_link_dims, target_position_in_C, link_position_in_link,
                                                   C_prefactor, C, AB_prefactor, A, B);
        }
        return GENERIC;
    }
}

/**
 * @brief Check to see if the index pack is Hadamard.
 *
 * This checks to see if there are duplicate indices within the pack.
 */
template <typename... Indices>
constexpr bool einsum_is_hadamard_found() {
    constexpr auto unique = UniqueT<std::tuple<Indices...>>();
    return std::tuple_size_v<std::tuple<Indices...>> != std::tuple_size_v<decltype(unique)>;
}

/**
 * @brief Check to see if any of the index packs are Hadamard.
 */
template <typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_all_hadamard_found(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &,
                                            std::tuple<BIndices...> const &) {
    return einsum_is_hadamard_found<CIndices...>() || einsum_is_hadamard_found<AIndices...>() || einsum_is_hadamard_found<BIndices...>();
}

/**
 * @brief Checks to see if the indices passed can be turned into a dot product.
 */
template <typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_dot_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {

    constexpr auto A_exactly_matches_B = same_indices<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    return sizeof...(CIndices) == 0 && A_exactly_matches_B;
}

/**
 * @brief Checks to see if the indices passed can be turned into a direct product.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_direct_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_unique = UniqueT<std::tuple<AIndices...>>();
    constexpr auto B_unique = UniqueT<std::tuple<BIndices...>>();
    constexpr auto C_unique = UniqueT<std::tuple<CIndices...>>();
    constexpr auto C_exactly_matches_A =
        sizeof...(CIndices) == sizeof...(AIndices) && same_indices<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto C_exactly_matches_B =
        sizeof...(CIndices) == sizeof...(BIndices) && same_indices<std::tuple<CIndices...>, std::tuple<BIndices...>>();

    return C_exactly_matches_A && C_exactly_matches_B && !ConjA && !ConjB;
}

/**
 * @brief Checks to see if the indices passed can be turned into an outer product.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_outer_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices                       = std::tuple<AIndices...>();
    constexpr auto B_indices                       = std::tuple<BIndices...>();
    constexpr auto C_indices                       = std::tuple<CIndices...>();
    constexpr auto linksAB                         = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto C_unique                        = UniqueT<std::tuple<CIndices...>>();
    constexpr auto target_position_in_A            = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B            = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto contiguous_target_position_in_A = detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B = detail::contiguous_positions(target_position_in_B);
    constexpr auto A_target_position_in_C          = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C          = detail::find_type_with_position(B_indices, C_indices);

    constexpr bool condition =
        std::tuple_size_v<decltype(linksAB)> == 0 && contiguous_target_position_in_A && contiguous_target_position_in_B;

    if constexpr (condition) {
        constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;

        constexpr bool straight_conjugation = !ConjA && !swap_AB;
        constexpr bool swapped_conjugation  = !ConjB && swap_AB;
        return straight_conjugation || swapped_conjugation;
    } else {
        return false;
    }
}

/**
 * @brief Sets up tensor views and performs the outer product on them.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
bool einsum_do_outer_product(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                             BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices              = std::tuple<AIndices...>();
    constexpr auto B_indices              = std::tuple<BIndices...>();
    constexpr auto C_indices              = std::tuple<CIndices...>();
    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = detail::find_type_with_position(B_indices, C_indices);

    EINSUMS_LOG_TRACE("outer_product");
    if (!A.full_view_of_underlying() || !B.full_view_of_underlying()) {
        EINSUMS_LOG_TRACE("do not have full view of underlying data A {} B{}", !A.full_view_of_underlying(), !B.full_view_of_underlying());
        return false;
    }

    if constexpr (DryRun) {
        return true;
    }

    constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;
    EINSUMS_LOG_TRACE("swap_AB {}", swap_AB);

    Dim<2> dC;
    dC[0] = product_dims(A_target_position_in_C, *C);
    dC[1] = product_dims(B_target_position_in_C, *C);
    if constexpr (swap_AB)
        std::swap(dC[0], dC[1]);

#    ifdef EINSUMS_COMPUTE_CODE
    std::conditional_t<IsIncoreRankTensorV<CType, CRank, CDataType>, TensorView<CDataType, 2>, DeviceTensorView<CDataType, 2>> tC{*C, dC};
#    else
    TensorView<CDataType, 2> tC{*C, dC};
#    endif
    if (C_prefactor != CDataType{1.0}) {
        EINSUMS_LOG_TRACE("scaling C");
        linear_algebra::scale(C_prefactor, C);
    }
    try {
        EINSUMS_LOG_TRACE("calling ger");
        if constexpr (swap_AB) {
            if constexpr (ConjA) {
                linear_algebra::gerc(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
            } else {
                linear_algebra::ger(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
            }
        } else {
            if constexpr (ConjB) {
                linear_algebra::gerc(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
            } else {
                linear_algebra::ger(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
            }
        }
    } catch (std::runtime_error &e) {
#    if defined(EINSUMS_SHOW_WARNING)
        EINSUMS_LOG_WARN("Optimized outer product failed. Likely from a non-contiguous "
                         "TensorView. Attempting to perform generic algorithm.");
#    endif
        if constexpr (IsComplexV<CDataType>) {
            if (C_prefactor == CDataType{0.0, 0.0}) {
#    if defined(EINSUMS_SHOW_WARNING)
                EINSUMS_LOG_WARN("WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor,
                                 C->name());
#    endif
            } else {
                linear_algebra::scale(CDataType{1.0, 0.0} / C_prefactor, C);
            }
        } else {
            if (C_prefactor == CDataType{0.0}) {
#    if defined(EINSUMS_SHOW_WARNING)
                EINSUMS_LOG_WARN("WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor,
                                 C->name());
#    endif
            } else {
                linear_algebra::scale(CDataType{1.0} / C_prefactor, C);
            }
        }
        return false;
    }
    return true;
}

/**
 * @brief Checks to see if the indices passed can be turned into a matrix-vector product where the second pack contains the indices for the
 * matrix.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_matrix_vector(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices                           = std::tuple<AIndices...>();
    constexpr auto B_indices                           = std::tuple<BIndices...>();
    constexpr auto C_indices                           = std::tuple<CIndices...>();
    constexpr auto C_unique                            = UniqueT<std::tuple<CIndices...>>();
    constexpr auto linksAB                             = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                               = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto link_unique                         = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A                  = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B                  = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A                = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B                = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto A_target_position_in_C              = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C              = detail::find_type_with_position(B_indices, C_indices);
    constexpr auto contiguous_link_position_in_A       = detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B       = detail::contiguous_positions(link_position_in_B);
    constexpr auto contiguous_target_position_in_A     = detail::contiguous_positions(target_position_in_A);
    constexpr auto same_ordering_link_position_in_AB   = detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA = detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB = detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr bool condition = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                               same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                               !same_ordering_target_position_in_CB && std::tuple_size_v<decltype(B_target_position_in_C)> == 0 && !ConjB;

    if constexpr (condition) {
        constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;
        return transpose_A || !ConjA;
    } else {
        return false;
    }
}

/**
 * @brief Sets up tensor views and performs a matrix-vector product on them.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
bool einsum_do_matrix_vector(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                             BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices              = std::tuple<AIndices...>();
    constexpr auto B_indices              = std::tuple<BIndices...>();
    constexpr auto C_indices              = std::tuple<CIndices...>();
    constexpr auto C_unique               = UniqueT<std::tuple<CIndices...>>();
    constexpr auto linksAB                = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                  = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto link_unique            = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A     = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B     = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A   = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);

    if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
        // Fall through to generic algorithm.
        EINSUMS_LOG_TRACE("do not have full view of underlying data A {} B{} C{}", !A.full_view_of_underlying(),
                          !B.full_view_of_underlying(), !C->full_view_of_underlying());
        return false;
    }

    if constexpr (DryRun) {
        return true;
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

#    ifdef EINSUMS_COMPUTE_CODE
    std::conditional_t<IsIncoreTensorV<AType>, TensorView<ADataType, 2> const, DeviceTensorView<ADataType, 2> const> tA{
        const_cast<AType &>(A), dA, sA};
    std::conditional_t<IsIncoreTensorV<BType>, TensorView<BDataType, 1> const, DeviceTensorView<BDataType, 1> const> tB{
        const_cast<BType &>(B), dB, sB};
    std::conditional_t<IsIncoreTensorV<CType>, TensorView<CDataType, 1>, DeviceTensorView<CDataType, 1>> tC{*C, dC, sC};
#    else
    TensorView<ADataType, 2> const tA{const_cast<AType &>(A), dA, sA};
    TensorView<BDataType, 1> const tB{const_cast<BType &>(B), dB, sB};
    TensorView<CDataType, 1>       tC{*C, dC, sC};
#    endif

    if constexpr (transpose_A) {
        linear_algebra::gemv((ConjA) ? 'c' : 't', AB_prefactor, tA, tB, C_prefactor, &tC);
    } else {
        linear_algebra::gemv<false>(AB_prefactor, tA, tB, C_prefactor, &tC);
    }

    return true;
}

/**
 * @brief Checks to see if the indices passed can be turned into a matrix-matrix product.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_matrix_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices                           = std::tuple<AIndices...>();
    constexpr auto B_indices                           = std::tuple<BIndices...>();
    constexpr auto C_indices                           = std::tuple<CIndices...>();
    constexpr auto CminusA                             = DifferenceT<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto CminusB                             = DifferenceT<std::tuple<CIndices...>, std::tuple<BIndices...>>();
    constexpr bool have_remaining_indices_in_CminusA   = std::tuple_size_v<decltype(CminusA)>;
    constexpr bool have_remaining_indices_in_CminusB   = std::tuple_size_v<decltype(CminusB)>;
    constexpr auto linksAB                             = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                               = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto C_unique                            = UniqueT<std::tuple<CIndices...>>();
    constexpr auto link_unique                         = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A                  = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B                  = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A                = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B                = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto A_target_position_in_C              = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C              = detail::find_type_with_position(B_indices, C_indices);
    constexpr auto contiguous_link_position_in_A       = detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B       = detail::contiguous_positions(link_position_in_B);
    constexpr auto contiguous_target_position_in_A     = detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B     = detail::contiguous_positions(target_position_in_B);
    constexpr auto contiguous_A_targets_in_C           = detail::contiguous_positions(A_target_position_in_C);
    constexpr auto contiguous_B_targets_in_C           = detail::contiguous_positions(B_target_position_in_C);
    constexpr auto same_ordering_link_position_in_AB   = detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA = detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB = detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr bool condition = have_remaining_indices_in_CminusA && have_remaining_indices_in_CminusB && contiguous_link_position_in_A &&
                               contiguous_link_position_in_B && contiguous_target_position_in_A && contiguous_target_position_in_B &&
                               contiguous_A_targets_in_C && contiguous_B_targets_in_C && same_ordering_link_position_in_AB &&
                               same_ordering_target_position_in_CA && same_ordering_target_position_in_CB;

    if constexpr (condition) {
        constexpr bool transpose_A     = std::get<1>(link_position_in_A) == 0;
        constexpr bool transpose_B     = std::get<1>(link_position_in_B) != 0;
        constexpr bool transpose_C     = std::get<1>(A_target_position_in_C) != 0;
        constexpr bool conjugate_works = (transpose_C && (transpose_A || !ConjA) && (transpose_B || !ConjB)) ||
                                         (!transpose_C && (!transpose_A || !ConjA) && (!transpose_B || !ConjB));
        return conjugate_works;
    } else {
        return false;
    }
}

/**
 * @brief Sets up tensor views and performs a matrix-vector product on them.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
bool einsum_do_matrix_product(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                              BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                              std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                              BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices              = std::tuple<AIndices...>();
    constexpr auto B_indices              = std::tuple<BIndices...>();
    constexpr auto C_indices              = std::tuple<CIndices...>();
    constexpr auto linksAB                = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                  = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto C_unique               = UniqueT<std::tuple<CIndices...>>();
    constexpr auto link_unique            = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A     = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B     = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A   = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B   = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = detail::find_type_with_position(B_indices, C_indices);

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

#    ifdef EINSUMS_COMPUTE_CODE
    std::conditional_t<IsIncoreRankTensorV<AType, ARank, ADataType>, TensorView<ADataType, 2> const, DeviceTensorView<ADataType, 2> const>
        tA{const_cast<AType &>(A), dA, sA};
    std::conditional_t<IsIncoreRankTensorV<BType, BRank, BDataType>, TensorView<BDataType, 2> const, DeviceTensorView<BDataType, 2> const>
        tB{const_cast<BType &>(B), dB, sB};
    std::conditional_t<IsIncoreRankTensorV<CType, CRank, CDataType>, TensorView<CDataType, 2>, DeviceTensorView<CDataType, 2>> tC{*C, dC,
                                                                                                                                  sC};
#    else
    TensorView<ADataType, 2> const tA{const_cast<AType &>(A), dA, sA};
    TensorView<BDataType, 2> const tB{const_cast<BType &>(B), dB, sB};
    TensorView<CDataType, 2>       tC{*C, dC, sC};
#    endif
    if constexpr (CoreTensorConcept<decltype(tA)>) {
        if (!tA.impl().is_gemmable() || !tB.impl().is_gemmable() || !tC.impl().is_gemmable()) {
            return false;
        }
    } else {
        if (tA.stride(1) != 1 || tB.stride(1) != 1 || tC.stride(1) != 1) {
            return false;
        }
    }

    if constexpr (DryRun) {
        return true;
    }

    if constexpr (!transpose_C) {
        constexpr char transA = (transpose_A) ? ((ConjA) ? 'c' : 't') : 'n', transB = (transpose_B) ? ((ConjB) ? 'c' : 't') : 'n';
        linear_algebra::gemm(transA, transB, AB_prefactor, tA, tB, C_prefactor, &tC);
    } else {
        constexpr char transA = (!transpose_A) ? ((ConjA) ? 'c' : 't') : 'n', transB = (!transpose_B) ? ((ConjB) ? 'c' : 't') : 'n';
        linear_algebra::gemm(transB, transA, AB_prefactor, tB, tA, C_prefactor, &tC);
    }

    return true;
}

/**
 * @brief Checks to see if there are indices that appear in all three index packs.
 *
 * The indices that appear in all three index packs can be used to batch einsum calls.
 */
template <typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_batchable(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();
    constexpr auto A_unique  = UniqueT<std::tuple<AIndices...>>();
    constexpr auto B_unique  = UniqueT<std::tuple<BIndices...>>();
    constexpr auto C_unique  = UniqueT<std::tuple<CIndices...>>();
    constexpr auto linksAB   = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto batches   = IntersectT<std::tuple<CIndices...>, decltype(linksAB)>();

    return std::tuple_size_v<decltype(batches)> > 0;
}

// CType has typename to allow for interoperability with scalar types.
template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
auto einsum(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
            BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor, std::tuple<AIndices...> const & /*As*/,
            AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B) -> AlgorithmChoice {
    // profile::Timer const _timer();
    print::Indent const _indent;

    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();

    // Ensure the ranks are correct. (Compile-time check.)
    static_assert(sizeof...(CIndices) == CRank, "Rank of C does not match Indices given for C.");
    static_assert(sizeof...(AIndices) == ARank, "Rank of A does not match Indices given for A.");
    static_assert(sizeof...(BIndices) == BRank, "Rank of B does not match Indices given for B.");

    // Runtime check of sizes
#    if defined(EINSUMS_RUNTIME_INDICES_CHECK)
    if constepxr (!DryRun) {
        einsum_runtime_check(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
    }
#    endif

    bool            has_performed_contraction = false;
    AlgorithmChoice retval                    = INDETERMINATE;

    if constexpr (OnlyUseGenericAlgorithm) {
        // Skip to the generic algorithm.
    } else if constexpr (einsum_is_all_hadamard_found(C_indices, A_indices, B_indices) || !std::is_same_v<CDataType, ADataType> ||
                         !std::is_same_v<CDataType, BDataType> ||
                         (!IsAlgebraTensorV<AType> || !IsAlgebraTensorV<BType> || (!IsAlgebraTensorV<CType> && !IsScalarV<CType>))) {
        // Mixed datatypes and poorly behaved tensor types go directly to the generic algorithm.
    } else if constexpr (einsum_is_dot_product(C_indices, A_indices, B_indices)) {
        if constexpr (!DryRun) {
            if constexpr (ConjA == ConjB || (!IsComplexV<ADataType> && !IsComplex<BDataType>)) {
                CDataType temp = linear_algebra::dot(A, B);
                if constexpr (ConjA && IsComplexV<CDataType>) {
                    temp = std::conj(temp);
                }
                (*C) *= C_prefactor;
                (*C) += AB_prefactor * temp;
            } else {
                CDataType temp = linear_algebra::true_dot(A, B);
                if constexpr (ConjB && IsComplexV<CDataType>) {
                    temp = std::conj(temp);
                }
                (*C) *= C_prefactor;
                (*C) += AB_prefactor * temp;
            }
        }

        has_performed_contraction = true;
        retval                    = DOT;
    } else if constexpr (einsum_is_direct_product<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
        if constexpr (!DryRun) {
            LabeledSection("element-wise multiplication");

            linear_algebra::direct_product(AB_prefactor, A, B, C_prefactor, C);
        }

        has_performed_contraction = true;
        retval                    = DIRECT;
    } else if constexpr (!IsBasicTensorV<AType> || !IsBasicTensorV<BType> || !IsBasicTensorV<CType>) {
        retval = einsum_generic_default<false, DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
        has_performed_contraction = true;
    } else if constexpr (einsum_is_outer_product<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
        has_performed_contraction =
            einsum_do_outer_product<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
        retval = GER;
    } else if constexpr (einsum_is_matrix_vector<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
        has_performed_contraction =
            einsum_do_matrix_vector<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
        retval = GEMV;
    } else if constexpr (einsum_is_matrix_vector<ConjA, ConjB>(C_indices, B_indices, A_indices)) {
        has_performed_contraction =
            einsum_do_matrix_vector<DryRun, ConjB, ConjA>(C_prefactor, C_indices, C, AB_prefactor, B_indices, B, A_indices, A);
        retval = GEMV;
    }
    // To use a gemm the input tensors need to be at least rank 2
    else if constexpr (CRank >= 2 && ARank >= 2 && BRank >= 2) {
        if constexpr (einsum_is_matrix_product<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
            has_performed_contraction =
                einsum_do_matrix_product<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
            retval = GEMM;
        }
    }

    if (!has_performed_contraction) {
        return einsum_generic_default<true, DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
    }
    return retval;
}
} // namespace detail

template <bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename U, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires InSamePlace<AType, CType> || !TensorConcept<CType>;
    }
void einsum(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UAB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    using ABDataType = std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    EINSUMS_LOG_TRACE("BEGIN: einsum");
#    if defined(EINSUMS_HAVE_PROFILER)
    std::unique_ptr<profile::ScopedZone> _section;
#    endif
    if constexpr (IsTensorV<CType>) {
        EINSUMS_LOG_INFO(
            std::fabs(UC_prefactor) > EINSUMS_ZERO
                ? fmt::format(R"(einsum: "{}"{} = {} {}"{}"{}{} * {}"{}"{}{} + {} "{}"{})", C->name(), C_indices, UAB_prefactor,
                              (ConjA) ? "conj(" : "", A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "", B.name(), B_indices,
                              (ConjB) ? ")" : "", UC_prefactor, C->name(), C_indices)
                : fmt::format(R"(einsum: "{}"{} = {} {}"{}"{}{} * {}"{}"{}{})", C->name(), C_indices, UAB_prefactor, (ConjA) ? "conj(" : "",
                              A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "", B.name(), B_indices, (ConjB) ? ")" : ""));
#    if defined(EINSUMS_HAVE_PROFILER)
        _section = std::make_unique<profile::ScopedZone>(std::fabs(UC_prefactor) > EINSUMS_ZERO
                                                             ? fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{} + {} "{}"{})", C->name(),
                                                                           C_indices, UAB_prefactor, A.name(), A_indices, B.name(),
                                                                           B_indices, UC_prefactor, C->name(), C_indices)
                                                             : fmt::format(R"(einsums: "{}"{} = {} "{}"{} * "{}"{})", C->name(), C_indices,
                                                                           UAB_prefactor, A.name(), A_indices, B.name(), B_indices),
                                                         __FILE__, __LINE__, __func__);
#    endif
    } else {
        EINSUMS_LOG_INFO(std::fabs(UC_prefactor) > EINSUMS_ZERO
                             ? fmt::format(R"(einsum: "C"{} = {} {}"{}"{}{} * {}"{}"{}{} + {} "C"{})", C_indices, UAB_prefactor,
                                           (ConjA) ? "conj(" : "", A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "",
                                           B.name(), B_indices, (ConjB) ? ")" : "", UC_prefactor, C_indices)
                             : fmt::format(R"(einsum: "C"{} = {} {}"{}"{}{} * {}"{}"{}{})", C_indices, UAB_prefactor,
                                           (ConjA) ? "conj(" : "", A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "",
                                           B.name(), B_indices, (ConjB) ? ")" : ""));
#    if defined(EINSUMS_HAVE_PROFILER)
        _section = std::make_unique<profile::ScopedZone>(
            std::fabs(UC_prefactor) > EINSUMS_ZERO
                ? fmt::format(R"(einsum: "C"{} = {} "{}"{} * "{}"{} + {} "C"{})", C_indices, UAB_prefactor, A.name(), A_indices, B.name(),
                              B_indices, UC_prefactor, C_indices)
                : fmt::format(R"(einsum: "C"{} = {} "{}"{} * "{}"{})", C_indices, UAB_prefactor, A.name(), A_indices, B.name(), B_indices),
            __FILE__, __LINE__, __func__);
#    endif
    }

    CDataType const  C_prefactor  = UC_prefactor;
    ABDataType const AB_prefactor = UAB_prefactor;

#    if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    // Clone C into a new tensor
    auto testC = Tensor<CDataType, CRank>(*C);
    {
        Section t1("testing");
#        ifdef EINSUMS_COMPUTE_CODE
        if constexpr (einsums::detail::IsDeviceTensorV<CType>) {
            auto testA = Tensor<ADataType, ARank>(A);
            auto testB = Tensor<BDataType, BRank>(B);
            // Perform the einsum using only the generic algorithm
            // #pragma omp task depend(in: A, B) depend(inout: testC)
            {
                detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, testB);
            }
        } else {
#        endif
            if constexpr (!einsums::detail::IsBasicTensorV<AType> && !einsums::detail::IsBasicTensorV<BType>) {
                auto testA = Tensor<ADataType, ARank>(A);
                auto testB = Tensor<BDataType, BRank>(B);
                {
                    detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices,
                                                              testB);
                }
            } else if constexpr (!einsums::detail::IsBasicTensorV<AType>) {
                auto testA = Tensor<ADataType, ARank>(A);
                {
                    detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, B);
                }
            } else if constexpr (!einsums::detail::IsBasicTensorV<BType>) {
                auto testB = Tensor<BDataType, BRank>(B);
                {
                    detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, testB);
                }
            } else {
                // Perform the einsum using only the generic algorithm
                // #pragma omp task depend(in: A, B) depend(inout: testC)
                {
                    detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, B);
                }
                // #pragma omp taskwait depend(in: testC)
            }
#        ifdef EINSUMS_COMPUTE_CODE
        }
#        endif
    }
#    endif

    // Default einsums.
    detail::AlgorithmChoice retval =
        detail::einsum<false, false, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);

#    if defined(EINSUMS_TEST_NANS)
    // The tests need a wait.
    // #pragma omp taskwait depend(in: *C, testC)
    if constexpr (CRank != 0) {
        Stride<CRank> index_strides;

        size_t elements = dims_to_strides(C->dims(), index_strides);

        for (size_t item = 0; item < elements; item++) {
            std::array<int64_t, CRank> target_combination;

            sentinel_to_indices(item, index_strides, target_combination);

            CDataType Cvalue{subscript_tensor(*C, target_combination)};
            if constexpr (!IsComplexV<CDataType>) {
                if (std::isnan(Cvalue)) {
                    EINSUMS_LOG_ERROR("NaN DETECTED!");
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "NAN detected in resulting tensor.");
                }

                if (std::isinf(Cvalue)) {
                    EINSUMS_LOG_ERROR("Infinity DETECTED!");
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Infinity detected in resulting tensor.");
                }

                if (std::abs(Cvalue) > 100000000) {
                    EINSUMS_LOG_ERROR("Large value DETECTED!");
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Large value detected in resulting tensor.");
                }
            }
        }
    }
#    endif

#    if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    if constexpr (CRank != 0) {
        // Need to walk through the entire C and testC comparing values and reporting differences.
        bool print_info_and_abort{false};

        Stride<CRank> index_strides;

        size_t elements = dims_to_strides(C->dims(), index_strides);

        for (size_t item = 0; item < elements; item++) {
            std::array<int64_t, CRank> target_combination;

            sentinel_to_indices(item, index_strides, target_combination);

            CDataType Cvalue{subscript_tensor(*C, target_combination)};
            CDataType Ctest{subscript_tensor(testC, target_combination)};

            if constexpr (!IsComplexV<CDataType>) {
                if (std::isnan(Cvalue) || std::isnan(Ctest)) {
                    EINSUMS_LOG_ERROR("NAN DETECTED!");
                    println("Source tensors");
                    println(A);
                    println(B);
                    if (std::isnan(Cvalue)) {
                        EINSUMS_LOG_ERROR("NAN detected in C");
                        EINSUMS_LOG_ERROR("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(*C);
                    }
                    if (std::isnan(Ctest)) {
                        EINSUMS_LOG_ERROR("NAN detected in reference Ctest");
                        EINSUMS_LOG_ERROR("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(testC);
                    }

                    print_info_and_abort = true;
                }
            }

#        if defined(EINSUMS_USE_CATCH2)
            if constexpr (!IsComplexV<CDataType>) {
                REQUIRE_THAT(Cvalue,
                             Catch::Matchers::WithinRel(Ctest, static_cast<CDataType>(0.001)) || Catch::Matchers::WithinAbs(0, 0.0001));
                CHECK(print_info_and_abort == false);
            }
#        endif

            if (std::abs(Cvalue - Ctest) > 1.0E-6) {
                print_info_and_abort = true;
            }

            if (print_info_and_abort) {
                EINSUMS_LOG_ERROR(emphasis::bold, "!!! EINSUM ERROR !!!");
                if constexpr (IsComplexV<CDataType>) {
                    EINSUMS_LOG_ERROR("Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                    EINSUMS_LOG_ERROR("Obtained {:20.14f} + {:20.14f}i", Cvalue.real(), Cvalue.imag());
                } else {
                    EINSUMS_LOG_ERROR("Expected {:20.14f}", Ctest);
                    EINSUMS_LOG_ERROR("Obtained {:20.14f}", Cvalue);
                }
                EINSUMS_LOG_ERROR("tensor element ({:})", print_tuple_no_type(target_combination));
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
                EINSUMS_LOG_ERROR("{} C({:}) += {:f} A({:}) * B({:})", C_prefactor_string, print_tuple_no_type(C_indices),
                                  AB_prefactor_string, print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

                println("Expected:");
                println(testC);
                println("Obtained");
                println(*C);
                println(A);
                println(B);
#        if defined(EINSUMS_TEST_EINSUM_ABORT)
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Continuous test failed!");
#        endif
            }
        }
    } else {
        CDataType const Cvalue = static_cast<CDataType const>(*C);
        CDataType const Ctest  = static_cast<CDataType const>(testC);

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

#        if defined(EINSUMS_TEST_EINSUM_ABORT)
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Continuous test failed!");
#        endif
        }
    }
#    endif
    EINSUMS_LOG_TRACE("END: einsum");

    if (algorithm_choice != nullptr) {
        *algorithm_choice = retval;
    }
}

template <bool ConjA, bool ConjB, Container CType, Container AType, Container BType, typename CPrefactorType, typename ABPrefactorType,
          typename... AIndices, typename... BIndices, typename... CIndices>
void einsum(CPrefactorType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C_list, ABPrefactorType const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A_list, std::tuple<BIndices...> const &B_indices, BType const &B_list,
            detail::AlgorithmChoice *algorithm_choice) {
    if (C_list->size() != A_list.size() || C_list->size() != B_list.size()) {
        EINSUMS_THROW_EXCEPTION(bad_logic, "Lists passed to batched einsum call do not have the same size!");
    }

    if (C_list->size() == 0) {
        return;
    }

    size_t tensors = C_list->size();

    *algorithm_choice = detail::einsum<false, true, ConjA, ConjB>(C_prefactor, C_indices, &(C_list->at(0)), AB_prefactor, A_indices,
                                                                  A_list.at(0), B_indices, B_list.at(0));

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < tensors; i++) {
        einsum<ConjA, ConjB>(C_prefactor, C_indices, &(C_list->at(i)), AB_prefactor, A_indices, A_list.at(i), B_indices, B_list.at(i));
    }
}

} // namespace einsums::tensor_algebra

#endif
