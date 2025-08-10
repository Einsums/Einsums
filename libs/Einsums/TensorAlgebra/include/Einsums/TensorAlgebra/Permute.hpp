//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/SmartPointer.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/StringUtil/StringOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorBase/Common.hpp>

#include "Einsums/TensorAlgebra/Detail/Index.hpp"

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/TensorAlgebra/Backends/DevicePermute.hpp>
#endif

namespace einsums::tensor_algebra {

#if !defined(EINSUMS_WINDOWS)
namespace detail {

std::shared_ptr<hptt::Transpose<float>> EINSUMS_EXPORT  permute(int const *perm, int const dim, float const alpha, float const *A,
                                                                size_t const *sizeA, float const beta, float *B, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<double>> EINSUMS_EXPORT permute(int const *perm, int const dim, double const alpha, double const *A,
                                                                size_t const *sizeA, double const beta, double *B, bool conjA,
                                                                bool row_major);
std::shared_ptr<hptt::Transpose<std::complex<float>>> EINSUMS_EXPORT permute(int const *perm, int const dim,
                                                                             std::complex<float> const alpha, std::complex<float> const *A,
                                                                             size_t const *sizeA, std::complex<float> const beta,
                                                                             std::complex<float> *B, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<std::complex<double>>>
    EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A,
                           size_t const *sizeA, std::complex<double> const beta, std::complex<double> *B, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<float>> EINSUMS_EXPORT  permute(int const *perm, int const dim, float const alpha, float const *A,
                                                                size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA,
                                                                float const beta, float *B, size_t const *offsetB, size_t const *outerSizeB,
                                                                bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<double>> EINSUMS_EXPORT permute(int const *perm, int const dim, double const alpha, double const *A,
                                                                size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA,
                                                                double const beta, double *B, size_t const *offsetB,
                                                                size_t const *outerSizeB, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<std::complex<float>>>
    EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A,
                           size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA, std::complex<float> const beta,
                           std::complex<float> *B, size_t const *offsetB, size_t const *outerSizeB, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<std::complex<double>>>
    EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A,
                           size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA, std::complex<double> const beta,
                           std::complex<double> *B, size_t const *offsetB, size_t const *outerSizeB, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<float>> EINSUMS_EXPORT  permute(int const *perm, int const dim, float const alpha, float const *A,
                                                                size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA,
                                                                size_t const innerStrideA, float const beta, float *B, size_t const *offsetB,
                                                                size_t const *outerSizeB, size_t const innerStrideB, bool conjA,
                                                                bool row_major);
std::shared_ptr<hptt::Transpose<double>> EINSUMS_EXPORT permute(int const *perm, int const dim, double const alpha, double const *A,
                                                                size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA,
                                                                size_t const innerStrideA, double const beta, double *B,
                                                                size_t const *offsetB, size_t const *outerSizeB, size_t const innerStrideB,
                                                                bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<std::complex<float>>>
    EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A,
                           size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA, size_t const innerStrideA,
                           std::complex<float> const beta, std::complex<float> *B, size_t const *offsetB, size_t const *outerSizeB,
                           size_t const innerStrideB, bool conjA, bool row_major);
std::shared_ptr<hptt::Transpose<std::complex<double>>>
    EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A,
                           size_t const *sizeA, size_t const *offsetA, size_t const *outerSizeA, size_t const innerStrideA,
                           std::complex<double> const beta, std::complex<double> *B, size_t const *offsetB, size_t const *outerSizeB,
                           size_t const innerStrideB, bool conjA, bool row_major);

template <bool ConjA = false, typename T, typename... CIndices, typename... AIndices>
void permute(T beta, std::tuple<CIndices...> const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha,
             std::tuple<AIndices...> const &A_indices, einsums::detail::TensorImpl<T> const &A) {
    constexpr size_t ARank = sizeof...(AIndices);
    constexpr size_t CRank = sizeof...(CIndices);

    // Error check:  If there are any remaining indices then we cannot perform a permute
    constexpr auto check = DifferenceT<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    einsums::for_sequence<ARank>([&](auto n) {
        if (C->dim((size_t)n) < A.dim(std::get<2 * (size_t)n + 1>(target_position_in_A))) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The {} dimension of the output tensor is smaller than the input tensor!",
                                    print::ordinal((size_t)n));
        }

        if (C->dim((size_t)n) == 0) {
            return;
        }
    });

    // Calculate reversed indices.
    using ReverseC                   = ReverseT<CIndices...>;
    constexpr auto reverse_C_indices = ReverseC();

    std::array<int, ARank>    perms{};
    std::array<size_t, ARank> size{};
    std::array<size_t, ARank> outerSizeA{};
    std::array<size_t, ARank> offsetA{};
    std::array<size_t, ARank> outerSizeC{};
    std::array<size_t, ARank> offsetC{};

    if (A.is_row_major() && C->is_row_major()) {
        auto   new_target_position_in_A = detail::find_type_with_position(C_indices, A_indices);
        size_t innerStrideA             = A.stride(ARank - 1);
        size_t innerStrideC             = C->stride(CRank - 1);
        perms[0]                        = arguments::get_from_tuple<size_t>(new_target_position_in_A, 1);
        size[0]                         = A.dim(0);
        outerSizeA[0]                   = A.dim(0);
        offsetA[0]                      = 0;
        outerSizeC[0]                   = C->dim(0);
        offsetC[0]                      = 0;
        for (int i0 = 1; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 - 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto result = detail::permute(perms.data(), ARank, alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C->data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, true);
    } else if (A.is_row_major() && C->is_column_major()) {
        auto   new_target_position_in_A = detail::find_type_with_position(reverse_C_indices, A_indices);
        auto   C_swap                   = C->to_row_major();
        size_t innerStrideA             = A.stride(ARank - 1);
        size_t innerStrideC             = C_swap.stride(CRank - 1);
        perms[0]                        = arguments::get_from_tuple<size_t>(new_target_position_in_A, 1);
        size[0]                         = A.dim(0);
        outerSizeA[0]                   = A.dim(0);
        offsetA[0]                      = 0;
        outerSizeC[0]                   = C_swap.dim(0);
        offsetC[0]                      = 0;
        for (int i0 = 1; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 - 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto result = detail::permute(perms.data(), ARank, alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C_swap.data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, true);
    } else if (A.is_column_major() && C->is_column_major()) {
        auto   new_target_position_in_A = detail::find_type_with_position(C_indices, A_indices);
        size_t innerStrideA             = A.stride(0);
        size_t innerStrideC             = C->stride(0);
        perms[ARank - 1]                = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * ARank) - 1);
        size[ARank - 1]                 = A.dim(-1);
        outerSizeA[ARank - 1]           = A.dim(-1);
        offsetA[ARank - 1]              = 0;
        outerSizeC[ARank - 1]           = C->dim(-1);
        offsetC[ARank - 1]              = 0;
        for (int i0 = 0; i0 < ARank - 1; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 + 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto result = detail::permute(perms.data(), ARank, alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C->data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, false);
    } else {
        auto   new_target_position_in_A = detail::find_type_with_position(reverse_C_indices, A_indices);
        auto   C_swap                   = C->to_column_major();
        size_t innerStrideA             = A.stride(0);
        size_t innerStrideC             = C_swap.stride(0);
        perms[ARank - 1]                = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * ARank) - 1);
        size[ARank - 1]                 = A.dim(-1);
        outerSizeA[ARank - 1]           = A.dim(-1);
        offsetA[ARank - 1]              = 0;
        outerSizeC[ARank - 1]           = C_swap.dim(-1);
        offsetC[ARank - 1]              = 0;
        for (int i0 = 0; i0 < ARank - 1; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 + 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto result = detail::permute(perms.data(), ARank, alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C_swap.data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, false);
    }
}

template <bool ConjA = false, typename T>
void permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha, std::string const &A_indices,
             einsums::detail::TensorImpl<T> const &A) {
    LabeledSection1((std::abs(beta) > EINSUMS_ZERO)
                        ? fmt::format(R"(permute: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), C_indices, alpha, A.name(), A_indices, beta,
                                      C->name(), C_indices)
                        : fmt::format(R"(permute: "{}"{} = {} "{}"{})", C->name(), C_indices, alpha, A.name(), A_indices));

    // Error check:  If there are any remaining indices then we cannot perform a permute
    auto check = difference(A_indices, C_indices);
    if (check.size() != 0) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The number of unique indices needs to be the same on the input and output for permute!");
    }

    // Calculate reversed indices.
    BufferVector<int>    perms(A.rank());
    BufferVector<size_t> size(A.rank());
    BufferVector<size_t> outerSizeA(A.rank());
    BufferVector<size_t> offsetA(A.rank());
    BufferVector<size_t> outerSizeC(A.rank());
    BufferVector<size_t> offsetC(A.rank());

    if (A.is_row_major() && C->is_row_major()) {
        size_t innerStrideA = A.stride(-1);
        size_t innerStrideC = C->stride(-1);
        size[0]             = A.dim(0);
        outerSizeA[0]       = A.dim(0);
        offsetA[0]          = 0;
        outerSizeC[0]       = C->dim(0);
        offsetC[0]          = 0;
        for (int i0 = 1; i0 < A.rank(); i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 - 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }

        find_char_with_position(A_indices, C_indices, &perms);
        auto result = detail::permute(perms.data(), A.rank(), alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C->data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, true);
    } else if (A.is_row_major() && C->is_column_major()) {
        auto   C_swap       = C->to_row_major();
        size_t innerStrideA = A.stride(-1);
        size_t innerStrideC = C_swap.stride(-1);
        size[0]             = A.dim(0);
        outerSizeA[0]       = A.dim(0);
        offsetA[0]          = 0;
        outerSizeC[0]       = C_swap.dim(0);
        offsetC[0]          = 0;
        for (int i0 = 1; i0 < A.rank(); i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 - 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        find_char_with_position(A_indices, reverse(C_indices), &perms);
        auto result = detail::permute(perms.data(), A.rank(), alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C_swap.data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, true);
    } else if (A.is_column_major() && C->is_column_major()) {
        size_t innerStrideA      = A.stride(0);
        size_t innerStrideC      = C->stride(0);
        size[A.rank() - 1]       = A.dim(-1);
        outerSizeA[A.rank() - 1] = A.dim(-1);
        offsetA[A.rank() - 1]    = 0;
        outerSizeC[A.rank() - 1] = C->dim(-1);
        offsetC[A.rank() - 1]    = 0;
        for (int i0 = 0; i0 < A.rank() - 1; i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 + 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }

        find_char_with_position(A_indices, C_indices, &perms);

        auto result = detail::permute(perms.data(), A.rank(), alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C->data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, false);
    } else {
        auto   C_swap            = C->to_column_major();
        size_t innerStrideA      = A.stride(0);
        size_t innerStrideC      = C_swap.stride(0);
        size[A.rank() - 1]       = A.dim(-1);
        outerSizeA[A.rank() - 1] = A.dim(-1);
        offsetA[A.rank() - 1]    = 0;
        outerSizeC[A.rank() - 1] = C_swap.dim(-1);
        offsetC[A.rank() - 1]    = 0;
        for (int i0 = 0; i0 < A.rank() - 1; i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 + 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        find_char_with_position(A_indices, C_indices, &perms);
        auto result = detail::permute(perms.data(), A.rank(), alpha, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                                      beta, C_swap.data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, false);
    }
}

} // namespace detail
#endif

/**
 * @brief Permutes the elements of a tensor and puts it into an output tensor.
 *
 * This function uses HPTT, which can only handle out-of-place tensor transpositions. The tensors passed in should not
 * have overlapping storage.
 *
 * @param UC_prefactor The prefactor for mixing the output tensor.
 * @param C_indices The indices for the output tensor.
 * @param C The output tensor.
 * @param UA_prefactor The prefactor for the input tensor.
 * @param A_indices The indices for the input tensor.
 * @param A The input tensor.
 * @tparam ConjA If true, conjugate the values of A as it is being permuted.
 */
template <bool ConjA = false, CoreTensorConcept AType, CoreTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType const &A) {
    using T                = typename AType::ValueType;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t CRank = CType::Rank;

    std::string description = std::abs(UC_prefactor) > EINSUMS_ZERO
                                  ? fmt::format(R"(permute: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), C_indices, UA_prefactor, A.name(),
                                                A_indices, UC_prefactor, C->name(), C_indices)
                                  : fmt::format(R"(permute: "{}"{} = {} "{}"{})", C->name(), C_indices, UA_prefactor, A.name(), A_indices);
    LabeledSection(fmt::runtime(description));

    T const C_prefactor = UC_prefactor;
    T const A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a permute
    constexpr auto check = DifferenceT<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    einsums::for_sequence<ARank>([&](auto n) {
        if (C->dim((size_t)n) < A.dim(std::get<2 * (size_t)n + 1>(target_position_in_A))) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The {} dimension of the output tensor is smaller than the input tensor!",
                                    print::ordinal((size_t)n));
        }

        if (C->dim((size_t)n) == 0) {
            return;
        }
    });

#if !defined(EINSUMS_WINDOWS)
    if (A.impl().is_row_major() != C->impl().is_row_major()) {
        detail::permute<ConjA>(C_prefactor, C_indices, &C->impl(), A_prefactor, A_indices, A.impl());
        return;
    }
    if constexpr (std::is_same_v<CType, Tensor<T, CRank>> && std::is_same_v<AType, Tensor<T, ARank>>) {
        std::array<int, ARank>    perms{};
        std::array<size_t, ARank> size{};

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0] = arguments::get_from_tuple<size_t>(target_position_in_A, (2 * i0) + 1);
            size[i0]  = A.dim(i0);
        }

        detail::permute(perms.data(), ARank, A_prefactor, A.data(), size.data(), C_prefactor, C->data(), ConjA, A.impl().is_row_major());
    } else if constexpr (std::is_same_v<CType, Tensor<T, CRank>> && std::is_same_v<AType, TensorView<T, ARank>>) {
        std::array<int, ARank>    perms{};
        std::array<size_t, ARank> size{};
        std::array<size_t, ARank> outerSizeA{};
        std::array<size_t, ARank> offsetA{};
        std::array<size_t, ARank> outerSizeC{};
        std::array<size_t, ARank> offsetC{};
        size_t                    innerStrideA = A.impl().get_incx();
        size_t                    innerStrideC = 1;

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.source_dim(i0);
            offsetA[i0]    = A.offset(i0);
            outerSizeC[i0] = 0;
            offsetC[i0]    = 0;
        }

        for (int i0 = 0; i0 < ARank; i0++) {
            outerSizeC[i0] = A.dim(perms[i0]);
        }
        detail::permute(perms.data(), ARank, A_prefactor, A.full_data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                        C_prefactor, C->data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, A.impl().is_row_major());
    } else if constexpr (std::is_same_v<CType, TensorView<T, CRank>> && std::is_same_v<AType, Tensor<T, ARank>>) {
        std::array<int, ARank>    perms{};
        std::array<size_t, ARank> size{};
        std::array<size_t, ARank> outerSizeA{};
        std::array<size_t, ARank> offsetA{};
        std::array<size_t, ARank> outerSizeC{};
        std::array<size_t, ARank> offsetC{};
        size_t                    innerStrideA = 1;
        size_t                    innerStrideC = C->impl().get_incx();

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.dim(i0);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->source_dim(i0);
            offsetC[i0]    = C->offset(i0);
        }
        detail::permute(perms.data(), ARank, A_prefactor, A.data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                        C_prefactor, C->full_data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, A.impl().is_row_major());
    } else if constexpr (std::is_same_v<CType, TensorView<T, CRank>> && std::is_same_v<AType, TensorView<T, ARank>>) {
        std::array<int, ARank>    perms{};
        std::array<size_t, ARank> size{};
        std::array<size_t, ARank> outerSizeA{};
        std::array<size_t, ARank> offsetA{};
        std::array<size_t, ARank> outerSizeC{};
        std::array<size_t, ARank> offsetC{};
        size_t                    innerStrideA = A.impl().get_incx();
        size_t                    innerStrideC = C->impl().get_incx();

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.source_dim(i0);
            offsetA[i0]    = A.offset(i0);
            outerSizeC[i0] = C->source_dim(i0);
            offsetC[i0]    = C->offset(i0);
        }
        detail::permute(perms.data(), ARank, A_prefactor, A.full_data(), size.data(), offsetA.data(), outerSizeA.data(), innerStrideA,
                        C_prefactor, C->full_data(), offsetC.data(), outerSizeC.data(), innerStrideC, ConjA, A.impl().is_row_major());
    } else
#endif
        if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)> && !(ConjA && IsComplexV<T>)) {
        // If the prefactor is zero, set the tensor to zero. This avoids NaNs.
        if (C_prefactor == T{0.0}) {
            *C = T{0.0};
        }
        linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        // If the prefactor is zero, set the tensor to zero. This avoids NaNs.
        if (C_prefactor == T{0.0}) {
            *C = T{0.0};
        }
        Stride<ARank> index_strides;
        size_t        elements = dims_to_strides(A.dims(), index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < elements; i++) {
            thread_local std::array<int64_t, ARank> index;

            sentinel_to_indices(i, index_strides, index);

            auto A_order = detail::construct_indices(A_indices, index, target_position_in_A, index, target_position_in_A);

            T &target_value = subscript_tensor(*C, index);
            T  A_value      = subscript_tensor(A, A_order);

            if constexpr (ConjA && IsComplexV<T>) {
                target_value = C_prefactor * target_value + A_prefactor * std::conj(A_value);
            } else {
                target_value = C_prefactor * target_value + A_prefactor * A_value;
            }
        }
    }
}

// Sort with default values, no smart pointers
/**
 * @brief Permutes the elements of a tensor and puts it into an output tensor.
 *
 * This function uses HPTT, which can only handle out-of-place tensor transpositions. The tensors passed in should not
 * have overlapping storage.
 *
 * @param C_indices The indices for the output tensor.
 * @param C The output tensor.
 * @param A_indices The indices for the input tensor.
 * @param A The input tensor.
 * @tparam ConjA If true, conjugate the values of A as it is being permuted.
 */
template <bool ConjA = false, NotASmartPointer ObjectA, NotASmartPointer ObjectC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, ObjectC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    permute<ConjA>(0, C_indices, C, 1, A_indices, A);
}

#ifndef DOXYGEN
// Sort with default values, two smart pointers
template <bool ConjA = false, SmartPointer SmartPointerA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    permute<ConjA>(0, C_indices, C->get(), 1, A_indices, *A);
}

// Sort with default values, one smart pointer (A)
template <bool ConjA = false, SmartPointer SmartPointerA, NotASmartPointer PointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, PointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    permute<ConjA>(0, C_indices, C, 1, A_indices, *A);
}

// Sort with default values, one smart pointer (C)
template <bool ConjA = false, NotASmartPointer ObjectA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    permute<ConjA>(0, C_indices, C->get(), 1, A_indices, A);
}

template <bool ConjA = false, BlockTensorConcept AType, BlockTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires InSamePlace<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType const &A) {

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        auto       &C_block = (*C)[i];
        auto const &A_block = A[i];
        permute<ConjA>(UC_prefactor, C_indices, &C_block, UA_prefactor, A_indices, A_block);
    }
}
#endif

/**
 * @brief Finds the tile grid dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, TensorConcept AType, TensorConcept BType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename... AllUniqueIndices>
inline auto get_grid_ranges_for_many(CType const &C, std::tuple<CIndices...> const &C_indices, AType const &A,
                                     std::tuple<AIndices...> const &A_indices, std::tuple<AllUniqueIndices...> const &All_unique_indices) {
    return std::array{get_grid_ranges_for_many_a<AllUniqueIndices, 0>(C, C_indices, A, A_indices)...};
}

#ifndef DOXYGEN
template <bool ConjA = false, TiledTensorConcept AType, TiledTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires InSamePlace<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType const &A) {

    using ADataType        = typename AType::ValueType;
    using CDataType        = typename CType::ValueType;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t CRank = CType::Rank;

    constexpr auto unique_indices = UniqueT<std::tuple<CIndices..., AIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides);

    std::array<int, ARank> A_index_table;
    std::array<int, CRank> C_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        thread_local std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        thread_local std::array<int, ARank> A_tile_index;
        thread_local std::array<int, CRank> C_tile_index;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
        }

        for (int i = 0; i < CRank; i++) {
            C_tile_index[i] = unique_index_table[C_index_table[i]];
        }

        if (!A.has_tile(A_tile_index) || A.has_zero_size(A_tile_index) || C->has_zero_size(C_tile_index)) {
            continue;
        }

        C->lock();
        auto &C_tile = C->tile(C_tile_index);
        C->unlock();
        C_tile.lock();
        permute<ConjA>(UC_prefactor, C_indices, &C_tile, UA_prefactor, A_indices, A.tile(A_tile_index));
        C_tile.unlock();
    }
}
#endif

template <bool ConjA = false, MatrixConcept CType, MatrixConcept AType>
void transpose(CType *C, AType const &A) {
    permute<ConjA>(0.0, einsums::Indices{index::i, index::j}, C, 1.0, einsums::Indices{index::j, index::i}, A);
}

} // namespace einsums::tensor_algebra
