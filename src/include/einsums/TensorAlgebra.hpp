//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/STL.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

/*
 * Dispatchers for einsum.
 */
template <template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename U>
    requires requires {
        requires std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>>;
        requires std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>>;
        requires std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>>;
#ifdef __HIP__
        requires(CoreRankTensor<AType<ADataType, ARank>, ARank, ADataType> && CoreRankTensor<BType<BDataType, BRank>, BRank, BDataType> &&
                 CoreRankTensor<CType<CDataType, CRank>, CRank, CDataType>) ||
                    (DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType> &&
                     DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType> &&
                     DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>) ||
                    (DiskRankTensor<AType<ADataType, ARank>, ARank, ARank, ADataType> &&
                     DiskRankTensor<BType<BDataType, BRank>, BRank, BRank, BDataType> &&
                     DiskRankTensor<CType<CDataType, CRank>, CRank, CRank, CDataType>);
#else
        requires(CoreRankTensor<AType<ADataType, ARank>, ARank, ADataType> && CoreRankTensor<BType<BDataType, BRank>, BRank, BDataType> &&
                 CoreRankTensor<CType<CDataType, CRank>, CRank, CDataType>) ||
                    (DiskRankTensor<AType<ADataType, ARank>, ARank, ARank, ADataType> &&
                     DiskRankTensor<BType<BDataType, BRank>, BRank, BRank, BDataType> &&
                     DiskRankTensor<CType<CDataType, CRank>, CRank, CRank, CDataType>);
#endif
    }
auto einsum(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C, const U UAB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B) -> void;

// Einsums with provided prefactors.
// 1. C n A n B n is defined above as the base implementation.

// 2. C n A n B y
template <NotASmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
void einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, *B);
}

// 3. C n A y B n
template <SmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, B);
}

// 4. C n A y B y
template <SmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, *B);
}

// 5. C y A n B n
template <NotASmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, B);
}

// 6. C y A n B y
template <NotASmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, *B);
}

// 7. C y A y B n
template <SmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, B);
}

// 8. C y A y B y
template <SmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, *B);
}

//
// Einsums with default prefactors.
//

// 1. C n A n B n
template <NotASmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, B);
}

// 2. C n A n B y
template <NotASmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, *B);
}

// 3. C n A y B n
template <SmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, B);
}

// 4. C n A y B y
template <SmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, *B);
}

// 5. C y A n B n
template <NotASmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, B);
}

// 6. C y A n B y
template <NotASmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, *B);
}

// 7. C y A y B n
template <SmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, B);
}

// 8. C y A y B y
template <SmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B) {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, *B);
}

//
// Element Transform
//

template <template <typename, size_t> typename CType, size_t CRank, typename UnaryOperator, typename T = double>
    requires std::derived_from<CType<T, CRank>, ::einsums::detail::TensorBase<T, CRank>>
auto element_transform(CType<T, CRank> *C, UnaryOperator unary_opt) -> void;

template <SmartPointer SmartPtr, typename UnaryOperator>
void element_transform(SmartPtr *C, UnaryOperator unary_opt) {
    element_transform(C->get(), unary_opt);
}

template <template <typename, size_t> typename CType, template <typename, size_t> typename... MultiTensors, size_t Rank,
          typename MultiOperator, typename T = double>
auto element(MultiOperator multi_opt, CType<T, Rank> *C, MultiTensors<T, Rank> &...tensors);

template <unsigned int N, typename... List>
constexpr auto get_n(const std::tuple<List...> &);

/**
 * Returns the mode-`mode` unfolding of `tensor` with modes startng at `0`
 *
 * @returns unfolded_tensor of shape ``(tensor.dim(mode), -1)``
 */
template <unsigned int mode, template <typename, size_t> typename CType, size_t CRank, typename T = double>
auto unfold(const CType<T, CRank> &source) -> Tensor<T, 2>
    requires(std::is_same_v<Tensor<T, CRank>, CType<T, CRank>>);

/** Computes the Khatri-Rao product of tensors A and B.
 *
 * Example:
 *    Tensor<2> result = khatri_rao(Indices{I, r}, A, Indices{J, r}, B);
 *
 * Result is described as {(I,J), r}. If multiple common indices are provided they will be collapsed into a single index in the result.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank,
          typename... AIndices, typename... BIndices, typename T = double>
auto khatri_rao(const std::tuple<AIndices...> &, const AType<T, ARank> &A, const std::tuple<BIndices...> &,
                const BType<T, BRank> &B) -> Tensor<T, 2>
    requires(std::is_base_of_v<::einsums::detail::TensorBase<T, ARank>, AType<T, ARank>> &&
             std::is_base_of_v<::einsums::detail::TensorBase<T, BRank>, BType<T, BRank>>);

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

#include "einsums/tensor_algebra_backends/Dispatch.hpp"
#include "einsums/tensor_algebra_backends/ElementTransform.hpp"
#include "einsums/tensor_algebra_backends/KhatriRao.hpp"
#include "einsums/tensor_algebra_backends/Unfold.hpp"