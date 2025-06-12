//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SmartPointer.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "Einsums/Concepts/NamedRequirements.hpp"

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

namespace einsums::tensor_algebra {
namespace detail {

// CType has typename to allow for interoperability with scalar types.
template <bool OnlyUseGenericAlgorithm, bool DryRun, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
AlgorithmChoice einsum(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                       BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                       std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B);
} // namespace detail

/*
 * Dispatchers for einsum.
 */
template <TensorConcept AType, TensorConcept BType, typename CType, typename U, typename... CIndices, typename... AIndices,
          typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires InSamePlace<AType, CType> || !TensorConcept<CType>;
    }
void einsum(U const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C, U const UAB_prefactor,
            std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr);

/*
 * Batched einsums calls over collections of tensors.
 */
template <Container CType, Container AType, Container BType, typename CPrefactorType, typename ABPrefactorType, typename... AIndices,
          typename... BIndices, typename... CIndices>
void einsum(CPrefactorType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C_list, ABPrefactorType const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A_list, std::tuple<BIndices...> const &B_indices, BType const &B_list,
            detail::AlgorithmChoice *algorithm_choice = nullptr);

// Einsums with provided prefactors.
// 1. C n A n B n is defined above as the base implementation.

// 2. C n A n B y
template <NotASmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, *B, algorithm_choice);
}

// 3. C n A y B n
template <SmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, B, algorithm_choice);
}

// 4. C n A y B y
template <SmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, *B, algorithm_choice);
}

// 5. C y A n B n
template <NotASmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, B, algorithm_choice);
}

// 6. C y A n B y
template <NotASmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, *B, algorithm_choice);
}

// 7. C y A y B n
template <SmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, B, algorithm_choice);
}

// 8. C y A y B y
template <SmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
void einsum(T const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, T const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, *B, algorithm_choice);
}

//
// Einsums with default prefactors.
//

// 1. C n A n B n
template <NotASmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, B, algorithm_choice);
}

// 2. C n A n B y
template <NotASmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, *B, algorithm_choice);
}

// 3. C n A y B n
template <SmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, B, algorithm_choice);
}

// 4. C n A y B y
template <SmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, *B, algorithm_choice);
}

// 5. C y A n B n
template <NotASmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, B, algorithm_choice);
}

// 6. C y A n B y
template <NotASmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, *B, algorithm_choice);
}

// 7. C y A y B n
template <SmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, B, algorithm_choice);
}

// 8. C y A y B y
template <SmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, *B, algorithm_choice);
}

//
// Element Transform
//

template <SmartPointer SmartPtr, typename UnaryOperator>
void element_transform(SmartPtr *C, UnaryOperator unary_opt) {
    element_transform(C->get(), unary_opt);
}

template <unsigned int N, typename... List>
constexpr auto get_n(std::tuple<List...> const &);

/**
 * Returns the mode-`mode` unfolding of `tensor` with modes startng at `0`
 *
 * @returns unfolded_tensor of shape ``(tensor.dim(mode), -1)``
 */
// template <unsigned int mode, template <typename, size_t> typename CType, size_t CRank, typename T = double>
// Tensor<T, 2> unfold(CType<T, CRank> const &source)
// requires(std::is_same_v<Tensor<T, CRank>, CType<T, CRank>>);

/** Computes the Khatri-Rao product of tensors A and B.
 *
 * Example:
 *    Tensor<2> result = khatri_rao(Indices{I, r}, A, Indices{J, r}, B);
 *
 * Result is described as {(I,J), r}. If multiple common indices are provided they will be collapsed into a single index in the result.
 */
template <TensorConcept AType, TensorConcept BType, typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires AType::Rank == sizeof...(AIndices);
        requires BType::Rank == sizeof...(BIndices);
    }
BasicTensorLike<AType, typename AType::ValueType, 2> khatri_rao(std::tuple<AIndices...> const &, AType const &A,
                                                                std::tuple<BIndices...> const &, BType const &B);
} // namespace einsums::tensor_algebra

#include <Einsums/TensorAlgebra/Backends/Dispatch.hpp>
#include <Einsums/TensorAlgebra/Backends/ElementTransform.hpp>
#include <Einsums/TensorAlgebra/Backends/KhatriRao.hpp>
#include <Einsums/TensorAlgebra/Backends/Unfold.hpp>