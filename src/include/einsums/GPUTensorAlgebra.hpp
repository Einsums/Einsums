//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_Index.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/GPULinearAlgebra.hpp"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/GPUTimer.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"

#include <algorithm>
#include <bits/utility.h>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
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

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra::gpu)

namespace detail {

/**
 * @brief The generic einsum algorithm performed on the GPU.
 *
 * This is the host call to access the GPU algorithms.
 *
 * @param unique_indices A list of all the indices with all duplicates removed.
 * @param C_indices The indices of the C tensor.
 * @param A_indices The indices of the A tensor.
 * @param B_indices The indices of the B tensor.
 * @param unique_dims The sizes of each of the indices involved in the computation.
 * @param C_prefactor The prefactor for the C tensor.
 * @param C The C tensor.
 * @param AB_prefactor The prefactor for the contraction between the A tensor and the B tensor.
 * @param A The A tensor.
 * @param B The B tensor.
 * @param stream The stream to use for the calculation.
 */
template <typename... UniqueIndices, typename... CIndices, typename... AIndices, typename... BIndices, typename... UniqueDims,
          template <typename, size_t> typename CType, typename CDataType, size_t CRank, template <typename, size_t> typename AType,
          typename ADataType, size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void einsum_generic_algorithm(const std::tuple<UniqueIndices...> &unique_indices, const std::tuple<CIndices...> &C_indices,
                              const std::tuple<AIndices...> &A_indices, const std::tuple<BIndices...> &B_indices,
                              const std::tuple<UniqueDims...> &unique_dims, const CDataType C_prefactor, CType<CDataType, CRank> *C,
                              const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                              const AType<ADataType, ARank> &A, const BType<BDataType, BRank> &B, hipStream_t stream);

/**
 * @brief Selector for the einsum algorithm.
 *
 * This function chooses between the various different algorithms.
 *
 * @param C_prefactor The prefactor for C.
 * @param C_indices The indices for the C tensor.
 * @param C The C tensor. Must be accesible by the device.
 * @param AB_prefactor The prefactor for the contraction between the A tensor and the B tensor.
 * @param A_indices The indices for the A tensor.
 * @param A The A tensor.
 * @param B_indices The indice for the B tensor.
 * @param B The B tensor.
 * @param stream The stream to use for the calculation.
 */
template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
            const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B, hipStream_t stream)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>>>;
} // namespace detail

/**
 * @brief Base call for einsums.
 *
 * This function wraps the call to @ref detail::einsum.
 *
 * @param UC_prefactor The prefactor for C.
 * @param C_indices The indices for the C tensor.
 * @param C The C tensor. Must be accesible by the device.
 * @param UAB_prefactor The prefactor for the contraction between the A tensor and the B tensor.
 * @param A_indices The indices for the A tensor.
 * @param A The A tensor.
 * @param B_indices The indice for the B tensor.
 * @param B The B tensor.
 * @param stream The stream to use for the calculation.
 */
template <template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename U>
auto einsum(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C, const U UAB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B, hipStream_t stream = 0)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>> &&
                        std::is_arithmetic_v<U>>;

// Einsums with provided prefactors.
// 1. C n A n B n is defined above as the base implementation.

// 2. C n A n B y
template <NotASmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
void einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, *B, stream);
}

// 3. C n A y B n
template <SmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, B, stream);
}

// 4. C n A y B y
template <SmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, *B, stream);
}

// 5. C y A n B n
template <NotASmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, B, stream);
}

// 6. C y A n B y
template <NotASmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, *B, stream);
}

// 7. C y A y B n
template <SmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, B, stream);
}

// 8. C y A y B y
template <SmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, *B, stream);
}

//
// Einsums with default prefactors.
//

// 1. C n A n B n
template <NotASmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, B, stream);
}

// 2. C n A n B y
template <NotASmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, *B, stream);
}

// 3. C n A y B n
template <SmartPointer AType, NotASmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, B, stream);
}

// 4. C n A y B y
template <SmartPointer AType, SmartPointer BType, NotASmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, *B, stream);
}

// 5. C y A n B n
template <NotASmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices,
          typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, B, stream);
}

// 6. C y A n B y
template <NotASmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, *B, stream);
}

// 7. C y A y B n
template <SmartPointer AType, NotASmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, B, stream);
}

// 8. C y A y B y
template <SmartPointer AType, SmartPointer BType, SmartPointer CType, typename... CIndices, typename... AIndices, typename... BIndices>
void einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B, hipStream_t stream = 0) {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, *B, stream);
}

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra::gpu)

#include "einsums/gpu/GPUTensorAlgebra.imp.hip"