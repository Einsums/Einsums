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

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

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
          requires requires {
                requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
                requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
                requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
          }
void einsum_generic_algorithm(const std::tuple<UniqueIndices...> &unique_indices, const std::tuple<CIndices...> &C_indices,
                              const std::tuple<AIndices...> &A_indices, const std::tuple<BIndices...> &B_indices,
                              const std::tuple<UniqueDims...> &unique_dims, const CDataType C_prefactor, CType<CDataType, CRank> *C,
                              const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                              const AType<ADataType, ARank> &A, const BType<BDataType, BRank> &B);

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
          requires requires {
                requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
                requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
                requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
          }
auto einsum(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
            const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B)
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
 */
template <template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename U>
          requires requires {
                requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
                requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
                requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
          }
auto einsum(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C, const U UAB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>> &&
                        std::is_arithmetic_v<U>>;

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

#include "einsums/gpu/GPUTensorAlgebra.hpp"