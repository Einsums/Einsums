//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>

#include <complex>
#include <string>

namespace einsums {

/**
 * @brief Create a tensor and zero its  data.
 *
 * @tparam T The type to be stored by the tensor.
 * @tparam MultiIndex The types fo the indices.
 * @param[in] name The name of the new tensor.
 * @param[in] index The dimensions for the new tensor.
 * @return A new tensor whose elements have been zeroed.
 *
 * @versionadded{1.0.0}
 */
template <typename T = double, typename... MultiIndex>
auto create_zero_tensor(std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating zero tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(GlobalConfigMap::get_singleton().get_bool("row-major"), name, std::forward<MultiIndex>(index)...);
    A.zero();

    return A;
}

template <typename T = double, typename... MultiIndex>
auto create_zero_tensor(bool row_major, std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating zero tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(row_major, name, std::forward<MultiIndex>(index)...);
    A.zero();

    return A;
}

} // namespace einsums
