//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

namespace einsums {

/**
 * @brief Creates a diagonal matrix from a vector.
 *
 * @tparam T The datatype of the underlying data.
 * @param v The input vector.
 * @return A new rank-2 tensor with the diagonal elements set to \p v .
 */
template <typename T>
auto diagonal(Tensor<T, 1> const &v) -> Tensor<T, 2> {
    auto result = create_tensor<T>(v.name(), v.dim(0), v.dim(0));
    result.zero();
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

template <typename T>
auto diagonal_like(Tensor<T, 1> const &v, Tensor<T, 2> const &like) -> Tensor<T, 2> {
    auto result = create_tensor_like(v.name(), like);
    result.zero();
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

} // namespace einsums