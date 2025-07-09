//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/TensorImpl.hpp>

namespace einsums {
namespace detail {

template <typename T>
void impl_axpy_contiguous(T &&alpha, TensorImpl<T const> const &in, TensorImpl<T> &out) {
    blas::vendor::axpy()
}

template <typename T>
void impl_axpy(T &&alpha, TensorImpl<T const> const &in, TensorImpl<T> &out) {

}

} // namespace detail
} // namespace einsums