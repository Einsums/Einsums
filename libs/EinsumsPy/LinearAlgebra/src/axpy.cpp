//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void axpy(pybind11::object const &alpha, pybind11::buffer const &x, pybind11::buffer &y) {
    py::buffer_info x_info = x.request(false), y_info = y.request(true);

    if (x_info.ndim != y_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to axpy need to have the same rank!");
    }

    if (x_info.format != y_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the buffer objects need to be the same!");
    }

    EINSUMS_PY_LINALG_CALL(x_info.item_type_is_equivalent_to<Float>(), [&]() {
        auto tens = buffer_to_tensor<Float>(y);
        einsums::linear_algebra::detail::axpy(alpha.cast<Float>(), buffer_to_tensor<Float>(x), &tens);
    }());
}

void axpby(pybind11::object const &alpha, pybind11::buffer const &x, pybind11::object const &beta, pybind11::buffer &y) {
    py::buffer_info x_info = x.request(false), y_info = y.request(true);

    if (x_info.ndim != y_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to axpby need to have the same rank!");
    }

    if (x_info.format != y_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the buffer objects need to be the same!");
    }

    EINSUMS_PY_LINALG_CALL(x_info.item_type_is_equivalent_to<Float>(), [&]() {
        auto tens = buffer_to_tensor<Float>(y);
        einsums::linear_algebra::detail::axpby(alpha.cast<Float>(), buffer_to_tensor<Float>(x), beta.cast<Float>(), &tens);
    }());
}

} // namespace detail
} // namespace python
} // namespace einsums