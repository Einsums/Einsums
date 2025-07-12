//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void ger(pybind11::object const &alpha, pybind11::buffer const &X, pybind11::buffer const &Y, pybind11::buffer &A) {
    py::buffer_info X_info = X.request(false), Y_info = Y.request(false), A_info = A.request(true);

    if (X_info.ndim != 1 || Y_info.ndim != 1 || A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The ger function takes two rank-1 tensors and outputs into a rank-2 tensor!");
    }

    if (X_info.shape[0] != A_info.shape[0] || Y_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of the input and output tensors are not compatible!");
    }

    if (X_info.format != A_info.format || X_info.format != Y_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the tensors passed to ger must be the same!");
    }

    size_t m = X_info.shape[0], n = Y_info.shape[0];

    size_t incx = X_info.strides[0] / X_info.itemsize, incy = Y_info.strides[0] / Y_info.itemsize,
           lda = A_info.strides[0] / A_info.itemsize;

    EINSUMS_PY_LINALG_CALL(
        (X_info.format == py::format_descriptor<Float>::format()),
        (blas::ger(m, n, alpha.cast<Float>(), (Float const *)X_info.ptr, incx, (Float const *)Y_info.ptr, incy, (Float *)A_info.ptr, lda)))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "ger can only handle real and complex floating point types!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums