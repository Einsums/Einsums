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

#include "EinsumsPy/Tensor/PyTensor.hpp"
#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void direct_product(pybind11::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B, pybind11::object const &beta,
                    pybind11::buffer &C) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false), C_info = C.request(true);

    if (A_info.ndim != B_info.ndim || A_info.ndim != C_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to direct_product need to have the same rank!");
    }

    for (int i = 0; i < A_info.ndim; i++) {
        if (A_info.shape[i] != B_info.shape[i] || A_info.shape[i] != C_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The shapes of the inputs to direct_product need to have the same dimensions!");
        }
    }

    if (A_info.format != B_info.format || A_info.format != C_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The inputs to direct_product need to store the same data types!");
    }

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), [&]() {
        auto C_tens = buffer_to_tensor<Float>(C);
        linear_algebra::detail::direct_product(alpha.cast<Float>(), buffer_to_tensor<Float>(A), buffer_to_tensor<Float>(B),
                                               beta.cast<Float>(), &C_tens);
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "direct_product can only handle real or complex floating point data types!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums