//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <omp.h>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

pybind11::object dot(pybind11::buffer const &A, pybind11::buffer const &B) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false);

    if (A_info.ndim != B_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to dot need to have the same rank!");
    }

    for (int i = 0; i < A_info.ndim; i++) {
        if (A_info.shape[i] != B_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The inputs to dot need to have the same dimensions!");
        }
    }

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage formats of the tensors need to be the same!");
    }

    pybind11::object out;

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(),
                           (out = py::cast(einsums::linear_algebra::detail::dot(buffer_to_tensor<Float>(A), buffer_to_tensor<Float>(B)))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The format of the tensor could not be handled!");
    }

    return out;
}

pybind11::object true_dot(pybind11::buffer const &A, pybind11::buffer const &B) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false);

    if (A_info.ndim != B_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to true_dot need to have the same rank!");
    }

    for (int i = 0; i < A_info.ndim; i++) {
        if (A_info.shape[i] != B_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The inputs to true_dot need to have the same dimensions!");
        }
    }

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage formats of the tensors need to be the same!");
    }

    pybind11::object out;

    EINSUMS_PY_LINALG_CALL(
        A_info.item_type_is_equivalent_to<Float>(),
        (out = py::cast(einsums::linear_algebra::detail::true_dot(buffer_to_tensor<Float>(A), buffer_to_tensor<Float>(B)))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The format of the tensor could not be handled!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums