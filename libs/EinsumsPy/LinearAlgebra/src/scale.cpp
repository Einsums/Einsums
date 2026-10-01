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
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "EinsumsPy/Tensor/PyTensor.hpp"
#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void scale(pybind11::object factor, pybind11::buffer &A) {
    py::buffer_info A_info = A.request(true);

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        einsums::linear_algebra::detail::scale(factor.cast<Float>(), &A_tens);
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only scale matrices of real or complex floating point values!");
    }
}

void scale_row(int row, pybind11::object factor, pybind11::buffer &A) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only scale rows of a matrix!");
    }

    if (row < 0) {
        row += A_info.shape[0];
    }

    if (row < 0 || row >= A_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Requested row index is out of range!");
    }

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        einsums::linear_algebra::detail::scale_row(row, factor.cast<Float>(), &A_tens);
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only scale matrices of real or complex floating point values!");
    }
}

void scale_column(int col, pybind11::object factor, pybind11::buffer &A) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only scale columns of a matrix!");
    }

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(),
                           [&]() {
                               auto A_tens = buffer_to_tensor<Float>(A);
                               einsums::linear_algebra::detail::scale_column(col, factor.cast<Float>(), &A_tens);
                           }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only scale matrices of real or complex floating point values!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums