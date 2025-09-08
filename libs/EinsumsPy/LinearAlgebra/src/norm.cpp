//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>

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

pybind11::object norm(einsums::linear_algebra::Norm type, pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    pybind11::object out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        out         = py::cast(einsums::linear_algebra::detail::norm(static_cast<char>(type), A_tens));
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the matrix norm of real or complex floating point matrices!");
    }

    return out;
}

pybind11::object vec_norm(pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The vec_norm function can only handle vectors!");
    }

    pybind11::object out;

    EINSUMS_PY_LINALG_CALL(A_info.format == py::format_descriptor<Float>::format(), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        out         = py::cast(einsums::linear_algebra::detail::vec_norm(A_tens));
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the vector norm of real or complex floating point vectors!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums