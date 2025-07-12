//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor.hpp>

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

template <typename T>
T det_work(pybind11::buffer const &A) {

    Tensor<T, 2> temp = PyTensorView<T>(A);

    std::vector<blas::int_t> pivots;
    int                      singular = linear_algebra::getrf(&temp, &pivots);

    if (singular > 0) {
        return T{0.0}; // Matrix is singular, so it has a determinant of zero.
    }

    if (singular < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument to getrf was invalid!", print::ordinal(-singular));
    }

    T ret{1.0};

    int parity = 0;

    // Calculate the effect of the pivots.
    size_t dim = temp.dim(0);
    for (int i = 0; i < dim; i++) {
        if (pivots[i] != i + 1) {
            parity++;
        }
    }

    // Calculate the contribution of the diagonal elements.
    // #pragma omp parallel for simd reduction(* : ret)
    for (int i = 0; i < dim; i++) {
        ret *= temp(i, i);
    }

    if (parity % 2 == 1) {
        ret *= T{-1.0};
    }

    return ret;
}

pybind11::object det(pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only take the determinant of matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only take the determinant of sqaure matrices!");
    }

    py::object out;
    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), (out = py::cast(det_work<Float>(A))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The input to det needs to store real or complex floating point values!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums