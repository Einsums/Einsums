#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>

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

void getrf(pybind11::buffer &A, std::vector<blas::int_t> &pivot) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU decomposition on matrices!");
    }

    if (pivot.size() < std::min(A_info.shape[0], A_info.shape[1])) {
        pivot.resize(std::min(A_info.shape[0], A_info.shape[1]));
    }

    int result = 0;

    EINSUMS_PY_LINALG_CALL(
        (A_info.format == py::format_descriptor<Float>::format()),
        (result = blas::getrf(A_info.shape[0], A_info.shape[1], (Float *)A_info.ptr, A_info.strides[0] / A_info.itemsize, pivot.data())))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only decompose matrices of real or complex floating point values!");
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument had an illegal value!", print::ordinal(-result));
    } else if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The system could not be factorized!");
    }
}

void getri(pybind11::buffer &A, std::vector<blas::int_t> &pivot) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU inversion on matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the inverse of a square matrix!");
    }

    if (pivot.size() != A_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The pivot list has not been initialized! Have you performed getrf on this matrix first?");
    }

    int result = 0;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()),
                           (result = blas::getri(A_info.shape[0], (Float *)A_info.ptr, A_info.strides[0] / A_info.itemsize, pivot.data())))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only decompose matrices of real or complex floating point values!");
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument had an illegal value!", print::ordinal(-result));
    } else if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The matrix is singular!");
    }
}

void invert(pybind11::buffer &A) {

    py::buffer_info A_info = A.request(true);

    std::vector<blas::int_t> pivot(A_info.shape[0]);
    getrf(A, pivot);

    getri(A, pivot);
}

} // namespace detail
} // namespace python
} // namespace einsums