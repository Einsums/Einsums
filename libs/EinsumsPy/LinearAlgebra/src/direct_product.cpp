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

template <typename T>
void direct_product_loop(int loop_level, T alpha, py::buffer_info const &A_info, py::buffer_info const &B_info, T beta,
                         py::buffer_info &C_info, size_t A_index, size_t B_index, size_t C_index) {
    if (loop_level == A_info.ndim) {
        ((T *)C_info.ptr)[C_index] = beta * ((T *)C_info.ptr)[C_index] + alpha * ((T *)A_info.ptr)[A_index] * ((T *)B_info.ptr)[B_index];
    } else {
        size_t const dim      = A_info.shape[loop_level];
        size_t const A_stride = A_info.strides[loop_level] / sizeof(T);
        size_t const B_stride = B_info.strides[loop_level] / sizeof(T);
        size_t const C_stride = C_info.strides[loop_level] / sizeof(T);
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < dim; i++) {
            direct_product_loop(loop_level + 1, alpha, A_info, B_info, beta, C_info, A_index + A_stride * i, B_index + B_stride * i,
                                C_index + C_stride * i);
        }
    }
}

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

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()),
                           (direct_product_loop<Float>(0, alpha.cast<Float>(), A_info, B_info, beta.cast<Float>(), C_info, 0, 0, 0)))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "direct_product can only handle real or complex floating point data types!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums