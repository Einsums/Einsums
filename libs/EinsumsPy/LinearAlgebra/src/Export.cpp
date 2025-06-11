//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void sum_square(pybind11::buffer const &A, py::object &scale, py::object &sum_sq) {
    pybind11::buffer_info A_info = A.request();

    if (A_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "sum_square can only be applied to rank-1 tensors!");
    }

    einsums::blas::int_t n    = A_info.shape[0];
    einsums::blas::int_t incx = A_info.strides[0] / A_info.itemsize;

    if (A_info.format == py::format_descriptor<float>::format()) {
        float new_scale  = scale.cast<float>();
        float new_sum_sq = sum_sq.cast<float>();

        blas::lassq<float>(n, (float const *)A_info.ptr, incx, &new_scale, &new_sum_sq);

        scale  = py::cast(new_scale);
        sum_sq = py::cast(new_sum_sq);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        double new_scale  = scale.cast<double>();
        double new_sum_sq = sum_sq.cast<double>();

        blas::lassq<double>(n, (double const *)A_info.ptr, incx, &new_scale, &new_sum_sq);

        scale  = py::cast(new_scale);
        sum_sq = py::cast(new_sum_sq);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        float new_scale  = scale.cast<float>();
        float new_sum_sq = sum_sq.cast<float>();

        blas::lassq<std::complex<float>>(n, (std::complex<float> const *)A_info.ptr, incx, &new_scale, &new_sum_sq);

        scale  = py::cast(new_scale);
        sum_sq = py::cast(new_sum_sq);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        double new_scale  = scale.cast<double>();
        double new_sum_sq = sum_sq.cast<double>();

        blas::lassq<std::complex<double>>(n, (std::complex<double> const *)A_info.ptr, incx, &new_scale, &new_sum_sq);

        scale  = py::cast(new_scale);
        sum_sq = py::cast(new_sum_sq);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Could not perform sum_square on the requested type {}!", A_info.format);
    }
}

void gemm(std::string const &transA, py::object const &alpha, std::string const &transB, pybind11::buffer const &A,
          pybind11::buffer const &B, py::object const &beta, pybind11::buffer &C) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false), C_info = C.request(true);

    if (A_info.ndim != 2 || B_info.ndim != 2 || C_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "A call to gemm can only take rank-2 tensors as input!");
    }

    blas::int_t A_m, A_k, lda = A_info.strides[0] / A_info.itemsize;
    blas::int_t B_k, B_n, ldb = B_info.strides[0] / B_info.itemsize;
    blas::int_t C_m = C_info.shape[0], C_n = C_info.shape[1], ldc = C_info.strides[0] / C_info.itemsize;

    if (transA != "N" && transA != "n") {
        A_m = A_info.shape[1];
        A_k = A_info.shape[0];
    } else {
        A_m = A_info.shape[0];
        A_k = A_info.shape[1];
    }

    if (transB != "N" && transB != "n") {
        B_n = B_info.shape[0];
        B_k = B_info.shape[1];
    } else {
        B_n = B_info.shape[1];
        B_k = B_info.shape[0];
    }

    if (A_m != C_m || A_k != B_k || B_n != C_n) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The rows and columns of the matrices passed into gemm are not compatible!");
    }

    if (A_info.format != B_info.format || A_info.format != C_info.format) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemm on matrices with the same stored type! Got A ({}), B ({}), C ({}).",
                                A_info.format, B_info.format, C_info.format);
    }

    if (A_info.format == py::format_descriptor<float>::format()) {
        blas::gemm<float>(transA[0], transB[0], A_m, B_n, A_k, alpha.cast<float>(), (float const *)A_info.ptr, lda,
                          (float const *)B_info.ptr, ldb, beta.cast<float>(), (float *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::gemm<double>(transA[0], transB[0], A_m, B_n, A_k, alpha.cast<double>(), (double const *)A_info.ptr, lda,
                           (double const *)B_info.ptr, ldb, beta.cast<double>(), (double *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        blas::gemm<std::complex<float>>(transA[0], transB[0], A_m, B_n, A_k, alpha.cast<std::complex<float>>(),
                                        (std::complex<float> const *)A_info.ptr, lda, (std::complex<float> const *)B_info.ptr, ldb,
                                        beta.cast<std::complex<float>>(), (std::complex<float> *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        blas::gemm<std::complex<double>>(transA[0], transB[0], A_m, B_n, A_k, alpha.cast<std::complex<double>>(),
                                         (std::complex<double> const *)A_info.ptr, lda, (std::complex<double> const *)B_info.ptr, ldb,
                                         beta.cast<std::complex<double>>(), (std::complex<double> *)C_info.ptr, ldc);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemm on floating point matrices! Got type {}.", A_info.format);
    }
}
} // namespace detail
} // namespace python
} // namespace einsums

EINSUMS_EXPORT void export_LinearAlgebra(py::module_ &mod) {
}