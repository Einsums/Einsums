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

#include "Einsums/BufferAllocator/BufferAllocator.hpp"
#include "Einsums/Utilities/InCollection.hpp"

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

void gemm(std::string const &transA, std::string const &transB, py::object const &alpha, pybind11::buffer const &A,
          pybind11::buffer const &B, py::object const &beta, pybind11::buffer &C) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false), C_info = C.request(true);

    if (A_info.ndim != 2 || B_info.ndim != 2 || C_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "A call to gemm can only take rank-2 tensors as input!");
    }

    blas::int_t A_m, A_k, lda = A_info.strides[0] / A_info.itemsize;
    blas::int_t B_k, B_n, ldb = B_info.strides[0] / B_info.itemsize;
    blas::int_t C_m = C_info.shape[0], C_n = C_info.shape[1], ldc = C_info.strides[0] / C_info.itemsize;

    char transA_ch = 'n';

    if (transA.length() >= 1) {
        if (is_in(transA[0], {'n', 'N', 't', 'T', 'c', 'C'})) {
            transA_ch = transA[0];
        }
    }

    char transB_ch = 'n';

    if (transB.length() >= 1) {
        if (is_in(transB[0], {'n', 'N', 't', 'T', 'c', 'C'})) {
            transB_ch = transB[0];
        }
    }

    if (not_in(transA_ch, {'n', 'N'})) {
        A_m = A_info.shape[1];
        A_k = A_info.shape[0];
    } else {
        A_m = A_info.shape[0];
        A_k = A_info.shape[1];
    }

    if (not_in(transB_ch, {'n', 'N'})) {
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
        blas::gemm<float>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<float>(), (float const *)A_info.ptr, lda,
                          (float const *)B_info.ptr, ldb, beta.cast<float>(), (float *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::gemm<double>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<double>(), (double const *)A_info.ptr, lda,
                           (double const *)B_info.ptr, ldb, beta.cast<double>(), (double *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        blas::gemm<std::complex<float>>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<std::complex<float>>(),
                                        (std::complex<float> const *)A_info.ptr, lda, (std::complex<float> const *)B_info.ptr, ldb,
                                        beta.cast<std::complex<float>>(), (std::complex<float> *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        blas::gemm<std::complex<double>>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<std::complex<double>>(),
                                         (std::complex<double> const *)A_info.ptr, lda, (std::complex<double> const *)B_info.ptr, ldb,
                                         beta.cast<std::complex<double>>(), (std::complex<double> *)C_info.ptr, ldc);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemm on floating point matrices! Got type {}.", A_info.format);
    }
}

void gemv(std::string const &transA, py::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B, py::object const &beta,
          pybind11::buffer &C) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false), C_info = C.request(true);

    if (A_info.ndim != 2 || B_info.ndim != 1 || C_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(
            dimension_error, "A call to gemv only takes a rank-2 tensor and a rank-1 tensor as input, and outputs into a rank-1 tensor.");
    }

    blas::int_t A_m, A_k, lda = A_info.strides[0] / A_info.itemsize;
    blas::int_t B_k = B_info.shape[0], ldb = B_info.strides[0] / B_info.itemsize;
    blas::int_t C_m = C_info.shape[0], ldc = C_info.strides[0] / C_info.itemsize;

    char transA_ch = 'n';

    if (transA.length() >= 1) {
        if (is_in(transA[0], {'n', 'N', 't', 'T', 'c', 'C'})) {
            transA_ch = transA[0];
        }
    }

    if (not_in(transA_ch, {'n', 'N'})) {
        A_m = A_info.shape[1];
        A_k = A_info.shape[0];
    } else {
        A_m = A_info.shape[0];
        A_k = A_info.shape[1];
    }

    if (A_m != C_m || A_k != B_k) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The rows and columns of the matrices passed into gemv are not compatible!");
    }

    if (A_info.format != B_info.format || A_info.format != C_info.format) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemv on tensors with the same stored type! Got A ({}), B ({}), C ({}).",
                                A_info.format, B_info.format, C_info.format);
    }

    if (A_info.format == py::format_descriptor<float>::format()) {
        blas::gemv<float>(transA_ch, A_m, A_k, alpha.cast<float>(), (float const *)A_info.ptr, lda, (float const *)B_info.ptr, ldb,
                          beta.cast<float>(), (float *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::gemv<double>(transA_ch, A_m, A_k, alpha.cast<double>(), (double const *)A_info.ptr, lda, (double const *)B_info.ptr, ldb,
                           beta.cast<double>(), (double *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        blas::gemv<std::complex<float>>(transA_ch, A_m, A_k, alpha.cast<std::complex<float>>(), (std::complex<float> const *)A_info.ptr,
                                        lda, (std::complex<float> const *)B_info.ptr, ldb, beta.cast<std::complex<float>>(),
                                        (std::complex<float> *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        blas::gemv<std::complex<double>>(transA_ch, A_m, A_k, alpha.cast<std::complex<double>>(), (std::complex<double> const *)A_info.ptr,
                                         lda, (std::complex<double> const *)B_info.ptr, ldb, beta.cast<std::complex<double>>(),
                                         (std::complex<double> *)C_info.ptr, ldc);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemv on floating point matrices! Got type {}.", A_info.format);
    }
}

void syev(std::string const &jobz, py::buffer &A, py::buffer &W) {
    py::buffer_info A_info = A.request(true), W_info = W.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "A call to syev/heev can only take a rank-2 tensor as input!");
    }

    if (W_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "A call to syev/heev can only take a rank-1 tensor as output!");
    }

    char jobz_ch = 'N';

    if (jobz.length() >= 1) {
        if (is_in(jobz[0], {'n', 'N', 'v', 'V'})) {
            jobz_ch = toupper(jobz[0]);
        }
    }

    blas::int_t n     = A_info.shape[0];
    blas::int_t lda   = A_info.strides[0] / A_info.itemsize;
    blas::int_t lwork = 3 * n;

    // Type check
    if (A_info.format == py::format_descriptor<float>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != A_info.format) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != A_info.format) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != py::format_descriptor<float>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != py::format_descriptor<double>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The tensors passed to syev/heev have the wrong storage types. Got A ({}), W ({}).",
                                A_info.format, W_info.format);
    }

    // Calculate the size of the work array.
    if (A_info.format == py::format_descriptor<float>::format()) {
        lwork = blas::syev<float>(jobz_ch, 'U', n, (float *)A_info.ptr, lda, (float *)W_info.ptr, (float *)nullptr, -1);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        lwork = blas::syev<double>(jobz_ch, 'U', n, (double *)A_info.ptr, lda, (double *)W_info.ptr, (double *)nullptr, -1);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        lwork = blas::syev<std::complex<float>>(jobz_ch, 'U', n, (std::complex<float> *)A_info.ptr, lda, (std::complex<float> *)W_info.ptr,
                                                (std::complex<float> *)nullptr, -1);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        lwork = blas::syev<std::complex<double>>(jobz_ch, 'U', n, (std::complex<double> *)A_info.ptr, lda,
                                                 (std::complex<double> *)W_info.ptr, (std::complex<double> *)nullptr, -1);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemv on floating point matrices! Got type {}.", A_info.format);
    }

    std::vector<char, BufferAllocator<char>> work_vec(lwork * A_info.itemsize);
    void                                    *work = (void *)work_vec.data();

    if (work == nullptr) {
        EINSUMS_THROW_EXCEPTION(
            std::runtime_error,
            "Could not allocate work array for syev call! Error unknown, but likely due to lack of memory for allocation.");
    }

    if (A_info.format == py::format_descriptor<float>::format()) {
        blas::syev<float>(jobz_ch, 'U', n, (float *)A_info.ptr, lda, (float *)W_info.ptr, (float *)work, lwork);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::syev<double>(jobz_ch, 'U', n, (double *)A_info.ptr, lda, (double *)W_info.ptr, (double *)work, lwork);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        std::vector<float, BufferAllocator<float>> rwork_vec(std::max(3 * n - 2, blas::int_t{1}));

        blas::heev<float>(jobz_ch, 'U', n, (std::complex<float> *)A_info.ptr, lda, (float *)W_info.ptr,
                                        (std::complex<float> *)work, lwork, rwork_vec.data());
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        std::vector<double, BufferAllocator<double>> rwork_vec(std::max(3 * n - 2, blas::int_t{1}));

        blas::heev<double>(jobz_ch, 'U', n, (std::complex<double> *)A_info.ptr, lda, (double *)W_info.ptr,
                                         (std::complex<double> *)work, lwork, rwork_vec.data());
    }
}

void geev(std::string const &jobvl, std::string const &jobvr, py::buffer &A, py::buffer &W, py::buffer &Vl, py::buffer &Vr) {
    EINSUMS_THROW_EXCEPTION(todo_error, "TODO: Handle the cases for geev.");
}

} // namespace detail
} // namespace python
} // namespace einsums

EINSUMS_EXPORT void export_LinearAlgebra(py::module_ &mod) {
}