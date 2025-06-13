#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void gesv(pybind11::buffer &A, pybind11::buffer &B) {
    py::buffer_info A_info = A.request(true), B_info = B.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The coefficient matrix needs to be a matrix!");
    }

    if (B_info.ndim != 1 && B_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The constant matrix needs to be a vector or a matrix!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The solver can only handle square matrices!");
    }

    blas::int_t n = A_info.shape[0], nrhs = (B_info.ndim == 2) ? B_info.shape[1] : 1;
    blas::int_t lda = A_info.strides[0] / A_info.itemsize, ldb = B_info.strides[0] / B_info.itemsize;
    std::vector<blas::int_t, BufferAllocator<blas::int_t>> ipiv(n);

    blas::int_t info = 0;

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The storage types of the input matrices need to be the same!");
    } else if (A_info.format == py::format_descriptor<float>::format()) {
        info = blas::gesv<float>(n, nrhs, (float *)A_info.ptr, lda, ipiv.data(), (float *)B_info.ptr, ldb);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        info = blas::gesv<double>(n, nrhs, (double *)A_info.ptr, lda, ipiv.data(), (double *)B_info.ptr, ldb);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        info = blas::gesv<std::complex<float>>(n, nrhs, (std::complex<float> *)A_info.ptr, lda, ipiv.data(),
                                               (std::complex<float> *)B_info.ptr, ldb);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        info = blas::gesv<std::complex<double>>(n, nrhs, (std::complex<double> *)A_info.ptr, lda, ipiv.data(),
                                                (std::complex<double> *)B_info.ptr, ldb);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The storage type of the input to the solver is invalid!");
    }

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The input matrix is singular!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums