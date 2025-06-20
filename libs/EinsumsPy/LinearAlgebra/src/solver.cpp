#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

template <typename T>
void gesv_work(pybind11::buffer &A, pybind11::buffer &B) {
    py::buffer_info A_info = A.request(true), B_info = B.request(true);
    Tensor<T, 2>    A_T{"A Transpose", A_info.shape[1], A_info.shape[0]},
        B_T{"B Transpose", (B_info.ndim == 1) ? 1 : B_info.shape[1], B_info.shape[0]};
    TensorView<T, 2> A_view{(T *)A_info.ptr, Dim{A_info.shape[0], A_info.shape[1]},
                            Stride{A_info.strides[0] / A_info.itemsize, A_info.strides[1] / A_info.itemsize}};
    TensorView<T, 2> B_view{(T *)B_info.ptr, Dim{B_info.shape[0], (B_info.ndim == 1) ? 1 : B_info.shape[1]},
                            (B_info.ndim == 1) ? Stride{B_info.strides[0] * B_info.shape[0] / B_info.itemsize, 1}
                                               : Stride{B_info.strides[0] / B_info.itemsize, B_info.strides[1] / B_info.itemsize}};

    // Convert to column major.
    tensor_algebra::permute(Indices{index::i, index::j}, &A_T, Indices{index::j, index::i}, A_view);
    tensor_algebra::permute(Indices{index::i, index::j}, &B_T, Indices{index::j, index::i}, B_view);

    // Solve.
    int info = linear_algebra::gesv(&A_T, &B_T);

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The input matrix is singular!");
    }

    // Convert back to row major.
    tensor_algebra::permute(Indices{index::i, index::j}, &A_view, Indices{index::j, index::i}, A_T);
    tensor_algebra::permute(Indices{index::i, index::j}, &B_view, Indices{index::j, index::i}, B_T);
}

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

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The storage types of the input matrices need to be the same!");
    } else if (A_info.format == py::format_descriptor<float>::format()) {
        gesv_work<float>(A, B);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        gesv_work<double>(A, B);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        gesv_work<std::complex<float>>(A, B);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        gesv_work<std::complex<double>>(A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The storage type of the input to the solver is invalid!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums