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

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

template <typename T>
void scale_work(T factor, py::buffer &A) {

    py::buffer_info A_info = A.request(false);

    T *A_data = reinterpret_cast<T *>(A_info.ptr);

    std::vector<size_t> index_strides(A_info.ndim), A_strides(A_info.ndim);

    size_t elements = 1, easy_elems = 1, hard_elems = 1;
    int    easy_scale = -1; // Find the rank for the inner loop.
    size_t A_stride   = A_info.strides[A_info.ndim - 1] / A_info.itemsize;

    easy_scale = determine_easy_vector(A, &easy_elems, &A_stride, &hard_elems, &A_strides, &index_strides);

    if (easy_scale == 0) {
        // The entire tensor is contiguous, though may skip in the smallest stride.
        blas::int_t n = elements, inc = A_strides[A_info.ndim - 1];

        blas::scal(n, factor, A_data, inc);
    } else {
        recalc_index_strides(&index_strides, easy_scale);
        A_strides.resize(easy_scale);
        
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < hard_elems; i++) {
            size_t sentinel;
            sentinel_to_sentinels(i, index_strides, A_strides, sentinel);

            blas::scal(easy_elems, factor, ((T *)A_data) + sentinel, A_stride);
        }
    }
}

void scale(pybind11::object factor, pybind11::buffer &A) {
    py::buffer_info A_info = A.request(true);

    if (A_info.format == py::format_descriptor<float>::format()) {
        scale_work<float>(factor.cast<float>(), A);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        scale_work<double>(factor.cast<double>(), A);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        scale_work<std::complex<float>>(factor.cast<std::complex<float>>(), A);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        scale_work<std::complex<double>>(factor.cast<std::complex<double>>(), A);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only scale matrices of real or complex floating point values!");
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

    uint8_t *A_row = reinterpret_cast<uint8_t *>(A_info.ptr) + row * A_info.strides[0];

    if (A_info.format == py::format_descriptor<float>::format()) {
        blas::scal<float>(A_info.shape[1], factor.cast<float>(), (float *)A_row, A_info.strides[1] / sizeof(float));
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::scal<double>(A_info.shape[1], factor.cast<double>(), (double *)A_row, A_info.strides[1] / sizeof(double));
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        blas::scal<std::complex<float>>(A_info.shape[1], factor.cast<std::complex<float>>(), (std::complex<float> *)A_row,
                                        A_info.strides[1] / sizeof(std::complex<float>));
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        blas::scal<std::complex<double>>(A_info.shape[1], factor.cast<std::complex<double>>(), (std::complex<double> *)A_row,
                                         A_info.strides[1] / sizeof(std::complex<double>));
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only scale matrices of real or complex floating point values!");
    }
}

void scale_column(int col, pybind11::object factor, pybind11::buffer &A) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only scale columns of a matrix!");
    }

    if (col < 0) {
        col += A_info.shape[1];
    }

    if (col < 0 || col >= A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Requested column index is out of range!");
    }

    uint8_t *A_col = reinterpret_cast<uint8_t *>(A_info.ptr) + col * A_info.strides[1];

    if (A_info.format == py::format_descriptor<float>::format()) {
        blas::scal<float>(A_info.shape[0], factor.cast<float>(), (float *)A_col, A_info.strides[0] / sizeof(float));
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::scal<double>(A_info.shape[0], factor.cast<double>(), (double *)A_col, A_info.strides[0] / sizeof(double));
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        blas::scal<std::complex<float>>(A_info.shape[0], factor.cast<std::complex<float>>(), (std::complex<float> *)A_col,
                                        A_info.strides[0] / sizeof(std::complex<float>));
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        blas::scal<std::complex<double>>(A_info.shape[0], factor.cast<std::complex<double>>(), (std::complex<double> *)A_col,
                                         A_info.strides[0] / sizeof(std::complex<double>));
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only scale matrices of real or complex floating point values!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums