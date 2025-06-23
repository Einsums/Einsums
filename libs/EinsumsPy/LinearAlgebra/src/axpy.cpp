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

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

template <typename T>
void axpy_work(T alpha, pybind11::buffer_info const &x_info, pybind11::buffer_info &y_info, std::vector<size_t> &index_strides,
               std::vector<size_t> &x_strides, std::vector<size_t> &y_strides, size_t easy, size_t easy_elems, size_t hard_elems,
               size_t x_stride, size_t y_stride) {
    index_strides.resize(easy);
    x_strides.resize(easy);
    y_strides.resize(easy);

    T const *x_data = reinterpret_cast<T const *>(x_info.ptr);
    T const *y_data = reinterpret_cast<T const *>(y_info.ptr);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < hard_elems; i++) {
        size_t x_sentinel, y_sentinel;
        sentinel_to_sentinels(i, index_strides, x_strides, x_sentinel, y_strides, y_sentinel);

        blas::axpy<T>(easy_elems, alpha, ((T const *)x_data) + x_sentinel, x_stride, ((T *)y_data) + y_sentinel, y_stride);
    }
}

void axpy(pybind11::object const &alpha, pybind11::buffer const &x, pybind11::buffer &y) {
    py::buffer_info x_info = x.request(false), y_info = y.request(true);

    if (x_info.ndim != y_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to axpy need to have the same rank!");
    }

    if (x_info.format != y_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the buffer objects need to be the same!");
    }

    for (int i = 0; i < x_info.ndim; i++) {
        if (x_info.shape[i] != y_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The inputs to axpy need to have the same shape!");
        }
    }

    std::vector<size_t> x_index_strides, y_index_strides, index_strides;

    size_t elements = 1, x_easy_elems = 1, x_hard_elems = 1, y_easy_elems = 1, y_hard_elems = 1;
    int    x_easy = -1, y_easy = -1; // Find the rank for the inner loop.

    size_t easy_elems, hard_elems;

    std::vector<size_t> x_strides(x_info.ndim), y_strides(y_info.ndim);

    size_t x_stride;
    size_t y_stride;

    x_easy = determine_easy_vector(x, &x_easy_elems, &x_stride, &x_hard_elems, &x_strides, &x_index_strides);
    y_easy = determine_easy_vector(y, &y_easy_elems, &y_stride, &y_hard_elems, &y_strides, &y_index_strides);

    int easy;

    if (x_easy > y_easy) {
        index_strides.swap(x_index_strides);
        easy = x_easy;
        y_strides.resize(easy);
        easy_elems = x_easy_elems;
        hard_elems = x_hard_elems;
    } else {
        index_strides.swap(y_index_strides);
        easy = y_easy;
        x_strides.resize(easy);
        easy_elems = y_easy_elems;
        hard_elems = y_hard_elems;
    }

    if (easy == 0) {
        if (x_info.format == py::format_descriptor<float>::format()) {
            return blas::axpy<float>(easy_elems, alpha.cast<float>(), (float const *)x_info.ptr, x_stride, (float *)y_info.ptr, y_stride);
        } else if (x_info.format == py::format_descriptor<double>::format()) {
            return blas::axpy<double>(easy_elems, alpha.cast<double>(), (double const *)x_info.ptr, x_stride, (double *)y_info.ptr,
                                      y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<float>>::format()) {
            return blas::axpy<std::complex<float>>(easy_elems, alpha.cast<std::complex<float>>(), (std::complex<float> const *)x_info.ptr,
                                                   x_stride, (std::complex<float> *)y_info.ptr, y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<double>>::format()) {
            return blas::axpy<std::complex<double>>(easy_elems, alpha.cast<std::complex<double>>(),
                                                    (std::complex<double> const *)x_info.ptr, x_stride, (std::complex<double> *)y_info.ptr,
                                                    y_stride);
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform the dot product on real or complex floating point inputs!");
        }
    } else {
        recalc_index_strides(&index_strides, easy);
        if (x_info.format == py::format_descriptor<float>::format()) {
            axpy_work<float>(alpha.cast<float>(), x_info, y_info, index_strides, x_strides, y_strides, easy, easy_elems, hard_elems,
                             x_stride, y_stride);
        } else if (x_info.format == py::format_descriptor<double>::format()) {
            axpy_work<double>(alpha.cast<double>(), x_info, y_info, index_strides, x_strides, y_strides, easy, easy_elems, hard_elems,
                              x_stride, y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<float>>::format()) {
            axpy_work<std::complex<float>>(alpha.cast<std::complex<float>>(), x_info, y_info, index_strides, x_strides, y_strides, easy,
                                           easy_elems, hard_elems, x_stride, y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<double>>::format()) {
            axpy_work<std::complex<double>>(alpha.cast<std::complex<double>>(), x_info, y_info, index_strides, x_strides, y_strides, easy,
                                            easy_elems, hard_elems, x_stride, y_stride);
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform axpy on real or complex floating point inputs!");
        }
    }
}

template <typename T>
void axpby_work(T alpha, pybind11::buffer_info const &x_info, T beta, pybind11::buffer_info &y_info, std::vector<size_t> &index_strides,
                std::vector<size_t> &x_strides, std::vector<size_t> &y_strides, size_t easy, size_t easy_elems, size_t hard_elems,
                size_t x_stride, size_t y_stride) {
    index_strides.resize(easy);
    x_strides.resize(easy);
    y_strides.resize(easy);

    T const *x_data = reinterpret_cast<T const *>(x_info.ptr);
    T const *y_data = reinterpret_cast<T const *>(y_info.ptr);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < hard_elems; i++) {
        size_t x_sentinel, y_sentinel;
        sentinel_to_sentinels(i, index_strides, x_strides, x_sentinel, y_strides, y_sentinel);

        blas::axpby<T>(easy_elems, alpha, ((T const *)x_data) + x_sentinel, x_stride, beta, ((T *)y_data) + y_sentinel, y_stride);
    }
}

void axpby(pybind11::object const &alpha, pybind11::buffer const &x, pybind11::object const &beta, pybind11::buffer &y) {
    py::buffer_info x_info = x.request(false), y_info = y.request(true);

    if (x_info.ndim != y_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to axpby need to have the same rank!");
    }

    if (x_info.format != y_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the buffer objects need to be the same!");
    }

    for (int i = 0; i < x_info.ndim; i++) {
        if (x_info.shape[i] != y_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The inputs to axpby need to have the same shape!");
        }
    }

    std::vector<size_t> x_index_strides, y_index_strides, index_strides;

    size_t elements = 1, x_easy_elems = 1, x_hard_elems = 1, y_easy_elems = 1, y_hard_elems = 1;
    int    x_easy = -1, y_easy = -1; // Find the rank for the inner loop.

    size_t easy_elems, hard_elems;

    std::vector<size_t> x_strides(x_info.ndim), y_strides(y_info.ndim);

    size_t x_stride;
    size_t y_stride;

    x_easy = determine_easy_vector(x, &x_easy_elems, &x_stride, &x_hard_elems, &x_strides, &x_index_strides);
    y_easy = determine_easy_vector(y, &y_easy_elems, &y_stride, &y_hard_elems, &y_strides, &y_index_strides);

    int easy;

    if (x_easy > y_easy) {
        index_strides.swap(x_index_strides);
        easy = x_easy;
        y_strides.resize(easy);
        easy_elems = x_easy_elems;
        hard_elems = x_hard_elems;
    } else {
        index_strides.swap(y_index_strides);
        easy = y_easy;
        x_strides.resize(easy);
        easy_elems = y_easy_elems;
        hard_elems = y_hard_elems;
    }

    if (easy == 0) {
        if (x_info.format == py::format_descriptor<float>::format()) {
            return blas::axpby<float>(easy_elems, alpha.cast<float>(), (float const *)x_info.ptr, x_stride, beta.cast<float>(),
                                      (float *)y_info.ptr, y_stride);
        } else if (x_info.format == py::format_descriptor<double>::format()) {
            return blas::axpby<double>(easy_elems, alpha.cast<double>(), (double const *)x_info.ptr, x_stride, beta.cast<double>(),
                                       (double *)y_info.ptr, y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<float>>::format()) {
            return blas::axpby<std::complex<float>>(easy_elems, alpha.cast<std::complex<float>>(), (std::complex<float> const *)x_info.ptr,
                                                    x_stride, beta.cast<std::complex<float>>(), (std::complex<float> *)y_info.ptr,
                                                    y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<double>>::format()) {
            return blas::axpby<std::complex<double>>(easy_elems, alpha.cast<std::complex<double>>(),
                                                     (std::complex<double> const *)x_info.ptr, x_stride, beta.cast<std::complex<double>>(),
                                                     (std::complex<double> *)y_info.ptr, y_stride);
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform the dot product on real or complex floating point inputs!");
        }
    } else {
        recalc_index_strides(&index_strides, easy);
        if (x_info.format == py::format_descriptor<float>::format()) {
            axpby_work<float>(alpha.cast<float>(), x_info, beta.cast<float>(), y_info, index_strides, x_strides, y_strides, easy,
                              easy_elems, hard_elems, x_stride, y_stride);
        } else if (x_info.format == py::format_descriptor<double>::format()) {
            axpby_work<double>(alpha.cast<double>(), x_info, beta.cast<double>(), y_info, index_strides, x_strides, y_strides, easy,
                               easy_elems, hard_elems, x_stride, y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<float>>::format()) {
            axpby_work<std::complex<float>>(alpha.cast<std::complex<float>>(), x_info, beta.cast<std::complex<float>>(), y_info,
                                            index_strides, x_strides, y_strides, easy, easy_elems, hard_elems, x_stride, y_stride);
        } else if (x_info.format == py::format_descriptor<std::complex<double>>::format()) {
            axpby_work<std::complex<double>>(alpha.cast<std::complex<double>>(), x_info, beta.cast<std::complex<double>>(), y_info,
                                             index_strides, x_strides, y_strides, easy, easy_elems, hard_elems, x_stride, y_stride);
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform axpy on real or complex floating point inputs!");
        }
    }
}

} // namespace detail
} // namespace python
} // namespace einsums