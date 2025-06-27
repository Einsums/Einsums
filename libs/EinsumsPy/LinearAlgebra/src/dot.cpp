#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <omp.h>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Einsums/Config/CompilerSpecific.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

template <typename T>
T dot_work(py::buffer_info const &A_info, py::buffer_info const &B_info, std::vector<size_t> &index_strides, std::vector<size_t> &A_strides,
           std::vector<size_t> &B_strides, size_t easy_dot, size_t easy_elems, size_t hard_elems, size_t A_stride, size_t B_stride) {
    T out{0.0};

    T const *A_data = reinterpret_cast<T const *>(A_info.ptr);
    T const *B_data = reinterpret_cast<T const *>(B_info.ptr);

    std::vector<T> buffer(omp_get_max_threads());

    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = T{0.0};
    }

#pragma omp parallel
    {
        T &out_ref = buffer[omp_get_thread_num()];
#pragma omp for
        for (size_t i = 0; i < hard_elems; i++) {
            size_t A_sentinel, B_sentinel;
            sentinel_to_sentinels(i, index_strides, A_strides, A_sentinel, B_strides, B_sentinel);

            out_ref += blas::dot<T>(easy_elems, A_data + A_sentinel, A_stride, B_data + B_sentinel, B_stride);
        }
    }

    for (size_t i = 0; i < buffer.size(); i++) {
        out += buffer[i];
    }

    return out;
}

pybind11::object dot(pybind11::buffer const &A, pybind11::buffer const &B) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false);

    if (A_info.ndim != B_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to dot need to have the same rank!");
    }

    for (int i = 0; i < A_info.ndim; i++) {
        if (A_info.shape[i] != B_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The inputs to dot need to have the same dimensions!");
        }
    }

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The storage formats of the tensors need to be the same!");
    }

    std::vector<size_t> A_index_strides(A_info.ndim), B_index_strides(B_info.ndim), index_strides(0);

    size_t A_easy_elems = 1, A_hard_elems = 1, B_easy_elems = 1, B_hard_elems = 1;
    int    A_easy_dot = -1, B_easy_dot = -1; // Find the rank for the inner loop.

    size_t easy_elems, hard_elems;

    std::vector<size_t> A_strides(A_info.ndim), B_strides(B_info.ndim);

    size_t A_stride = A_info.strides[A_info.ndim - 1] / A_info.itemsize;
    size_t B_stride = B_info.strides[B_info.ndim - 1] / B_info.itemsize;

    A_easy_dot = determine_easy_vector(A, &A_easy_elems, &A_stride, &A_hard_elems, &A_strides, &A_index_strides);
    B_easy_dot = determine_easy_vector(B, &B_easy_elems, &B_stride, &B_hard_elems, &B_strides, &B_index_strides);

    int easy_dot;

    if (A_easy_dot > B_easy_dot) {
        index_strides.swap(A_index_strides);
        easy_dot   = A_easy_dot;
        easy_elems = A_easy_elems;
        hard_elems = A_hard_elems;
    } else {
        index_strides.swap(B_index_strides);
        easy_dot   = B_easy_dot;
        easy_elems = B_easy_elems;
        hard_elems = B_hard_elems;
    }

    A_strides.resize(easy_dot);
    B_strides.resize(easy_dot);

    if (easy_dot == 0) {
        if (A_info.format == py::format_descriptor<float>::format()) {
            return py::cast(blas::dot<float>(easy_elems, (float const *)A_info.ptr, A_stride, (float const *)B_info.ptr, B_stride));
        } else if (A_info.format == py::format_descriptor<double>::format()) {
            return py::cast(blas::dot<double>(easy_elems, (double const *)A_info.ptr, A_stride, (double const *)B_info.ptr, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
            return py::cast(blas::dot<std::complex<float>>(easy_elems, (std::complex<float> const *)A_info.ptr, A_stride,
                                                           (std::complex<float> const *)B_info.ptr, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
            return py::cast(blas::dot<std::complex<double>>(easy_elems, (std::complex<double> const *)A_info.ptr, A_stride,
                                                            (std::complex<double> const *)B_info.ptr, B_stride));
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform the dot product on real or complex floating point inputs!");
        }
    } else {
        recalc_index_strides(&index_strides, easy_dot);
        if (A_info.format == py::format_descriptor<float>::format()) {
            return py::cast(
                dot_work<float>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems, hard_elems, A_stride, B_stride));
        } else if (A_info.format == py::format_descriptor<double>::format()) {
            return py::cast(dot_work<double>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems, hard_elems,
                                             A_stride, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
            return py::cast(dot_work<std::complex<float>>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems,
                                                          hard_elems, A_stride, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
            return py::cast(dot_work<std::complex<double>>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems,
                                                           hard_elems, A_stride, B_stride));
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform the dot product on real or complex floating point inputs!");
        }
    }
}

template <typename T>
T true_dot_work(py::buffer_info const &A_info, py::buffer_info const &B_info, std::vector<size_t> &index_strides,
                std::vector<size_t> &A_strides, std::vector<size_t> &B_strides, size_t easy_dot, size_t easy_elems, size_t hard_elems,
                size_t A_stride, size_t B_stride) {
    T out{0.0};

    T const *A_data = reinterpret_cast<T const *>(A_info.ptr);
    T const *B_data = reinterpret_cast<T const *>(B_info.ptr);

    std::vector<T> buffer(omp_get_max_threads());

    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = T{0.0};
    }

#pragma omp parallel
    {
        T &out_ref = buffer[omp_get_thread_num()];
#pragma omp for
        for (size_t i = 0; i < hard_elems; i++) {
            size_t A_sentinel, B_sentinel;
            sentinel_to_sentinels(i, index_strides, A_strides, A_sentinel, B_strides, B_sentinel);

            out_ref += blas::dotc<T>(easy_elems, A_data + A_sentinel, A_stride, B_data + B_sentinel, B_stride);
        }
    }

    for (size_t i = 0; i < buffer.size(); i++) {
        out += buffer[i];
    }

    return out;
}

pybind11::object true_dot(pybind11::buffer const &A, pybind11::buffer const &B) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false);

    if (A_info.ndim != B_info.ndim) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to true_dot need to have the same rank!");
    }

    for (int i = 0; i < A_info.ndim; i++) {
        if (A_info.shape[i] != B_info.shape[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The inputs to true_dot need to have the same dimensions!");
        }
    }

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The storage formats of the tensors need to be the same!");
    }

    std::vector<size_t> A_index_strides(A_info.ndim), B_index_strides(B_info.ndim), index_strides(0);

    size_t A_easy_elems = 1, A_hard_elems = 1, B_easy_elems = 1, B_hard_elems = 1;
    int    A_easy_dot = -1, B_easy_dot = -1; // Find the rank for the inner loop.

    size_t easy_elems, hard_elems;

    std::vector<size_t> A_strides(A_info.ndim), B_strides(B_info.ndim);

    size_t A_stride = A_info.strides[A_info.ndim - 1] / A_info.itemsize;
    size_t B_stride = B_info.strides[B_info.ndim - 1] / B_info.itemsize;

    A_easy_dot = determine_easy_vector(A, &A_easy_elems, &A_stride, &A_hard_elems, &A_strides, &A_index_strides);
    B_easy_dot = determine_easy_vector(B, &B_easy_elems, &B_stride, &B_hard_elems, &B_strides, &B_index_strides);

    int easy_dot;

    if (A_easy_dot > B_easy_dot) {
        index_strides.swap(A_index_strides);
        easy_dot = A_easy_dot;
        B_strides.resize(easy_dot);
        easy_elems = A_easy_elems;
        hard_elems = A_hard_elems;
    } else {
        index_strides.swap(B_index_strides);
        easy_dot = B_easy_dot;
        A_strides.resize(easy_dot);
        easy_elems = B_easy_elems;
        hard_elems = B_hard_elems;
    }

    if (easy_dot == 0) {
        if (A_info.format == py::format_descriptor<float>::format()) {
            return py::cast(blas::dot<float>(easy_elems, (float const *)A_info.ptr, A_stride, (float const *)B_info.ptr, B_stride));
        } else if (A_info.format == py::format_descriptor<double>::format()) {
            return py::cast(blas::dot<double>(easy_elems, (double const *)A_info.ptr, A_stride, (double const *)B_info.ptr, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
            return py::cast(blas::dotc<std::complex<float>>(easy_elems, (std::complex<float> const *)A_info.ptr, A_stride,
                                                            (std::complex<float> const *)B_info.ptr, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
            return py::cast(blas::dotc<std::complex<double>>(easy_elems, (std::complex<double> const *)A_info.ptr, A_stride,
                                                             (std::complex<double> const *)B_info.ptr, B_stride));
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform the dot product on real or complex floating point inputs!");
        }
    } else {
        recalc_index_strides(&index_strides, easy_dot);
        if (A_info.format == py::format_descriptor<float>::format()) {
            return py::cast(
                dot_work<float>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems, hard_elems, A_stride, B_stride));
        } else if (A_info.format == py::format_descriptor<double>::format()) {
            return py::cast(dot_work<double>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems, hard_elems,
                                             A_stride, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
            return py::cast(true_dot_work<std::complex<float>>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems,
                                                               hard_elems, A_stride, B_stride));
        } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
            return py::cast(true_dot_work<std::complex<double>>(A_info, B_info, index_strides, A_strides, B_strides, easy_dot, easy_elems,
                                                                hard_elems, A_stride, B_stride));
        } else {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform the dot product on real or complex floating point inputs!");
        }
    }
}

} // namespace detail
} // namespace python
} // namespace einsums