#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
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
        EINSUMS_THROW_EXCEPTION(rank_error, "sum_square can only be applied to rank-1 tensors!");
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

} // namespace detail
} // namespace python
} // namespace einsums