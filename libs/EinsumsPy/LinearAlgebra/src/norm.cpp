#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>

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

pybind11::object norm(einsums::linear_algebra::Norm type, pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim == 1) {
        EINSUMS_THROW_EXCEPTION(rank_error,
                                "This is the matrix norm function. It can't handle vectors. Use the vec_norm function instead.");
    } else if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The norm function can only handle matrices!");
    }

    blas::int_t m = A_info.shape[1], n = A_info.shape[0];
    blas::int_t lda = A_info.strides[0] / A_info.itemsize;

    std::vector<uint8_t> work;

    if (type == einsums::linear_algebra::Norm::Infinity) {
        work.resize(m * A_info.itemsize);
    }

    pybind11::object out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()),
                           (out = py::cast(blas::lange<Float>(static_cast<char>(type), m, n, (Float const *)A_info.ptr, lda,
                                                             (RemoveComplexT<Float> *)work.data()))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the matrix norm of real or complex floating point matrices!");
    }

    return out;
}

pybind11::object vec_norm(pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The vec_norm function can only handle vectors!");
    }

    pybind11::object out;

    EINSUMS_PY_LINALG_CALL(A_info.format == py::format_descriptor<Float>::format(), ({
                               RemoveComplexT<Float> sumsq{0.0}, scale{0.0};
                               blas::lassq(A_info.shape[0], (Float const *)A_info.ptr, A_info.strides[0] / A_info.itemsize, &scale, &sumsq);
                               out = py::cast(std::sqrt(sumsq) * scale);
                           }))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the vector norm of real or complex floating point vectors!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums