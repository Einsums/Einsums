#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void getrf(pybind11::buffer &A, std::vector<blas::int_t> &pivot) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU decomposition on matrices!");
    }

    if (pivot.size() < std::min(A_info.shape[0], A_info.shape[1])) {
        pivot.resize(std::min(A_info.shape[0], A_info.shape[1]));
    }

    int result = 0;

    EINSUMS_PY_LINALG_CALL(
        (A_info.format == py::format_descriptor<Float>::format()),
        (result = blas::getrf(A_info.shape[0], A_info.shape[1], (Float *)A_info.ptr, A_info.strides[0] / A_info.itemsize, pivot.data())))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only decompose matrices of real or complex floating point values!");
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument had an illegal value!", print::ordinal(-result));
    } else if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The system could not be factorized!");
    }
}

void getri(pybind11::buffer &A, std::vector<blas::int_t> &pivot) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU inversion on matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the inverse of a square matrix!");
    }

    if (pivot.size() != A_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The pivot list has not been initialized! Have you performed getrf on this matrix first?");
    }

    int result = 0;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()),
                           (result = blas::getri(A_info.shape[0], (Float *)A_info.ptr, A_info.strides[0] / A_info.itemsize, pivot.data())))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only decompose matrices of real or complex floating point values!");
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument had an illegal value!", print::ordinal(-result));
    } else if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The matrix is singular!");
    }
}

void invert(pybind11::buffer &A) {

    py::buffer_info A_info = A.request(true);

    std::vector<blas::int_t> pivot(A_info.shape[0]);
    getrf(A, pivot);

    getri(A, pivot);
}

template <typename T>
RuntimeTensor<T> pseudoinverse_work(pybind11::buffer const &_A, RemoveComplexT<T> tol) {
    Tensor<T, 2> A = PyTensorView<T>(_A);

    auto [U, S, Vh] = linear_algebra::svd_dd(A);

    size_t new_dim{0};
    for (size_t v = 0; v < S.dim(0); v++) {
        RemoveComplexT<T> val = S(v);
        if (val > tol)
            linear_algebra::scale_column(v, T{1.0} / val, &U);
        else {
            new_dim = v;
            break;
        }
    }

    TensorView<T, 2> U_view = U(All, Range{0, new_dim});
    TensorView<T, 2> V_view = Vh(Range{0, new_dim}, All);

    Tensor<T, 2> pinv("pinv", A.dim(0), A.dim(1));
    linear_algebra::gemm<false, false>(T{1.0}, U_view, V_view, T{0.0}, &pinv);

    return RuntimeTensor<T>(std::move(pinv));
}

py::object pseudoinverse(pybind11::buffer const &A, pybind11::object const &tol) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only take the pseudoinverse of matrices!");
    }

    py::object out;
    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()),
                           (out = py::cast(pseudoinverse_work<Float>(A, tol.cast<RemoveComplexT<Float>>()))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The input to pseudo need to store real or complex floating point values!");
    }

    return out;
}

template <typename T>
pybind11::tuple qr_work(pybind11::buffer const &_A) {
    // Copy A because it will be overwritten by the QR call.
    Tensor<T, 2>      A = PyTensorView<T>(_A);
    blas::int_t const m = A.dim(0);
    blas::int_t const n = A.dim(1);

    Tensor<T, 1> tau("tau", std::min(m, n));
    // Compute QR factorization of Y
    blas::int_t info = blas::geqrf(m, n, A.data(), n, tau.data());

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "{} parameter to geqrf has an illegal value.", print::ordinal(-info));
    }

    // Extract Matrix Q out of QR factorization
    // blas::int_t info2 = blas::orgqr(m, n, tau.dim(0), A.data(), n, const_cast<const double *>(tau.data()));
    return py::make_tuple(PyTensor<T>(std::move(A)), PyTensor<T>(std::move(tau)));
}

py::tuple qr(pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only take the pseudoinverse of matrices!");
    }

    py::tuple out;
    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), (out = qr_work<Float>(A)))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The inputs to qr need to store real or complex floating point values!");
    }

    return out;
}

template <typename T>
RuntimeTensor<T> q_work(pybind11::buffer const &qr, pybind11::buffer const &tau) {
    py::buffer_info qr_info = qr.request(false), tau_info = tau.request(false);

    blas::int_t const m = qr_info.shape[1];
    blas::int_t const p = qr_info.shape[0];

    Tensor<T, 2> Q = PyTensorView<T>(qr);

    blas::int_t info;
    if constexpr (!IsComplexV<T>) {

        info = blas::orgqr(m, m, p, Q.data(), m, (T *)tau_info.ptr);
    } else {
        info = blas::ungqr(m, m, p, Q.data(), m, (T *)tau_info.ptr);
    }
    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "{} parameter to orgqr has an illegal value. {} {} {}", print::ordinal(-info), m, m, p);
    }

    return RuntimeTensor<T>(std::move(Q));
}

py::object q(pybind11::buffer const &qr, pybind11::buffer const &tau) {
    py::buffer_info qr_info = qr.request(false), tau_info = tau.request(false);

    if (qr_info.ndim != 2 || tau_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The q function takes a matrix and a vector!");
    }

    if (qr_info.format != tau_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The arguments to the q function need to store the same data types!");
    }

    py::object out;
    EINSUMS_PY_LINALG_CALL((qr_info.format == py::format_descriptor<Float>::format()), (out = py::cast(q_work<Float>(qr, tau))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The inputs to qr need to store real or complex floating point values!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums