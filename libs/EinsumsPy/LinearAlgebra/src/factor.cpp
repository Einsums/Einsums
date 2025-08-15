//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

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

std::vector<blas::int_t> getrf(pybind11::buffer &A) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU decomposition on matrices!");
    }

    std::vector<blas::int_t> pivot(std::min(A_info.shape[0], A_info.shape[1]));

    int result = 0;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        result      = einsums::linear_algebra::detail::getrf(&A_tens, &pivot);
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only decompose matrices of real or complex floating point values!");
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument had an illegal value!", print::ordinal(-result));
    } else if (result > 0) {
        EINSUMS_LOG_WARN("The system was factorized, but the the matrix was singular. The {} value is zero.", print::ordinal(result));
    }

    return pivot;
}

template <typename T>
py::tuple extract_plu_work(pybind11::buffer const &A, std::vector<blas::int_t> const &pivot) {
    py::buffer_info A_info = A.request(false);

    size_t m = A_info.shape[0], n = A_info.shape[1];
    size_t P_rows = m, P_cols = m;
    size_t L_rows = m, L_cols = std::min(m, n);
    size_t U_rows = std::min(m, n), U_cols = n;

    RuntimeTensor<T> P("Pivot", {P_rows, P_cols}), L("Lower Triangle", {L_rows, L_cols}), U("Upper Triangle", {U_rows, U_cols});

    P.zero();
    L.zero();
    U.zero();

    // Set up the diagonal elements of L.
    for (size_t i = 0; i < L_cols; i++) {
        L(i, i) = T{1.0};
    }

    // Set up P. First, set it to the identity matrix.
    for (size_t i = 0; i < P_rows; i++) {
        P(i, i) = T{1.0};
    }

    // Now, go through and pivot.
    for (size_t i = 0; i < pivot.size(); i++) {
        for (size_t j = 0; j < P_cols; j++) {
            std::swap(P(i, j), P(pivot[i] - 1, j));
        }
    }

    T const *A_data    = (T const *)A_info.ptr;
    size_t   A_stride0 = A_info.strides[0] / A_info.itemsize, A_stride1 = A_info.strides[1] / A_info.itemsize;

    // Now, set up L.
    for (size_t i = 0; i < L_rows; i++) {
        for (size_t j = 0; j < i && j < L_cols; j++) {
            L(i, j) = A_data[i * A_stride0 + j * A_stride1];
        }
    }

    // Finally, set up U.
    for (size_t i = 0; i < U_rows; i++) {
        for (size_t j = i; j < U_cols; j++) {
            U(i, j) = A_data[i * A_stride0 + j * A_stride1];
        }
    }

    return py::make_tuple(P, L, U);
}

pybind11::tuple extract_plu(pybind11::buffer const &A, std::vector<blas::int_t> const &pivot) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU decomposition on matrices!");
    }

    if (pivot.size() != std::min(A_info.shape[0], A_info.shape[1])) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The pivot list does not have the correct size! Did you run getrf first?");
    }

    py::tuple out;

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(), out = extract_plu_work<Float>(A, pivot))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The input to extract_plu needs to store real or complex floating point data!");
    }

    return out;
}

void getri(pybind11::buffer &A, std::vector<blas::int_t> &pivot) {
    py::buffer_info A_info = A.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU inversion on matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only take the inverse of a square matrix!");
    }

    if (pivot.size() != A_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The pivot list has not been initialized! Have you performed getrf on this matrix first?");
    }

    int result = 0;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        result      = einsums::linear_algebra::detail::getri(&A_tens, pivot);
    }())
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

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform LU inversion on matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only take the inverse of a square matrix!");
    }

    std::vector<blas::int_t> pivot = getrf(A);

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

    blas::int_t const m   = qr_info.shape[0];
    blas::int_t const n   = std::min(qr_info.shape[0], qr_info.shape[1]);
    blas::int_t const k   = tau_info.shape[0];
    blas::int_t const lda = qr_info.strides[0] / qr_info.itemsize;

    Tensor<T, 2> Q = PyTensorView<T>(qr);

    blas::int_t info;
    if constexpr (!IsComplexV<T>) {
        info = blas::orgqr(m, n, k, Q.data(), lda, (T *)tau_info.ptr);
    } else {
        info = blas::ungqr(m, n, k, Q.data(), lda, (T *)tau_info.ptr);
    }

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "{} parameter to orgqr has an illegal value. m = {}, n = {}, k = {}, lda = {}",
                                print::ordinal(-info), m, n, k, lda);
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
        EINSUMS_THROW_EXCEPTION(py::value_error, "The inputs to q need to store real or complex floating point values!");
    }

    return out;
}

template <typename T>
RuntimeTensor<T> r_work(pybind11::buffer const &qr) {
    py::buffer_info qr_info = qr.request(false);

    RuntimeTensor<T> out("R", {(size_t)std::min(qr_info.shape[0], qr_info.shape[1]), (size_t)qr_info.shape[1]});

    out.zero();

    T const *qr_data = (T const *)qr_info.ptr;

    size_t qr_stride0 = qr_info.strides[0] / qr_info.itemsize, qr_stride1 = qr_info.strides[1] / qr_info.itemsize;

    for (size_t i = 0; i < out.dim(0); i++) {
        for (size_t j = i; j < out.dim(1); j++) {
            out(i, j) = qr_data[i * qr_stride0 + j * qr_stride1];
        }
    }

    return out;
}

py::object r(pybind11::buffer const &qr, pybind11::buffer const &tau) {
    py::buffer_info qr_info = qr.request(false), tau_info = tau.request(false);

    if (qr_info.ndim != 2 || tau_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The q function takes a matrix and a vector!");
    }

    py::object out;
    EINSUMS_PY_LINALG_CALL((qr_info.format == py::format_descriptor<Float>::format()), (out = py::cast(r_work<Float>(qr))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The inputs to r need to store real or complex floating point values!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums