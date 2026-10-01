//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>

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

template <typename T>
pybind11::tuple svd_work(pybind11::buffer const &A, char jobu, char jobvt) {
    auto [U, S, Vt] = einsums::linear_algebra::detail::svd(buffer_to_tensor<T>(A), jobu, jobvt);

    if (std::tolower(jobu) == 'n' && std::tolower(jobvt) == 'n') {
        return pybind11::make_tuple(std::move(RuntimeTensor<RemoveComplexT<T>>(std::move(S))));
    } else if (std::tolower(jobu) == 'n') {
        return pybind11::make_tuple(std::move(RuntimeTensor<RemoveComplexT<T>>(std::move(S))),
                                    std::move(RuntimeTensor<T>(std::move(Vt.value()))));
    } else if (std::tolower(jobvt) == 'n') {
        return pybind11::make_tuple(std::move(RuntimeTensor<T>(std::move(U.value()))),
                                    std::move(RuntimeTensor<RemoveComplexT<T>>(std::move(S))));
    } else {
        return pybind11::make_tuple(std::move(RuntimeTensor<T>(std::move(U.value()))),
                                    std::move(RuntimeTensor<RemoveComplexT<T>>(std::move(S))),
                                    std::move(RuntimeTensor<T>(std::move(Vt.value()))));
    }
}

pybind11::tuple svd(pybind11::buffer const &A, einsums::linear_algebra::Vectors jobu, einsums::linear_algebra::Vectors jobvt) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform svd on matrices!");
    }

    pybind11::tuple out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()),
                           (out = svd_work<Float>(A, static_cast<char>(jobu), static_cast<char>(jobvt))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the svd of real or complex floating point matrices!");
    }

    return out;
}

template <typename T>
RuntimeTensor<T> svd_nullspace_work(pybind11::buffer const &_A) {
    TensorView<T, 2> A{buffer_to_tensor<T>(_A)};

    return RuntimeTensor<T>{std::move(einsums::linear_algebra::svd_nullspace(A))};
}

pybind11::object svd_nullspace(pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only compute the nullspace of matrices!");
    }

    py::object out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), (out = py::cast(svd_nullspace_work<Float>(A))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the svd of real or complex floating point matrices!");
    }

    return out;
}

template <typename T>
pybind11::tuple svd_dd_work(pybind11::buffer const &A, linear_algebra::Vectors job) {
    auto [U, S, Vt] = einsums::linear_algebra::detail::svd_dd(buffer_to_tensor<T>(A), static_cast<char>(job));

    if (std::tolower(static_cast<char>(job)) == 'n') {
        return pybind11::make_tuple(std::move(RuntimeTensor<RemoveComplexT<T>>(std::move(S))));
    } else {
        return pybind11::make_tuple(std::move(RuntimeTensor<T>(std::move(U.value()))),
                                    std::move(RuntimeTensor<RemoveComplexT<T>>(std::move(S))),
                                    std::move(RuntimeTensor<T>(std::move(Vt.value()))));
    }
}

pybind11::tuple svd_dd(pybind11::buffer const &A, linear_algebra::Vectors job) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform svd_dd on matrices!");
    }

    py::tuple out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), out = svd_dd_work<Float>(A, job))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only perform svd_dd on real or complex floating point values!");
    }

    return out;
}

template <typename T>
pybind11::tuple truncated_svd_work(pybind11::buffer const &_A, size_t k) {
    Tensor<T, 2> A_view = PyTensorView<T>(_A);

    size_t m = A_view.dim(0);
    size_t n = A_view.dim(1);

    // Omega Test Matrix
    auto omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<T, 2> Y("Y", m, k + 5);
    linear_algebra::gemm<false, false>(T{1.0}, A_view, omega, T{0.0}, &Y);

    Tensor<T, 1> tau("tau", std::min(m, k + 5));
    // Compute QR factorization of Y
    int info1 = blas::geqrf(m, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    if constexpr (!IsComplexV<T>) {
        int info2 = blas::orgqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<T const *>(tau.data()));
    } else {
        int info2 = blas::ungqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<T const *>(tau.data()));
    }

    // Cast the matrix A into a smaller rank (B)
    Tensor<T, 2> B("B", k + 5, n);
    linear_algebra::gemm<true, false>(T{1.0}, Y, A_view, T{0.0}, &B);

    // Perform svd on B
    auto [Utilde, S, Vt] = linear_algebra::svd_dd(B);

    // Cast U back into full basis
    RuntimeTensor<T> U("U", {m, k + 5});
    TensorView<T, 2> U_view(U);
    linear_algebra::gemm<false, false>(T{1.0}, Y, Utilde.value(), T{0.0}, &U_view);

    return py::make_tuple(U, RuntimeTensor<RemoveComplexT<T>>(std::move(S)), RuntimeTensor<T>(std::move(Vt.value())));
}

pybind11::tuple truncated_svd(pybind11::buffer const &A, size_t k) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform truncated svd on matrices!");
    }

    py::tuple out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), out = truncated_svd_work<Float>(A, k))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only perform truncated_svd on real or complex floating point values!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums