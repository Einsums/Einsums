//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Einsums/TensorAlgebra/Detail/Index.hpp"
#include "Einsums/TensorAlgebra/Permute.hpp"
#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

template <typename T>
py::tuple svd_work(pybind11::buffer const &A) {
    PyTensor<T> A_copy = A;

    size_t m   = A_copy.dim(0);
    size_t n   = A_copy.dim(1);
    size_t lda = A_copy.stride(0);

    // Test if it is absolutely necessary to zero out these tensors first.
    auto U = RuntimeTensor<T>("U (stored columnwise)", {m, m});
    U.zero();
    auto S = RuntimeTensor<RemoveComplexT<T>>("S", {std::min(m, n)});
    S.zero();
    auto Vt = RuntimeTensor<T>("Vt (stored rowwise)", {n, n});
    Vt.zero();
    auto superb = RuntimeTensor<T>("superb", {std::min(m, n)});
    superb.zero();

    //    int info{0};
    int info = blas::gesvd('A', 'A', m, n, A_copy.data(), lda, S.data(), U.data(), m, Vt.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(py::value_error,
                                    "svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info,
                                    m, n, n, m);
        } else {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "SVD iterations did not converge!");
        }
    }

    return py::make_tuple(U, S, Vt);
}

pybind11::tuple svd(pybind11::buffer const &A) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform svd on matrices!");
    }

    pybind11::tuple out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), (out = std::move(svd_work<Float>(A))))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only take the svd of real or complex floating point matrices!");
    }

    return out;
}

template <typename T>
RuntimeTensor<T> svd_nullspace_work(pybind11::buffer const &_A) {
    // Calling svd will destroy the original data. Make a copy of it.
    PyTensor<T> A = _A;

    blas::int_t m   = A.dim(0);
    blas::int_t n   = A.dim(1);
    blas::int_t lda = A.stride(0);

    EINSUMS_LOG_DEBUG("Making tensors.");

    auto U = create_tensor<T>("U", m, m);
    zero(U);
    auto S = create_tensor<RemoveComplexT<T>>("S", n);
    zero(S);
    auto V = create_tensor<T>("V", n, n);
    zero(V);
    auto superb = create_tensor<T>("superb", std::min(m, n));

    int info = blas::gesvd('N', 'A', m, n, A.data(), lda, S.data(), U.data(), m, V.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(py::value_error,
                                    "svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info,
                                    m, n, n, m);
        } else {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "SVD could not converge!");
        }
    }

    // Determine the rank of the nullspace matrix
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (std::abs(S(i)) > 1e-12) {
            rank++;
        }
    }

    // println("rank {}", rank);

    auto Vview          = V(Range{rank, V.dim(0)}, All);
    auto nullspace      = RuntimeTensor<T>("Nullspace", {Vview.dim(1), Vview.dim(0)});
    auto nullspace_view = (TensorView<T, 2>)nullspace;

    if (nullspace.dim(0) == 0 || nullspace.dim(1) == 0) {
        return nullspace;
    }

    tensor_algebra::permute<true>(T{0.0}, Indices{index::i, index::j}, &nullspace_view, T{1.0}, Indices{index::j, index::i}, Vview);

    // for (size_t i = 0; i < nullspace.dim(0); i++) {
    //     for (size_t j = 0; j < nullspace.dim(1); j++) {
    //         if constexpr (IsComplexV<T>) {
    //             nullspace(i, j) = std::conj(Vview(j, i));
    //         } else {
    //             nullspace(i, j) = Vview(j, i);
    //         }
    //     }
    // }

    // Normalize nullspace. LAPACK does not guarentee them to be orthonormal
    for (int i = 0; i < nullspace_view.dim(1); i++) {
        // Make the first non-zero element positive real.
        T rescale{0.0};
        for (int j = 0; j < nullspace_view.dim(0); j++) {
            rescale = nullspace_view(j, i);

            if (std::abs(rescale) > 1e-12) {
                break;
            }
        }

        rescale /= std::abs(rescale);

        // Normalize
        T norm{0.0};

        for (int j = 0; j < nullspace_view.dim(0); j++) {
            norm += std::norm(nullspace_view(j, i));
        }

        norm = T{1.0} / (norm * rescale);
        linear_algebra::scale_column(i, norm, &nullspace_view);
    }

    return nullspace;
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
pybind11::tuple svd_dd_work(pybind11::buffer const &_A, linear_algebra::Vectors job) {

    //    DisableOMPThreads const nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = PyTensorView<T>(_A);

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    zero(U);
    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    zero(S);
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    zero(Vt);

    int info = blas::gesdd(static_cast<char>(job), static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(n), S.data(),
                           U.data(), static_cast<int>(m), Vt.data(), static_cast<int>(n));

    if (info != 0) {
        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "svd_dd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}",
                                    -info, m, n, n, m);
        } else {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "svd_dd did not converge!");
        }
    }

    return py::make_tuple(RuntimeTensor<T>(std::move(U)), RuntimeTensor<RemoveComplexT<T>>(std::move(S)), RuntimeTensor<T>(std::move(Vt)));
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
    linear_algebra::gemm<false, false>(T{1.0}, Y, Utilde, T{0.0}, &U_view);

    return py::make_tuple(U, RuntimeTensor<RemoveComplexT<T>>(std::move(S)), RuntimeTensor<T>(std::move(Vt)));
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