//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Utilities/InCollection.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include "Einsums/LinearAlgebra/Base.hpp"
#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void syev(std::string const &jobz, py::buffer &A, py::buffer &W) {
    py::buffer_info A_info = A.request(true), W_info = W.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "A call to syev/heev can only take a rank-2 tensor as input!");
    }

    if (W_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "A call to syev/heev can only take a rank-1 tensor as output!");
    }

    if (jobz.length() == 0 || !strchr("NVnv", jobz.at(0))) {
        EINSUMS_THROW_EXCEPTION(
            py::value_error,
            "The job argument was invalid! Expected n or v, case insensitive as the first/only character. Got the string \"{}\".", jobz);
    }

    // Type check
    if (A_info.format == py::format_descriptor<float>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::value_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != A_info.format) {
            EINSUMS_THROW_EXCEPTION(py::value_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::value_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != A_info.format) {
            EINSUMS_THROW_EXCEPTION(py::value_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::value_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != py::format_descriptor<float>::format()) {
            EINSUMS_THROW_EXCEPTION(py::value_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::value_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != py::format_descriptor<double>::format()) {
            EINSUMS_THROW_EXCEPTION(py::value_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The tensors passed to syev/heev have the wrong storage types. Got A ({}), W ({}).",
                                A_info.format, W_info.format);
    }

    blas::int_t info;

    // Calculate the size of the work array.
    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(), [&]() {
        auto A_tens = buffer_to_tensor<Float>(A);
        auto W_tens = buffer_to_tensor<RemoveComplexT<Float>>(W);
        if (jobz.at(0) == 'n' || jobz.at(0) == 'N') {
            einsums::linear_algebra::detail::syev<false>(&A_tens, &W_tens);
        } else {
            einsums::linear_algebra::detail::syev<true>(&A_tens, &W_tens);
        }
    }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Could not handle the type of the buffer!");
    }
}

template <NotComplex T>
void geev_work(einsums::detail::TensorImpl<T> &A, einsums::detail::TensorImpl<AddComplexT<T>> &W,
               einsums::detail::TensorImpl<AddComplexT<T>> *vl, einsums::detail::TensorImpl<AddComplexT<T>> *vr) {
    Tensor<T, 2> vl_tens, vr_tens;

    einsums::detail::TensorImpl<T> *vl_impl = nullptr, *vr_impl = nullptr;

    if (vl != nullptr) {
        vl_tens = Tensor<T, 2>("left eigenvectors", vl->dim(0), vl->dim(1));
        vl_impl = &(vl_tens.impl());
    }

    if (vr != nullptr) {
        vr_tens = Tensor<T, 2>("right eigenvectors", vr->dim(0), vr->dim(1));
        vr_impl = &(vr_tens.impl());
    }

    einsums::linear_algebra::detail::geev(&A, &W, vl_impl, vr_impl);

    if (vl != nullptr || vr != nullptr) {
        einsums::linear_algebra::detail::process_geev_vectors(W, vl_impl, vr_impl, vl, vr);
    }
}

template <Complex T>
void geev_work(einsums::detail::TensorImpl<T> &A, einsums::detail::TensorImpl<T> &W, einsums::detail::TensorImpl<T> *vl,
               einsums::detail::TensorImpl<T> *vr) {

    einsums::linear_algebra::detail::geev(&A, &W, vl, vr);
}

template <typename T>
void geev_setup(py::buffer &A, py::buffer &W, pybind11::buffer *Vl_or_none, pybind11::buffer *Vr_or_none) {

    einsums::detail::TensorImpl<AddComplexT<T>> Vl_tens, Vr_tens, *Vl_ptr = nullptr, *Vr_ptr = nullptr;
    auto                                        A_tens = buffer_to_tensor<T>(A);
    auto                                        W_tens = buffer_to_tensor<AddComplexT<T>>(W);

    if (Vl_or_none != nullptr) {
        Vl_tens = buffer_to_tensor<AddComplexT<T>>(*Vl_or_none);
        Vl_ptr  = &Vl_tens;
    }

    if (Vr_or_none != nullptr) {
        Vr_tens = buffer_to_tensor<AddComplexT<T>>(*Vr_or_none);
        Vr_ptr  = &Vr_tens;
    }

    geev_work<T>(A_tens, W_tens, Vl_ptr, Vr_ptr);
}

void geev(py::buffer &A, py::buffer &W, std::variant<pybind11::buffer, pybind11::none> &Vl_or_none,
          std::variant<pybind11::buffer, pybind11::none> &Vr_or_none) {
    py::buffer_info A_info = A.request(false), W_info = W.request(false), Vl_info, Vr_info;

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform eigendecomposition on matrices!");
    }

    if (W_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The output of geev is a vector of eigenvalues!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can only perform eigendecomposition on square matrices!");
    }

    if (A_info.shape[0] != W_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                "The number of eigenvalues must be the same as the length along one dimension of the input matrix!");
    }

    if (Vl_or_none.index() == 0) {
        Vl_info = std::get<0>(Vl_or_none).request(true);

        if (Vl_info.ndim != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "The rank of the left eigenvector tensor needs to be 2!");
        }

        if (Vl_info.shape[0] != Vl_info.shape[1]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The left eigenvector tensor needs to be square!");
        }

        if (Vl_info.shape[0] != A_info.shape[0]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the left eigenvector tensor need to match the input matrix!");
        }
    }

    if (Vr_or_none.index() == 0) {
        Vr_info = std::get<0>(Vr_or_none).request(true);

        if (Vr_info.ndim != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "The rank of the right eigenvector tensor needs to be 2!");
        }

        if (Vr_info.shape[0] != Vr_info.shape[1]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The right eigenvector tensor needs to be square!");
        }

        if (Vr_info.shape[0] != A_info.shape[0]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the right eigenvector tensor need to match the input matrix!");
        }
    }

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(),
                           geev_setup<Float>(A, W, (Vl_or_none.index() == 0) ? &std::get<0>(Vl_or_none) : nullptr,
                                             (Vr_or_none.index() == 0) ? &std::get<0>(Vr_or_none) : nullptr))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Could not handle the types passed to geev!");
    }
}

template <typename T>
pybind11::tuple truncated_syev_work(pybind11::buffer const &_A, size_t k) {
    Tensor<T, 2> A = PyTensorView<T>(_A);

    size_t n = A.dim(0);

    // Omega Test Matrix
    Tensor<T, 2> omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<T, 2> Y("Y", n, k + 5);
    linear_algebra::gemm<false, false>(T{1.0}, A, omega, T{0.0}, &Y);

    Tensor<T, 1> tau("tau", std::min(n, k + 5));
    // Compute QR factorization of Y
    blas::int_t const info1 = blas::geqrf(n, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    blas::int_t info2;
    if constexpr (IsComplexV<T>) {
        info2 = blas::ungqr(n, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<T const *>(tau.data()));
    } else {
        info2 = blas::orgqr(n, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<T const *>(tau.data()));
    }
    Tensor<T, 2> &Q1 = Y;

    // Cast the matrix A into a smaller rank (B)
    // B = Q^T * A * Q
    Tensor<T, 2> Btemp("Btemp", k + 5, n);
    linear_algebra::gemm<true, false>(1.0, Q1, A, 0.0, &Btemp);
    Tensor<T, 2> B("B", k + 5, k + 5);
    linear_algebra::gemm<false, false>(1.0, Btemp, Q1, 0.0, &B);

    // Create buffer for eigenvalues
    Tensor<RemoveComplexT<T>, 1> w("eigenvalues", k + 5);

    // Diagonalize B
    if constexpr (IsComplexV<T>) {
        linear_algebra::heev(&B, &w);
    } else {
        linear_algebra::syev(&B, &w);
    }

    // Cast U back into full basis (B is column-major so we need to transpose it)
    Tensor<T, 2> U("U", n, k + 5);
    linear_algebra::gemm<false, true>(1.0, Q1, B, 0.0, &U);

    return py::make_tuple(PyTensor<T>(std::move(U)), PyTensor<RemoveComplexT<T>>(std::move(w)));
}

pybind11::tuple truncated_syev(pybind11::buffer const &A, size_t k) {
    py::buffer_info A_info = A.request(false);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only take the truncated_syev of matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only take the truncated_syev of square matrices!");
    }

    py::tuple out;

    EINSUMS_PY_LINALG_CALL((A_info.format == py::format_descriptor<Float>::format()), (out = truncated_syev_work<Float>(A, k)))
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The input matrix needs to store real or complex floating point data!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums