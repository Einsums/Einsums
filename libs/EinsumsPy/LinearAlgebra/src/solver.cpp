#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor.hpp>
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
void gesv_work(pybind11::buffer &A, pybind11::buffer &B) {
    py::buffer_info A_info = A.request(true), B_info = B.request(true);
    Tensor<T, 2>    A_T{"A Transpose", A_info.shape[1], A_info.shape[0]},
        B_T{"B Transpose", (B_info.ndim == 1) ? 1 : B_info.shape[1], B_info.shape[0]};
    TensorView<T, 2> A_view{(T *)A_info.ptr, Dim{A_info.shape[0], A_info.shape[1]},
                            Stride{A_info.strides[0] / A_info.itemsize, A_info.strides[1] / A_info.itemsize}};
    TensorView<T, 2> B_view{(T *)B_info.ptr, Dim{B_info.shape[0], (B_info.ndim == 1) ? 1 : B_info.shape[1]},
                            (B_info.ndim == 1) ? Stride{B_info.strides[0] * B_info.shape[0] / B_info.itemsize, 1}
                                               : Stride{B_info.strides[0] / B_info.itemsize, B_info.strides[1] / B_info.itemsize}};

    // Convert to column major.
    tensor_algebra::permute(Indices{index::i, index::j}, &A_T, Indices{index::j, index::i}, A_view);
    tensor_algebra::permute(Indices{index::i, index::j}, &B_T, Indices{index::j, index::i}, B_view);

    // Solve.
    int info = linear_algebra::gesv(&A_T, &B_T);

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The input matrix is singular!");
    }

    // Convert back to row major.
    tensor_algebra::permute(Indices{index::i, index::j}, &A_view, Indices{index::j, index::i}, A_T);
    tensor_algebra::permute(Indices{index::i, index::j}, &B_view, Indices{index::j, index::i}, B_T);
}

void gesv(pybind11::buffer &A, pybind11::buffer &B) {
    py::buffer_info A_info = A.request(true), B_info = B.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The coefficient matrix needs to be a matrix!");
    }

    if (B_info.ndim != 1 && B_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The constant matrix needs to be a vector or a matrix!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The solver can only handle square matrices!");
    }

    if (A_info.shape[0] != B_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The rows of A and B must match!");
    }

    if (A_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the input matrices need to be the same!");
    } else if (A_info.format == py::format_descriptor<float>::format()) {
        gesv_work<float>(A, B);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        gesv_work<double>(A, B);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        gesv_work<std::complex<float>>(A, B);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        gesv_work<std::complex<double>>(A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage type of the input to the solver is invalid!");
    }
}

template <typename T>
RuntimeTensor<T> solve_continuous_lyapunov_work(pybind11::buffer const &_A, pybind11::buffer const &_Q) {
    if constexpr (!IsComplexV<T>) {
        Tensor<T, 2> R = PyTensorView<T>(_A), Q = PyTensorView<T>(_Q);

        size_t n = R.dim(0);

        //// @todo Break this off into a separate schur function
        // Compute Schur Decomposition of A
        Tensor<T, 2>             wr("Schur Real Buffer", n, n);
        Tensor<T, 2>             wi("Schur Imaginary Buffer", n, n);
        Tensor<T, 2>             U("Lyapunov U", n, n);
        std::vector<blas::int_t> sdim(1);
        blas::int_t              info = blas::gees('V', n, R.data(), n, sdim.data(), wr.data(), wi.data(), U.data(), n);

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to gees was invalid!", print::ordinal(-info));
        } else if (info > 0) {
            if (info <= n) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "The QR algorithm failed to compute all the eigenvalues!");
            } else if (info == n + 1) {
                EINSUMS_LOG_WARN("The eigenvalues could not be reordered.");
            } else if (info == n + 2) {
                EINSUMS_LOG_INFO("After reordering, some eigenvalues changed due to roundoff error!");
            } else {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "gees failed for an unknown reason!");
            }
        }

        // Compute F = U^T * Q * U
        Tensor<T, 2> Fbuff = linear_algebra::gemm<true, false>(1.0, U, Q);
        Tensor<T, 2> F     = linear_algebra::gemm<false, false>(1.0, Fbuff, U);

        // Call the Sylvester Solve
        std::vector<RemoveComplexT<T>> scale(1);
        info = blas::trsyl('N', 'N', 1, n, n, const_cast<T const *>(R.data()), n, const_cast<T const *>(R.data()), n, F.data(), n,
                           scale.data());

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to trsyl was invalid!", print::ordinal(-info));
        } else if (info > 0) {
            EINSUMS_LOG_INFO("The matrices passed to trsyl have common or close eigenvalues. Solving perturbativeley")
        }

        Tensor<T, 2> Xbuff = linear_algebra::gemm<false, false>(scale[0], U, F);
        Tensor<T, 2> X     = linear_algebra::gemm<false, true>(1.0, Xbuff, U);

        return RuntimeTensor<T>(std::move(X));
    } else {
        Tensor<T, 2> R = PyTensorView<T>(_A), Q = PyTensorView<T>(_Q);

        size_t n = R.dim(0);

        //// @todo Break this off into a separate schur function
        // Compute Schur Decomposition of A
        Tensor<T, 2>             w("Schur Buffer", n, n);
        Tensor<T, 2>             U("Lyapunov U", n, n), Uh("Lyapunov U hermitian transpose", n, n);
        std::vector<blas::int_t> sdim(1);
        blas::int_t              info = blas::gees('V', n, R.data(), n, sdim.data(), w.data(), U.data(), n);

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to gees was invalid!", print::ordinal(-info));
        } else if (info > 0) {
            if (info <= n) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "The QR algorithm failed to compute all the eigenvalues!");
            } else if (info == n + 1) {
                EINSUMS_LOG_WARN("The eigenvalues could not be reordered.");
            } else if (info == n + 2) {
                EINSUMS_LOG_INFO("After reordering, some eigenvalues changed due to roundoff error!");
            } else {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "gees failed for an unknown reason!");
            }
        }

        // Compute F = U^T * Q * U
        einsums::tensor_algebra::permute<true>(Indices{index::i, index::j}, &Uh, Indices{index::j, index::i}, U);
        Tensor<T, 2> Fbuff = linear_algebra::gemm<false, false>(1.0, Uh, Q);
        Tensor<T, 2> F     = linear_algebra::gemm<false, false>(1.0, Fbuff, U);

        // Call the Sylvester Solve
        std::vector<RemoveComplexT<T>> scale(1);
        info = blas::trsyl('N', 'N', 1, n, n, const_cast<T const *>(R.data()), n, const_cast<T const *>(R.data()), n, F.data(), n,
                           scale.data());

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to trsyl was invalid!", print::ordinal(-info));
        } else if (info > 0) {
            EINSUMS_LOG_INFO("The matrices passed to trsyl have common or close eigenvalues. Solving perturbativeley")
        }

        Tensor<T, 2> Xbuff = linear_algebra::gemm<false, false>(scale[0], U, F);
        Tensor<T, 2> X     = linear_algebra::gemm<false, false>(1.0, Xbuff, Uh);

        return RuntimeTensor<T>(std::move(X));
    }
}

py::object solve_continuous_lyapunov(pybind11::buffer const &A, pybind11::buffer const &Q) {
    py::buffer_info A_info = A.request(false), Q_info = Q.request(false);

    if (A_info.ndim != 2 || Q_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only solve continuous Lyapunov with matrices!");
    }

    if (A_info.shape[0] != A_info.shape[1] || A_info.shape[0] != Q_info.shape[0] || A_info.shape[0] != Q_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(dimension_error,
                                "The arguments to solve_continuous_lyapunove need to be square matrices and have the same size!");
    }

    if (A_info.format != Q_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The storage types of the input matrices need to be the same!");
    }

    py::object out;
    if (A_info.format == py::format_descriptor<float>::format()) {
        return py::cast(solve_continuous_lyapunov_work<float>(A, Q));
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        return py::cast(solve_continuous_lyapunov_work<double>(A, Q));
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        return py::cast(solve_continuous_lyapunov_work<std::complex<float>>(A, Q));
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        return py::cast(solve_continuous_lyapunov_work<std::complex<float>>(A, Q));
    } else {
        EINSUMS_THROW_EXCEPTION(py::value_error,
                                "The inputs to solve_continuous_lyapunov need to store real or complex floating point values!");
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums