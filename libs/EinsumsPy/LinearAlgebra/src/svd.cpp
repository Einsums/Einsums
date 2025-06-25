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

    auto U = create_tensor<T>("U", m, m);
    zero(U);
    auto S = create_tensor<RemoveComplexT<T>>("S", n);
    zero(S);
    auto V = create_tensor<T>("V", n, n);
    zero(V);
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);

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
    auto nullspace      = RuntimeTensor(V);
    auto nullspace_view = (TensorView<T, 2>)nullspace;

    // Normalize nullspace. LAPACK does not guarentee them to be orthonormal
    for (int i = 0; i < nullspace.dim(0); i++) {
        T sum{0};
        for (int j = 0; j < nullspace.dim(1); j++) {
            sum += std::pow(nullspace(i, j), 2.0);
        }
        sum = std::sqrt(sum);
        linear_algebra::scale_row(i, sum, &nullspace_view);
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

} // namespace detail
} // namespace python
} // namespace einsums