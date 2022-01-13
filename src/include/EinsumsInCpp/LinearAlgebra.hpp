#pragma once

#include "Blas.hpp"
#include "STL.hpp"
#include "Tensor.hpp"
#include "Timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace EinsumsInCpp::LinearAlgebra {

template <bool TransA, bool TransB, typename AType, typename BType, typename CType>
auto gemm(const double alpha, const AType &A, const BType &B, const double beta, CType *C) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2> && is_incore_rank_tensor_v<BType, 2> && is_incore_rank_tensor_v<CType, 2>> {
    auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    Timer::push("gemm");
    Blas::dgemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C->data(), ldc);
    Timer::pop();
}

/**
 * @brief Version of gemm that returns a new tensor object.
 *
 * @tparam TransA
 * @tparam TransB
 * @tparam AType
 * @tparam BType
 * @param alpha
 * @param A
 * @param B
 * @param beta
 *
 * @return std::enable_if_t<std::is_base_of_v<Detail::TensorBase<2, double>, AType> &&
 * std::is_base_of_v<Detail::TensorBase<2, double>, BType>,
 * Tensor<2, double>>
 *
 */
template <bool TransA, bool TransB, typename AType, typename BType>
auto gemm(const double alpha, const AType &A, const BType &B) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2> && is_incore_rank_tensor_v<BType, 2>, Tensor<2, double>> {
    Tensor<2, double> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(1) : B.dim(0)};

    gemm<TransA, TransB>(alpha, A, B, 0.0, &C);

    return C;
}

template <bool TransA, typename AType, typename XType, typename YType>
auto gemv(const double alpha, const AType &A, const XType &x, const double beta, YType *y) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2> && is_incore_rank_tensor_v<XType, 1> && is_incore_rank_tensor_v<YType, 1>> {
    auto m = A.dim(0), n = A.dim(1);
    auto lda = A.stride(0);
    auto incx = x.stride(0);
    auto incy = y->stride(0);

    Timer::push("gemv");
    Blas::dgemv(TransA ? 't' : 'n', m, n, alpha, A.data(), lda, x.data(), incx, beta, y->data(), incy);
    Timer::pop();
}

template <typename AType, typename WType, bool ComputeEigenvectors = true>
auto syev(AType *A, WType *W) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2> && is_incore_rank_tensor_v<WType, 1>> {
    assert(A->dim(0) == A->dim(1));

    auto n = A->dim(0);
    auto lda = A->stride(0);
    int lwork = 3 * n;
    std::vector<double> work(lwork);

    Timer::push("syev");
    Blas::dsyev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
    Timer::pop();
}

template <typename AType, typename BType>
auto gesv(AType *A, BType *B) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2> && is_incore_rank_tensor_v<BType, 2>, int> {
    auto n = A->dim(0);
    auto lda = A->stride(0);
    auto ldb = B->dim(0);

    auto nrhs = B->dim(1);

    int lwork = n;
    std::vector<int> ipiv(lwork);

    Timer::push("gesv");
    int info = Blas::dgesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    Timer::pop();
    return info;
}

template <typename AType, bool ComputeEigenvectors = true>
auto syev(const AType &A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2>, std::tuple<Tensor<2, double>, Tensor<1, double>>> {
    assert(A.dim(0) == A.dim(1));

    Tensor<2, double> a = A;
    Tensor<1, double> w{"eigenvalues", A.dim(0)};

    syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

template <template <size_t, typename> typename AType, size_t Rank, typename T = double>
auto scale(double scale, AType<Rank, T> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<Rank, T>, Rank, double>> {
    Timer::push("scal");
    Blas::dscal(A->dim(0) * A->stride(0), scale, A->data(), 1);
    Timer::pop();
}

template <typename AType>
auto scale_row(size_t row, double scale, AType *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2, double>> {
    Blas::dscal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <typename AType>
auto scale_column(size_t col, double scale, AType *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2, double>> {
    Blas::dscal(A->dim(0), scale, A->data(0ul, col), A->stride(0));
}

/**
 * @brief Computes the matrix power of a to alpha.  Return a new tensor, does not destroy a.
 *
 * @tparam AType
 * @param a Matrix to take power of
 * @param alpha The power to take
 * @param cutoff Values below cutoff are considered zero.
 *
 * @return std::enable_if_t<std::is_base_of_v<Detail::TensorBase<2, double>, AType>, AType>
 *
 */
template <typename AType>
auto pow(const AType &a, double alpha, double cutoff = std::numeric_limits<double>::epsilon()) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2>, AType> {
    assert(a.dim(0) == a.dim(1));

    size_t n = a.dim(0);
    Tensor<2> a1 = a;
    Tensor<2> result{"pow result", a.dim(0), a.dim(1)};
    Tensor<1> e{"e", n};

    // Diagonalize
    syev(&a1, &e);

    Tensor<2> a2 = a1;

    // Determine the largest magnitude of the eigenvalues to use as a scaling factor for the cutoff.
    double max_e = std::fabs(e(n - 1)) > std::fabs(e(0)) ? std::fabs(e(n - 1)) : std::fabs(e(0));

    for (size_t i = 0; i < n; i++) {
        if (alpha < 0.0 && std::fabs(e(i)) < cutoff * max_e) {
            e(i) = 0.0;
        } else {
            e(i) = std::pow(e(i), alpha);
            if (!std::isfinite(e(i))) {
                e(i) = 0.0;
            }
        }

        scale_row(i, e(i), &a2);
    }

    gemm<true, false>(1.0, a2, a1, 0.0, &result);

    return result;
}

template <template <size_t, typename> typename Type>
auto dot(const Type<1, double> &A, const Type<1, double> &B) ->
    typename std::enable_if_t<std::is_base_of_v<Detail::TensorBase<1, double>, Type<1, double>>, double> {
    assert(A.dim(0) == B.dim(0));

    Timer::push("dot");
    auto result = Blas::ddot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    Timer::pop();
    return result;
}

template <template <size_t, typename> typename Type, size_t Rank>
auto dot(const Type<Rank, double> &A, const Type<Rank, double> &B) ->
    typename std::enable_if_t<std::is_base_of_v<Detail::TensorBase<Rank, double>, Type<Rank, double>>, double> {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i));
        dim[0] *= A.dim(i);
    }

    return dot(TensorView<1>(const_cast<Type<Rank, double> &>(A), dim), TensorView<1>(const_cast<Type<Rank, double> &>(B), dim));
}

template <template <size_t, typename> typename Type, size_t Rank>
auto dot(const Type<Rank, double> &A, const Type<Rank, double> &B, const Type<Rank, double> &C) ->
    typename std::enable_if_t<std::is_base_of_v<Detail::TensorBase<Rank, double>, Type<Rank, double>>, double> {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
        dim[0] *= A.dim(i);
    }

    auto vA = TensorView<1>(const_cast<Type<Rank, double> &>(A), dim);
    auto vB = TensorView<1>(const_cast<Type<Rank, double> &>(B), dim);
    auto vC = TensorView<1>(const_cast<Type<Rank, double> &>(C), dim);

    Timer::push("dot3");
    double result = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < dim[0]; i++) {
        result += vA(i) * vB(i) * vC(i);
    }
    Timer::pop();
    return result;
}

template <template <size_t, typename> typename XType, template <size_t, typename> typename YType, size_t Rank>
auto axpy(double alpha, const XType<Rank, double> &X, YType<Rank, double> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<Rank, double>, Rank, double> &&
                        is_incore_rank_tensor_v<YType<Rank, double>, Rank, double>> {
    Timer::push("axpy");
    Blas::daxpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
    Timer::pop();
}

template <template <size_t, typename> typename XYType, size_t XYRank, template <size_t, typename> typename AType, size_t ARank>
auto ger(double alpha, const XYType<XYRank, double> &X, const XYType<XYRank, double> &Y, AType<ARank, double> *A)
    -> std::enable_if_t<is_incore_rank_tensor_v<XYType<XYRank, double>, 2> && is_incore_rank_tensor_v<AType<ARank, double>, 2>, double> {
    Timer::push("ger");
    Blas::dger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
    Timer::pop();
}

} // namespace EinsumsInCpp::LinearAlgebra