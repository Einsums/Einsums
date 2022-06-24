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
#include <tuple>
#include <type_traits>

namespace einsums::linear_algebra {

// template <size_t, typename> typename AType, size_t Rank, typename T

template <bool TransA, bool TransB, template <size_t, typename> typename AType, template <size_t, typename> typename BType,
          template <size_t, typename> typename CType, size_t Rank, typename T>
auto gemm(const T alpha, const AType<Rank, T> &A, const BType<Rank, T> &B, const T beta, CType<Rank, T> *C) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<Rank, T>, 2, T> && is_incore_rank_tensor_v<BType<Rank, T>, 2, T> &&
                              is_incore_rank_tensor_v<CType<Rank, T>, 2, T>> {
    auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    timer::push(fmt::format("gemm<{}, {}>", TransA, TransB));
    blas::gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C->data(), ldc);
    timer::pop();
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
template <bool TransA, bool TransB, template <size_t, typename> typename AType, template <size_t, typename> typename BType, size_t Rank,
          typename T>
auto gemm(const T alpha, const AType<Rank, T> &A, const BType<Rank, T> &B)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<Rank, T>, 2, T> && is_incore_rank_tensor_v<BType<Rank, T>, 2, T>, Tensor<2, double>> {
    Tensor<2, double> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(1) : B.dim(0)};

    gemm<TransA, TransB>(alpha, A, B, 0.0, &C);

    return C;
}

template <bool TransA, template <size_t, typename> typename AType, template <size_t, typename> typename XType,
          template <size_t, typename> typename YType, size_t ARank, size_t XYRank, typename T>
auto gemv(const double alpha, const AType<ARank, T> &A, const XType<XYRank, T> &x, const double beta, YType<XYRank, T> *y)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, T>, 2, T> && is_incore_rank_tensor_v<XType<XYRank, T>, 1, T> &&
                        is_incore_rank_tensor_v<YType<XYRank, T>, 1, T>> {
    auto m = A.dim(0), n = A.dim(1);
    auto lda = A.stride(0);
    auto incx = x.stride(0);
    auto incy = y->stride(0);

    timer::push(fmt::format("gemv<{}>", TransA));
    blas::dgemv(TransA ? 't' : 'n', m, n, alpha, A.data(), lda, x.data(), incx, beta, y->data(), incy);
    timer::pop();
}

template <template <size_t, typename> typename AType, size_t ARank, template <size_t, typename> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
auto syev(AType<ARank, T> *A, WType<WRank, T> *W)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, T>, 2, T> && is_incore_rank_tensor_v<WType<WRank, T>, 1, T>> {
    assert(A->dim(0) == A->dim(1));

    auto n = A->dim(0);
    auto lda = A->stride(0);
    int lwork = 3 * n;
    std::vector<double> work(lwork);

    timer::push(fmt::format("syev<ComputeEigenvectors={}>", ComputeEigenvectors));
    blas::dsyev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
    timer::pop();
}

// This assumes column-major ordering!!
template <template <size_t, typename> typename AType, size_t ARank, template <size_t, typename> typename BType, size_t BRank, typename T>
auto gesv(AType<ARank, T> *A, BType<BRank, T> *B)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, T>, 2, T> && is_incore_rank_tensor_v<BType<BRank, T>, 2, T>, int> {
    auto n = A->dim(0);
    auto lda = A->dim(0);
    auto ldb = B->dim(1);

    auto nrhs = B->dim(0);

    int lwork = n;
    std::vector<int> ipiv(lwork);

    timer::push("gesv");
    int info = blas::dgesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    timer::pop();
    return info;
}

template <template <size_t, typename> typename AType, size_t ARank, typename T, bool ComputeEigenvectors = true>
auto syev(const AType<ARank, T> &A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, T>, 2, T>, std::tuple<Tensor<2, T>, Tensor<1, T>>> {
    assert(A.dim(0) == A.dim(1));

    Tensor<2, double> a = A;
    Tensor<1, double> w{"eigenvalues", A.dim(0)};

    syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

template <template <size_t, typename> typename AType, size_t ARank, typename T = double>
auto scale(double scale, AType<ARank, T> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, T>, ARank, double>> {
    timer::push("scal");
    blas::dscal(A->dim(0) * A->stride(0), scale, A->data(), 1);
    timer::pop();
}

template <typename AType>
auto scale_row(size_t row, double scale, AType *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2, double>> {
    blas::dscal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <typename AType>
auto scale_column(size_t col, double scale, AType *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType, 2, double>> {
    blas::dscal(A->dim(0), scale, A->data(0ul, col), A->stride(0));
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
template <template <size_t, typename> typename AType, size_t ARank, typename T>
auto pow(const AType<ARank, T> &a, T alpha, T cutoff = std::numeric_limits<T>::epsilon()) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, T>, 2, T>, AType<ARank, T>> {
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
    typename std::enable_if_t<std::is_base_of_v<detail::TensorBase<1, double>, Type<1, double>>, double> {
    assert(A.dim(0) == B.dim(0));

    timer::push("dot");
    auto result = blas::ddot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    timer::pop();
    return result;
}

template <template <size_t, typename> typename Type, size_t Rank>
auto dot(const Type<Rank, double> &A, const Type<Rank, double> &B) ->
    typename std::enable_if_t<std::is_base_of_v<detail::TensorBase<Rank, double>, Type<Rank, double>>, double> {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i));
        dim[0] *= A.dim(i);
    }

    return dot(TensorView<1>(const_cast<Type<Rank, double> &>(A), dim), TensorView<1>(const_cast<Type<Rank, double> &>(B), dim));
}

template <template <size_t, typename> typename Type, size_t Rank>
auto dot(const Type<Rank, double> &A, const Type<Rank, double> &B, const Type<Rank, double> &C) -> // NOLINT
    typename std::enable_if_t<std::is_base_of_v<detail::TensorBase<Rank, double>, Type<Rank, double>>, double> {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
        dim[0] *= A.dim(i);
    }

    auto vA = TensorView<1>(const_cast<Type<Rank, double> &>(A), dim);
    auto vB = TensorView<1>(const_cast<Type<Rank, double> &>(B), dim);
    auto vC = TensorView<1>(const_cast<Type<Rank, double> &>(C), dim);

    timer::push("dot3");
    double result = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < dim[0]; i++) {
        result += vA(i) * vB(i) * vC(i);
    }
    timer::pop();
    return result;
}

template <template <size_t, typename> typename XType, template <size_t, typename> typename YType, size_t Rank>
auto axpy(double alpha, const XType<Rank, double> &X, YType<Rank, double> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<Rank, double>, Rank, double> &&
                        is_incore_rank_tensor_v<YType<Rank, double>, Rank, double>> {
    timer::push("axpy");
    blas::daxpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
    timer::pop();
}

template <template <size_t, typename> typename XYType, size_t XYRank, template <size_t, typename> typename AType, size_t ARank>
auto ger(double alpha, const XYType<XYRank, double> &X, const XYType<XYRank, double> &Y, AType<ARank, double> *A)
    -> std::enable_if_t<is_incore_rank_tensor_v<XYType<XYRank, double>, 1, double> &&
                        is_incore_rank_tensor_v<AType<ARank, double>, 2, double>> {
    timer::Timer timer{"ger"};
    blas::dger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <template <size_t, typename> typename TensorType, size_t TensorRank>
auto getrf(TensorType<TensorRank, double> *A, std::vector<int> *pivot)
    -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<TensorRank, double>, 2, double>, int> {
    timer::push("getrf");
    if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
        println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
        pivot->resize(std::min(A->dim(0), A->dim(1)));
    }
    int result = blas::dgetrf(A->dim(0), A->dim(1), A->data(), A->stride(0), pivot->data());
    timer::pop();

    if (result < 0) {
        println("getrf: argument {} has an invalid value", -result);
    std:
        abort();
    }

    return result;
}

template <template <size_t, typename> typename TensorType, size_t TensorRank>
auto getri(TensorType<TensorRank, double> *A, const std::vector<int> &pivot)
    -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<TensorRank, double>, 2, double>, int> {
    timer::push("getri");

    // Call dgetri once to determine work size
    std::vector<double> work(1);
    blas::dgetri(A->dim(0), A->data(), A->stride(0), pivot.data(), work.data(), -1);
    work.resize(static_cast<int>(work[0]));
    std::fill(work.begin(), work.end(), 0.0);
    int result = blas::dgetri(A->dim(0), A->data(), A->stride(0), pivot.data(), work.data(), (int)work.size());
    timer::pop();

    if (result < 0) {
        println("getri: argument {} has an invalid value", -result);
    }
    return result;
}

template <template <size_t, typename> typename TensorType, size_t TensorRank>
auto invert(TensorType<TensorRank, double> *A) -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<TensorRank, double>, 2, double>> {
    timer::push("invert");

    std::vector<int> pivot(A->dim(0));
    int result = getrf(A, &pivot);
    if (result > 0) {
        timer::pop();
        println("invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed", result, result);
        std::abort();
    }

    result = getri(A, pivot);
    if (result > 0) {
        timer::pop();
        println("invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed", result, result);
        std::abort();
    }
    timer::pop();
}

template <typename SmartPtr>
auto invert(SmartPtr *A) -> std::enable_if_t<is_smart_pointer_v<SmartPtr>> {
    return invert(A->get());
}

enum class Norm : char { MaxAbs = 'M', One = 'O', Infinity = 'I', Frobenius = 'F', Two = 'F' };

template <template <size_t, typename> typename AType, size_t ARank>
auto norm(Norm norm_type, const AType<ARank, double> &a) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, double>, 2, double>, AType<ARank, double>> {
    if (norm_type != Norm::Infinity) {
        return blas::dlange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), nullptr);
    } else {
        std::vector<double> work(a->dim(0), 0.0);
        return blas::dlange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), work.data());
    }
}

template <template <size_t, typename> typename AType, size_t ARank>
auto svd_a(const AType<ARank, double> &_A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ARank, double>, 2, double>,
                              std::tuple<Tensor<2, double>, Tensor<1, double>, Tensor<2, double>>> {
    timer::Timer timer{"svd_a"};
    // Calling svd will destroy the original data.
    Tensor<2> A = _A;

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    auto U = Tensor{"U (stored columnwise)", m, m};
    U.zero();
    auto S = Tensor{"S", std::min(m, n)};
    S.zero();
    auto Vt = Tensor{"Vt (stored rowwise)", n, n};
    Vt.zero();

    // Workspace is not needed if we are using cblas/LAPACKE C wrapper.
    std::vector<int> iwork(8 * n);

    double lwork;
    // workspace query
    // blas::dgesdd('A', m, n, A.data(), n, S.data(), U.data(), k, Vt.data(), n, &lwork, -1, iwork.data());

    // std::vector<double> work((int)lwork);

    // int info = blas::dgesdd('A', m, n, A.data(), n, S.data(), U.data(), k, Vt.data(), n, work.data(), lwork, iwork.data());
    int info = blas::dgesdd('A', static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(n), S.data(), U.data(),
                            static_cast<int>(m), Vt.data(), static_cast<int>(n), nullptr, 0, nullptr);

    if (info != 0) {
        if (info < 0) {
            println("svd_a: Argument {} has an invalid parameter", -info);
            std::abort();
        } else {
            println("svd_a: error value {}", info);
            std::abort();
        }
    }

    return std::make_tuple(U, S, Vt);
}

} // namespace einsums::linear_algebra