#pragma once

#include "Blas.hpp"
#include "STL.hpp"
#include "Tensor.hpp"
#include "Timer.hpp"
#include "einsums/Section.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace einsums::linear_algebra {

// template <typename, size_t> typename AType, size_t Rank, typename T

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
auto gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, Rank>, 2, T> && is_incore_rank_tensor_v<BType<T, Rank>, 2, T> &&
                              is_incore_rank_tensor_v<CType<T, Rank>, 2, T>> {
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
 * @return std::enable_if_t<std::is_base_of_v<Detail::TensorBase<double, 2>, AType> &&
 * std::is_base_of_v<Detail::TensorBase<double, 2>, BType>,
 * Tensor<double, 2>>
 *
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType, size_t Rank,
          typename T>
auto gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, Rank>, 2, T> && is_incore_rank_tensor_v<BType<T, Rank>, 2, T>, Tensor<T, 2>> {
    Tensor<T, 2> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(1) : B.dim(0)};

    gemm<TransA, TransB>(alpha, A, B, 0.0, &C);

    return C;
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
auto gemv(const double alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const double beta, YType<T, XYRank> *y)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> && is_incore_rank_tensor_v<XType<T, XYRank>, 1, T> &&
                        is_incore_rank_tensor_v<YType<T, XYRank>, 1, T>> {
    auto m = A.dim(0), n = A.dim(1);
    auto lda = A.stride(0);
    auto incx = x.stride(0);
    auto incy = y->stride(0);

    timer::push(fmt::format("gemv<{}>", TransA));
    blas::gemv(TransA ? 't' : 'n', m, n, alpha, A.data(), lda, x.data(), incx, beta, y->data(), incy);
    timer::pop();
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
auto syev(AType<T, ARank> *A, WType<T, WRank> *W) -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> &&
                                                                      is_incore_rank_tensor_v<WType<T, WRank>, 1, T> && !is_complex_v<T>> {
    Section section{fmt::format("syev<ComputeEigenvectors={}>", ComputeEigenvectors)};
    assert(A->dim(0) == A->dim(1));

    auto n = A->dim(0);
    auto lda = A->stride(0);
    int lwork = 3 * n;
    std::vector<T> work(lwork);

    blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
auto heev(AType<T, ARank> *A, WType<typename complex_type<T>::type, WRank> *W)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> && is_incore_rank_tensor_v<WType<T, WRank>, 1, T> &&
                        is_complex_v<T>> {
    Section section{fmt::format("heev<ComputeEigenvectors={}>", ComputeEigenvectors)};
    assert(A->dim(0) == A->dim(1));

    auto n = A->dim(0);
    auto lda = A->stride(0);
    int lwork = 2 * n;
    std::vector<T> work(lwork);
    std::vector<typename complex_type<T>::type> rwork(3 * n);

    blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
}

// This assumes column-major ordering!!
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
auto gesv(AType<T, ARank> *A, BType<T, BRank> *B)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> && is_incore_rank_tensor_v<BType<T, BRank>, 2, T>, int> {
    Section section{"gesv"};
    auto n = A->dim(0);
    auto lda = A->dim(0);
    auto ldb = B->dim(1);

    auto nrhs = B->dim(0);

    int lwork = n;
    std::vector<int> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

template <template <typename, size_t> typename AType, size_t ARank, typename T, bool ComputeEigenvectors = true>
auto syev(const AType<T, ARank> &A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>, std::tuple<Tensor<T, 2>, Tensor<T, 1>>> {
    assert(A.dim(0) == A.dim(1));

    Tensor<T, 2> a = A;
    Tensor<T, 1> w{"eigenvalues", A.dim(0)};

    blas::syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T = double>
auto scale(double scale, AType<T, ARank> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, ARank, double>> {
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
 * @return std::enable_if_t<std::is_base_of_v<Detail::TensorBase<double, 2>, AType>, AType>
 *
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto pow(const AType<T, ARank> &a, T alpha, T cutoff = std::numeric_limits<T>::epsilon()) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>, AType<T, ARank>> {
    assert(a.dim(0) == a.dim(1));

    size_t n = a.dim(0);
    Tensor<T, 2> a1 = a;
    Tensor<T, 2> result{"pow result", a.dim(0), a.dim(1)};
    Tensor<T, 1> e{"e", n};

    // Diagonalize
    syev(&a1, &e);

    Tensor<T, 2> a2 = a1;

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

template <template <typename, size_t> typename Type>
auto dot(const Type<double, 1> &A, const Type<double, 1> &B) ->
    typename std::enable_if_t<std::is_base_of_v<detail::TensorBase<double, 1>, Type<double, 1>>, double> {
    assert(A.dim(0) == B.dim(0));

    timer::push("dot");
    auto result = blas::ddot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    timer::pop();
    return result;
}

template <template <typename, size_t> typename Type, size_t Rank>
auto dot(const Type<double, Rank> &A, const Type<double, Rank> &B) ->
    typename std::enable_if_t<std::is_base_of_v<detail::TensorBase<double, Rank>, Type<double, Rank>>, double> {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i));
        dim[0] *= A.dim(i);
    }

    return dot(TensorView<double, 1>(const_cast<Type<double, Rank> &>(A), dim),
               TensorView<double, 1>(const_cast<Type<double, Rank> &>(B), dim));
}

template <template <typename, size_t> typename Type, size_t Rank>
auto dot(const Type<double, Rank> &A, const Type<double, Rank> &B, const Type<double, Rank> &C) -> // NOLINT
    typename std::enable_if_t<std::is_base_of_v<detail::TensorBase<double, Rank>, Type<double, Rank>>, double> {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
        dim[0] *= A.dim(i);
    }

    auto vA = TensorView<double, 1>(const_cast<Type<double, Rank> &>(A), dim);
    auto vB = TensorView<double, 1>(const_cast<Type<double, Rank> &>(B), dim);
    auto vC = TensorView<double, 1>(const_cast<Type<double, Rank> &>(C), dim);

    timer::push("dot3");
    double result = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < dim[0]; i++) {
        result += vA(i) * vB(i) * vC(i);
    }
    timer::pop();
    return result;
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, size_t Rank>
auto axpy(double alpha, const XType<double, Rank> &X, YType<double, Rank> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<double, Rank>, Rank, double> &&
                        is_incore_rank_tensor_v<YType<double, Rank>, Rank, double>> {
    timer::push("axpy");
    blas::daxpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
    timer::pop();
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, size_t ARank>
auto ger(double alpha, const XYType<double, XYRank> &X, const XYType<double, XYRank> &Y, AType<double, ARank> *A)
    -> std::enable_if_t<is_incore_rank_tensor_v<XYType<double, XYRank>, 1, double> &&
                        is_incore_rank_tensor_v<AType<double, ARank>, 2, double>> {
    timer::Timer timer{"ger"};
    blas::dger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <template <typename, size_t> typename TensorType, size_t TensorRank>
auto getrf(TensorType<double, TensorRank> *A, std::vector<int> *pivot)
    -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<double, TensorRank>, 2, double>, int> {
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

template <template <typename, size_t> typename TensorType, size_t TensorRank>
auto getri(TensorType<double, TensorRank> *A, const std::vector<int> &pivot)
    -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<double, TensorRank>, 2, double>, int> {
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

template <template <typename, size_t> typename TensorType, size_t TensorRank>
auto invert(TensorType<double, TensorRank> *A) -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<double, TensorRank>, 2, double>> {
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

template <template <typename, size_t> typename AType, size_t ARank>
auto norm(Norm norm_type, const AType<double, ARank> &a) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<double, ARank>, 2, double>, AType<double, ARank>> {
    if (norm_type != Norm::Infinity) {
        return blas::dlange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), nullptr);
    } else {
        std::vector<double> work(a->dim(0), 0.0);
        return blas::dlange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), work.data());
    }
}

template <template <typename, size_t> typename AType, size_t ARank>
auto svd_a(const AType<double, ARank> &_A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<double, ARank>, 2, double>,
                              std::tuple<Tensor<double, 2>, Tensor<double, 1>, Tensor<double, 2>>> {
    timer::Timer timer{"svd_a"};
    // Calling svd will destroy the original data.
    Tensor<double, 2> A = _A;

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