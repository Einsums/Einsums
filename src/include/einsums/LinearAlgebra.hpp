#pragma once

#include "einsums/Blas.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/_Common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
auto sum_square(const AType<ADataType, ARank> &a, remove_complex_t<ADataType> *scale, remove_complex_t<ADataType> *sumsq) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ADataType, ARank>, 1, ADataType>> {
    LabeledSection0();

    int n    = a.dim(0);
    int incx = a.stride(0);
    blas::lassq(n, a.data(), incx, scale, sumsq);
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
auto gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, Rank>, 2, T> && is_incore_rank_tensor_v<BType<T, Rank>, 2, T> &&
                              is_incore_rank_tensor_v<CType<T, Rank>, 2, T>> {
    LabeledSection0();

    auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    blas::gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C->data(), ldc);
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
    LabeledSection0();

    Tensor<T, 2> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(1) : B.dim(0)};

    gemm<TransA, TransB>(alpha, A, B, 0.0, &C);

    return C;
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
auto gemv(const double alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const double beta, YType<T, XYRank> *y)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> && is_incore_rank_tensor_v<XType<T, XYRank>, 1, T> &&
                        is_incore_rank_tensor_v<YType<T, XYRank>, 1, T>> {
    LabeledSection1(fmt::format("<TransA={}>", TransA));
    auto m = A.dim(0), n = A.dim(1);
    auto lda  = A.stride(0);
    auto incx = x.stride(0);
    auto incy = y->stride(0);

    blas::gemv(TransA ? 't' : 'n', m, n, alpha, A.data(), lda, x.data(), incx, beta, y->data(), incy);
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
auto syev(AType<T, ARank> *A, WType<T, WRank> *W) -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> &&
                                                                      is_incore_rank_tensor_v<WType<T, WRank>, 1, T> && !is_complex_v<T>> {
    LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));

    assert(A->dim(0) == A->dim(1));

    auto           n     = A->dim(0);
    auto           lda   = A->stride(0);
    int            lwork = 3 * n;
    std::vector<T> work(lwork);

    blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
auto heev(AType<T, ARank> *A, WType<remove_complex_t<T>, WRank> *W)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> && is_incore_rank_tensor_v<WType<T, WRank>, 1, T> &&
                        is_complex_v<T>> {
    LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));
    assert(A->dim(0) == A->dim(1));

    auto                             n     = A->dim(0);
    auto                             lda   = A->stride(0);
    int                              lwork = 2 * n;
    std::vector<T>                   work(lwork);
    std::vector<remove_complex_t<T>> rwork(3 * n);

    blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
}

// This assumes column-major ordering!!
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
auto gesv(AType<T, ARank> *A, BType<T, BRank> *B)
    -> std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T> && is_incore_rank_tensor_v<BType<T, BRank>, 2, T>, int> {
    LabeledSection0();

    auto n   = A->dim(0);
    auto lda = A->dim(0);
    auto ldb = B->dim(1);

    auto nrhs = B->dim(0);

    int               lwork = n;
    std::vector<eint> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

template <template <typename, size_t> typename AType, size_t ARank, typename T, bool ComputeEigenvectors = true>
auto syev(const AType<T, ARank> &A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>, std::tuple<Tensor<T, 2>, Tensor<T, 1>>> {
    LabeledSection0();

    assert(A.dim(0) == A.dim(1));

    Tensor<T, 2> a = A;
    Tensor<T, 1> w{"eigenvalues", A.dim(0)};

    blas::syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto scale(T scale, AType<T, ARank> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, ARank, T>> {
    LabeledSection0();

    blas::scal(A->dim(0) * A->stride(0), scale, A->data(), 1);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto scale_row(size_t row, T scale, AType<T, ARank> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>> {
    LabeledSection0();

    blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto scale_column(size_t col, T scale, AType<T, ARank> *A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, double>> {
    LabeledSection0();

    blas::scal(A->dim(0), scale, A->data(0ul, col), A->stride(0));
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
    LabeledSection0();

    assert(a.dim(0) == a.dim(1));

    size_t       n  = a.dim(0);
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

template <template <typename, size_t> typename Type, typename T>
auto dot(const Type<T, 1> &A, const Type<T, 1> &B) ->
    typename std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, 1>, Type<T, 1>>, T> {
    LabeledSection0();

    assert(A.dim(0) == B.dim(0));

    auto result = blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    return result;
}

template <template <typename, size_t> typename Type, typename T, size_t Rank>
auto dot(const Type<T, Rank> &A, const Type<T, Rank> &B) ->
    typename std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, Rank>, Type<T, Rank>>, T> {
    LabeledSection0();

    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i));
        dim[0] *= A.dim(i);
    }

    return dot(TensorView<double, 1>(const_cast<Type<double, Rank> &>(A), dim),
               TensorView<double, 1>(const_cast<Type<double, Rank> &>(B), dim));
}

template <template <typename, size_t> typename Type, typename T, size_t Rank>
auto dot(const Type<T, Rank> &A, const Type<T, Rank> &B, const Type<T, Rank> &C) -> // NOLINT
    typename std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, Rank>, Type<T, Rank>>, T> {
    LabeledSection0();

    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
        dim[0] *= A.dim(i);
    }

    auto vA = TensorView<T, 1>(const_cast<Type<T, Rank> &>(A), dim);
    auto vB = TensorView<T, 1>(const_cast<Type<T, Rank> &>(B), dim);
    auto vC = TensorView<T, 1>(const_cast<Type<T, Rank> &>(C), dim);

    T result{0};
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < dim[0]; i++) {
        result += vA(i) * vB(i) * vC(i);
    }
    return result;
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
auto axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<T, Rank>, Rank, T> && is_incore_rank_tensor_v<YType<T, Rank>, Rank, T>> {
    LabeledSection0();

    blas::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
auto axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<T, Rank>, Rank, T> && is_incore_rank_tensor_v<YType<T, Rank>, Rank, T>> {
    LabeledSection0();

    blas::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
auto ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A)
    -> std::enable_if_t<is_incore_rank_tensor_v<XYType<T, XYRank>, 1, T> && is_incore_rank_tensor_v<AType<T, ARank>, 2, T>> {
    LabeledSection0();

    blas::ger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
auto getrf(TensorType<T, TensorRank> *A, std::vector<eint> *pivot)
    -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<T, TensorRank>, 2, T>, int> {
    LabeledSection0();

    if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
        println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
        pivot->resize(std::min(A->dim(0), A->dim(1)));
    }
    int result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->stride(0), pivot->data());

    if (result < 0) {
        println("getrf: argument {} has an invalid value", -result);
    std:
        abort();
    }

    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
auto getri(TensorType<T, TensorRank> *A, const std::vector<eint> &pivot)
    -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<T, TensorRank>, 2, T>, int> {
    LabeledSection0();

    int result = blas::getri(A->dim(0), A->data(), A->stride(0), pivot.data());

    if (result < 0) {
        println("getri: argument {} has an invalid value", -result);
    }
    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
auto invert(TensorType<T, TensorRank> *A) -> std::enable_if_t<is_incore_rank_tensor_v<TensorType<T, TensorRank>, 2, T>> {
    LabeledSection0();

    std::vector<eint> pivot(A->dim(0));
    int               result = getrf(A, &pivot);
    if (result > 0) {
        println("invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed", result, result);
        std::abort();
    }

    result = getri(A, pivot);
    if (result > 0) {
        println("invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed", result, result);
        std::abort();
    }
}

template <typename SmartPtr>
auto invert(SmartPtr *A) -> std::enable_if_t<is_smart_pointer_v<SmartPtr>> {
    LabeledSection0();

    return invert(A->get());
}

enum class Norm : char { MaxAbs = 'M', One = 'O', Infinity = 'I', Frobenius = 'F', Two = 'F' };

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
auto norm(Norm norm_type, const AType<ADataType, ARank> &a) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ADataType, ARank>, 2, ADataType>, remove_complex_t<ADataType>> {
    LabeledSection0();

    if (norm_type != Norm::Infinity) {
        return blas::lange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), nullptr);
    } else {
        std::vector<remove_complex_t<ADataType>> work(a->dim(0), 0.0);
        return blas::lange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), work.data());
    }
}

// Uses the original svd function found in lapack, gesvd, request all left and right vectors.
template <template <typename, size_t> typename AType, typename T, size_t ARank>
auto svd(const AType<T, ARank> &_A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>,
                                                                 std::tuple<Tensor<T, 2>, Tensor<remove_complex_t<T>, 1>, Tensor<T, 2>>> {
    LabeledSection0();

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    U.zero();
    auto S = create_tensor<remove_complex_t<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    Vt.zero();
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);
    superb.zero();

    int info = blas::gesvd('A', 'A', m, n, A.data(), n, S.data(), U.data(), m, Vt.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            println_abort("svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

enum class Vectors : char { All = 'A', Some = 'S', Overwrite = 'O', None = 'N' };

template <template <typename, size_t> typename AType, typename T, size_t ARank>
auto svd_dd(const AType<T, ARank> &_A, Vectors job = Vectors::All) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>,
                              std::tuple<Tensor<T, 2>, Tensor<remove_complex_t<T>, 1>, Tensor<T, 2>>> {
    LabeledSection0();

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    U.zero();
    auto S = create_tensor<remove_complex_t<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    Vt.zero();

    int info = blas::gesdd(static_cast<char>(job), static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(n), S.data(),
                           U.data(), static_cast<int>(m), Vt.data(), static_cast<int>(n));

    if (info != 0) {
        if (info < 0) {
            println_abort("svd_a: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd_a: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

template <template <typename, size_t> typename AType, typename T, size_t ARank>
auto truncated_svd(const AType<T, ARank> &_A, size_t k) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>,
                              std::tuple<Tensor<T, 2>, Tensor<remove_complex_t<T>, 1>, Tensor<T, 2>>> {
    LabeledSection0();

    size_t m = _A.dim(0);
    size_t n = _A.dim(1);

    // Omega Test Matrix
    auto omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<T, 2> Y("Y", m, k + 5);
    gemm<false, false>(T{1.0}, _A, omega, T{0.0}, &Y);

    Tensor<T, 1> tau("tau", std::min(m, k + 5));
    // Compute QR factorization of Y
    int info1 = blas::geqrf(m, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    if constexpr (!is_complex_v<T>) {
        int info2 = blas::orgqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));
    } else {
        int info2 = blas::ungqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));
    }

    // Cast the matrix A into a smaller rank (B)
    Tensor<T, 2> B("B", k + 5, n);
    gemm<true, false>(T{1.0}, Y, _A, T{0.0}, &B);

    // Perform svd on B
    auto [Utilde, S, Vt] = svd_dd(B);

    // Cast U back into full basis
    Tensor<T, 2> U("U", m, k + 5);
    gemm<false, false>(T{1.0}, Y, Utilde, T{0.0}, &U);

    return std::make_tuple(U, S, Vt);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto truncated_syev(const AType<T, ARank> &A, size_t k) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>, std::tuple<Tensor<T, 2>, Tensor<T, 1>>> {
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        println_abort("Non-square matrix used as input of truncated_syev!");
    }

    size_t n = A.dim(0);

    // Omega Test Matrix
    Tensor<double, 2> omega = create_random_tensor("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<double, 2> Y("Y", n, k + 5);
    gemm<false, false>(1.0, A, omega, 0.0, &Y);

    Tensor<double, 1> tau("tau", std::min(n, k + 5));
    // Compute QR factorization of Y
    eint info1 = blas::geqrf(n, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    eint info2 = blas::orgqr(n, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const double *>(tau.data()));

    Tensor<double, 2> &Q1 = Y;

    // Cast the matrix A into a smaller rank (B)
    // B = Q^T * A * Q
    Tensor<double, 2> Btemp("Btemp", k + 5, n);
    gemm<true, false>(1.0, Q1, A, 0.0, &Btemp);
    Tensor<double, 2> B("B", k + 5, k + 5);
    gemm<false, false>(1.0, Btemp, Q1, 0.0, &B);

    // Create buffer for eigenvalues
    Tensor<double, 1> w("eigenvalues", k + 5);

    // Diagonalize B
    syev(&B, &w);

    // Cast U back into full basis (B is column-major so we need to transpose it)
    Tensor<double, 2> U("U", n, k + 5);
    gemm<false, true>(1.0, Q1, B, 0.0, &U);

    return std::make_tuple(U, w);
}

template <typename T>
inline auto pseudoinverse(const Tensor<T, 2> &A, double tol) -> Tensor<T, 2> {
    LabeledSection0();

    auto [U, S, Vh] = svd_a(A);

    size_t new_dim;
    for (size_t v = 0; v < S.dim(0); v++) {
        T val = S(v);
        if (val > tol)
            scale_column(v, 1.0 / val, &U);
        else {
            new_dim = v;
            break;
        }
    }

    TensorView<T, 2> U_view = U(All, Range{0, new_dim});
    TensorView<T, 2> V_view = Vh(Range{0, new_dim}, All);

    Tensor<T, 2> pinv("pinv", A.dim(0), A.dim(1));
    gemm<false, false>(1.0, U_view, V_view, 0.0, &pinv);

    return pinv;
}

template <typename T>
inline auto solve_continuous_lyapunov(const Tensor<T, 2> &A, const Tensor<T, 2> &Q) -> Tensor<T, 2> {
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        println_abort("solve_continuous_lyapunov: Dimensions of A ({} x {}), do not match", A.dim(0), A.dim(1));
    }
    if (Q.dim(0) != Q.dim(1)) {
        println_abort("solve_continuous_lyapunov: Dimensions of Q ({} x {}), do not match", Q.dim(0), Q.dim(1));
    }
    if (A.dim(0) != Q.dim(0)) {
        println_abort("solve_continuous_lyapunov: Dimensions of A ({} x {}) and Q ({} x {}), do not match", A.dim(0), A.dim(1), Q.dim(0),
                      Q.dim(1));
    }

    size_t n = A.dim(0);

    /// TODO: Break this off into a separate schur function
    // Compute Schur Decomposition of A
    Tensor<T, 2>      R = A; // R is a copy of A
    Tensor<T, 2>      wr("Schur Real Buffer", n, n);
    Tensor<T, 2>      wi("Schur Imaginary Buffer", n, n);
    Tensor<T, 2>      U("Lyapunov U", n, n);
    std::vector<eint> sdim(1);
    blas::gees('V', n, R.data(), n, sdim.data(), wr.data(), wi.data(), U.data(), n);

    // Compute F = U^T * Q * U
    Tensor<T, 2> Fbuff = gemm<true, false>(1.0, U, Q);
    Tensor<T, 2> F     = gemm<false, false>(1.0, Fbuff, U);

    // Call the Sylvester Solve
    std::vector<T> scale(1);
    blas::trsyl('N', 'N', 1, n, n, const_cast<const T *>(R.data()), n, const_cast<const T *>(R.data()), n, F.data(), n, scale.data());

    Tensor<T, 2> Xbuff = gemm<false, false>(scale[0], U, F);
    Tensor<T, 2> X     = gemm<false, true>(1.0, Xbuff, U);

    return X;
}

ALIAS_TEMPLATE_FUNCTION(solve_lyapunov, solve_continuous_lyapunov)

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)