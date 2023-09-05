/**
 * @file LinearAlgebra.hpp
 * 
 * Functions to perform linear algebra.
 */

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

#if defined(EINSUMS_HAVE_EIGEN3)
#    include <Eigen/Core>
#    include <Eigen/SVD>

template <typename T>
using RowMatrix = ::Eigen::Matrix<T, ::Eigen::Dynamic, ::Eigen::Dynamic, ::Eigen::RowMajor>;

using RowMatrixXd = RowMatrix<double>;

#endif

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)

/**
 * @todo What does this do?
 *
 * @param a The tensor.
 * @param scale The scalar.
 * @param The output.
 *
 * @return The sum squared?
 */
template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
auto sum_square(const AType<ADataType, ARank> &a, remove_complex_t<ADataType> *scale, remove_complex_t<ADataType> *sumsq) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ADataType, ARank>, 1, ADataType>> {
    LabeledSection0();

    int n    = a.dim(0);
    int incx = a.stride(0);
    blas::lassq(n, a.data(), incx, scale, sumsq);
}

/**
 * Wrapper around Blas's gemm.
 * 
 * @param alpha Scale constant.
 * @param A The left tensor.
 * @param B The right tensor.
 * @param beta Another scale constant.
 * @param C The third tensor that is added on to the end. Also the result.
 */
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

    Tensor<T, 2> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(0) : B.dim(1)};

    gemm<TransA, TransB>(alpha, A, B, 0.0, &C);

    return C;
}

/**
 * Wrapper around Blas's gemv.
 *
 * @param alpha The first scale constant.
 * @param A The array.
 * @param x The first vector.
 * @param beta The second scale constant.
 * @param y The vector to add. Also the output vector.
 */
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

/**
 * Computes eigenvalues of symmetric matrices.
 *
 * @param A The tensor to decompose. Will contain the eigenvectors at the end.
 * @param W The output for eigenvalues.
 */
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

/**
 * Computes the eigenvalues and eigenvectors of a hermitian matrix.
 *
 * @param A The input matrix. At the end, it will contain the eigenvectors.
 * @param W The eigenvalue output.
 */
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
/**
 * Compute the solution to a linear system.
 * 
 * @param A The input array.
 * @param B The constants array. On exit, it contains the solutions.
 *
 * @return An informational flag from LAPACK.
 */
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

/**
 * Computes eigenvalues and eigenvectors of a symmetric matrix.
 *
 * @param A The matrix to decompose.
 * 
 * @return A tuple containing the eigenvalues and eigenvectors.
 */
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

/**
 * Performs scalar multiplication on a matrix.
 *
 * @param scale The scalar value.
 * @param A The matrix.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto scale(T scale, AType<T, ARank> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, ARank, T>> {
    LabeledSection0();

    blas::scal(A->dim(0) * A->stride(0), scale, A->data(), 1);
}

/**
 * Scales a row within a matrix.
 *
 * @param row The row to scale.
 * @param scale The scalar.
 * @param A The matrix to scale.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto scale_row(size_t row, T scale, AType<T, ARank> *A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>> {
    LabeledSection0();

    blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

/**
 * Scales a column in a matrix.
 *
 * @param col The column to scale.
 * @param scale The scalar.
 * @param A The matrix.
 */
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

/**
 * Computes the dot product between two vectors.
 *
 * @param A The first vector.
 * @param B The second vector.
 * 
 * @return The dot product.
 */
template <template <typename, size_t> typename Type, typename T>
auto dot(const Type<T, 1> &A, const Type<T, 1> &B) ->
    typename std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, 1>, Type<T, 1>>, T> {
    LabeledSection0();

    assert(A.dim(0) == B.dim(0));

    auto result = blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    return result;
}

/**
 * Computes the multi-dimensional dot product. @todo Double check this.
 *
 * @param A The first tensor.
 * @param B The second tensor.
 *
 * @return The dot product.
 */
template <template <typename, size_t> typename Type, typename T, size_t Rank>
auto dot(const Type<T, Rank> &A, const Type<T, Rank> &B) ->
    typename std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, Rank>, Type<T, Rank>>, T> {
    LabeledSection0();

    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i));
        dim[0] *= A.dim(i);
    }

    return dot(TensorView<T, 1>(const_cast<Type<T, Rank> &>(A), dim), TensorView<T, 1>(const_cast<Type<T, Rank> &>(B), dim));
}

/**
 * Returns the trinary dot product. @todo Double check this description.
 * 
 * @param A The first tensor.
 * @param B The second tensor.
 * @param C The third tensor.
 *
 * @return The trinary dot product.
 */
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

/**
 * Computes a linear combination of vectors.
 *
 * @param alpha The scalar to multiply to the first tensor.
 * @param X The first tensor.
 * @param Y The second tensor.
 */
template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
auto axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<T, Rank>, Rank, T> && is_incore_rank_tensor_v<YType<T, Rank>, Rank, T>> {
    LabeledSection0();

    blas::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
}

/**
 * Computes a linear combination of tensors.
 *
 * @param alpha The scalar to multiply the first tensor by.
 * @param X The first tensor.
 * @param beta The scalar to multiply the second tensor by.
 * @param Y The second tensor.
 */
template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
auto axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y)
    -> std::enable_if_t<is_incore_rank_tensor_v<XType<T, Rank>, Rank, T> && is_incore_rank_tensor_v<YType<T, Rank>, Rank, T>> {
    LabeledSection0();

    blas::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
}

/**
 * Wraps Blas's ger function, which is a bit complicated to describe.
 *
 * @param alpha The scalar to multiply the first product by.
 * @param X The first tensor.
 * @param Y The second tensor.
 * @param A The matrix to add to. Also the result.
 */
template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
auto ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A)
    -> std::enable_if_t<is_incore_rank_tensor_v<XYType<T, XYRank>, 1, T> && is_incore_rank_tensor_v<AType<T, ARank>, 2, T>> {
    LabeledSection0();

    blas::ger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

/**
 * Computes LU decomposition.
 * 
 * @param A The matrix to decompose. On output, it contains L and U.
 * @param pivot The pivots used.
 */
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

/**
 * Computes the inverse of an LU decomposed matrix.
 *
 * @param A The LU decomposed matrix as returned by getrf. At exit, contains
 * the inverse of A.
 * @param pivot The pivots used in getrf.
 *
 * @return A return value from Blas.
 */
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

/**
 * Computes the inverse of a matrix.
 *
 * @param A The matrix to invert. At exit, contains the inverse.
 *
 */
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

/**
 * Computes the inverse of a matrix.
 *
 * @param A The matrix to invert.
 */
template <typename SmartPtr>
auto invert(SmartPtr *A) -> std::enable_if_t<is_smart_pointer_v<SmartPtr>> {
    LabeledSection0();

    return invert(A->get());
}

/**
 * @enum Norm
 *
 * Represents different ways of computing a norm.
 */
enum class Norm : char { MaxAbs = 'M', One = 'O', Infinity = 'I', Frobenius = 'F', Two = 'F' };

/**
 * Compute a norm of a matrix.
 *
 * @param norm_type The kind of norm to compute.
 * @param a The matrix to find the norm of.
 */
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

#if defined(EINSUMS_HAVE_EIGEN3)
/**
 * Computes the singular value decomposition of a matrix.
 *
 * @param A The matrix to decompose.
 *
 * @return A tuple containing the singular values and the singular vectors.
 */
template <template <typename, size_t> typename AType, typename T, size_t ARank>
auto svd_eigen(const AType<T, ARank> &_A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>,
                              std::tuple<Tensor<T, 2>, Tensor<remove_complex_t<T>, 1>, Tensor<T, 2>>> {
    LabeledSection0();

    // using namespace einsums::tensor_algebra;

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> eA = _A;
    // auto eA = create_tensor("tmpA", _A.dim(1), _A.dim(2));
    // sort(Indices{index::i, index::j}, &eA, Indices{index::j, index::i}, _A);

    Eigen::Map<RowMatrixXd>       A(eA.vector_data().data(), _A.dim(0), _A.dim(1));
    Eigen::JacobiSVD<RowMatrixXd> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU);

    auto U = svd.matrixU();
    auto S = svd.singularValues();
    auto V = svd.matrixV();

    auto eU = create_tensor("U", U.rows(), U.cols());
    auto eS = create_tensor("S", S.size());
    auto eV = create_tensor("V", V.rows(), V.cols());

    // Eigen::Map<Eigen::MatrixX2d>
    RowMatrixXd::Map(eU.vector_data().data(), eU.dim(0), eU.dim(1)) = U;
    Eigen::VectorXd::Map(eS.vector_data().data(), eS.dim(0))        = S;
    RowMatrixXd::Map(eV.vector_data().data(), eV.dim(0), eV.dim(1)) = V;

    return std::make_tuple(eU, eS, eV);
}
#endif

// Uses the original svd function found in lapack, gesvd, request all left and right vectors.
/**
 * Compute the singular value decomposition.
 * 
 * @param A The matrix to decomopse.
 *
 * @return A tuple containing the singular values and the singular vectors.
 */
template <template <typename, size_t> typename AType, typename T, size_t ARank>
auto svd(const AType<T, ARank> &_A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>,
                                                                 std::tuple<Tensor<T, 2>, Tensor<remove_complex_t<T>, 1>, Tensor<T, 2>>> {
    LabeledSection0();

    DisableOMPThreads nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m   = A.dim(0);
    size_t n   = A.dim(1);
    size_t lda = A.stride(0);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    U.zero();
    auto S = create_tensor<remove_complex_t<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    Vt.zero();
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);
    superb.zero();

    int info = blas::gesvd('A', 'A', m, n, A.data(), lda, S.data(), U.data(), m, Vt.data(), n, superb.data());

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

/**
 * Compute the nullspace of a matrix.
 *
 * @param A The matrix to find the nullspace of.
 *
 * @return The matrix representation of the nullspace.
 */
template <template <typename, size_t> typename AType, typename T, size_t Rank>
auto svd_nullspace(const AType<T, Rank> &_A) -> typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, Rank>, 2, T>, Tensor<T, 2>> {
    LabeledSection0();

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    eint m   = A.dim(0);
    eint n   = A.dim(1);
    eint lda = A.stride(0);

    auto U = create_tensor<T>("U", m, m);
    zero(U);
    auto S = create_tensor<T>("S", n);
    zero(S);
    auto V = create_tensor<T>("V", n, n);
    zero(V);
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);

    int info = blas::gesvd('N', 'A', m, n, A.data(), lda, S.data(), U.data(), m, V.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            println_abort("svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd: error value {}", info);
        }
    }

    // Determine the rank of the nullspace matrix
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (S(i) > 1e-12) {
            rank++;
        }
    }

    // println("rank {}", rank);
    auto Vview     = V(Range{rank, V.dim(0)}, All);
    auto nullspace = Tensor(V);

    // Normalize nullspace. LAPACK does not guarentee them to be orthonormal
    for (int i = 0; i < nullspace.dim(0); i++) {
        T sum{0};
        for (int j = 0; j < nullspace.dim(1); j++) {
            sum += std::pow(nullspace(i, j), 2.0);
        }
        sum = std::sqrt(sum);
        scale_row(i, sum, &nullspace);
    }

    return nullspace;
}

/**
 * @enum Vectors
 *
 * Options for computing vectors in the singular value decomposition algorithm.
 */
enum class Vectors : char { All = 'A', Some = 'S', Overwrite = 'O', None = 'N' };

/**
 * Perform singular value decomposition.
 *
 * @param A The array to decompose.
 * @param job Which vectors to compute.
 *
 * @return A tuple containing the singular values and the singular vectors.
 */
template <template <typename, size_t> typename AType, typename T, size_t ARank>
auto svd_dd(const AType<T, ARank> &_A, Vectors job = Vectors::All) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>,
                              std::tuple<Tensor<T, 2>, Tensor<remove_complex_t<T>, 1>, Tensor<T, 2>>> {
    LabeledSection0();

    DisableOMPThreads nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    zero(U);
    auto S = create_tensor<remove_complex_t<T>>("S", std::min(m, n));
    zero(S);
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    zero(Vt);

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

/**
 * Compute the truncated singular value. @todo How is this different?
 *
 * @param _A The matrix to decompose.
 * @param k The truncation parameter.
 *
 * @return A tuple containing the singular values and singular vectors.
 */
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

/**
 * Computes a truncated eigendecomposition of a symmetric matrix.
 * @todo How is this different?
 *
 * @param A The matrix to decompose.
 * @param k The truncation parameter.
 *
 * @return A tuple containing the eigenvalues and eigenvectors.
 */
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

/**
 * Computes a pseudoinverse of a non-square matrix.
 *
 * @param The matrix to invert.
 * @param tol The tolerance.
 *
 * @return The pseudoinverse.
 */
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

/**
 * Solves a continuous Lyapunov. @todo What does this mean?
 *
 * @param A The matrix to solve. 
 * @param Q The other parameters.
 *
 * @return The solution.
 */
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

/**
 * Computes the QR decomposition of a matrix.
 *
 * @param A The matrix to decompose.
 *
 * @return A tuple containing information for the final step of 
 * QR decomposition.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
auto qr(const AType<T, ARank> &_A) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, ARank>, 2, T>, std::tuple<Tensor<T, 2>, Tensor<T, 1>>> {
    LabeledSection0();

    // Copy A because it will be overwritten by the QR call.
    Tensor<T, 2> A = _A;
    const eint   m = A.dim(0);
    const eint   n = A.dim(1);

    Tensor<double, 1> tau("tau", std::min(m, n));
    // Compute QR factorization of Y
    eint info = blas::geqrf(m, n, A.data(), n, tau.data());

    if (info != 0) {
        println_abort("{} parameter to geqrf has an illegal value.", -info);
    }

    // Extract Matrix Q out of QR factorization
    // eint info2 = blas::orgqr(m, n, tau.dim(0), A.data(), n, const_cast<const double *>(tau.data()));
    return {A, tau};
}

/**
 * Compute the QR decomposition of a matrix.
 *
 * @param qr The first part of the tuple from calling qr.
 * @param tau The second part of the tuple from calling qr.
 *
 * @return The Q matrix.
 */
template <typename T>
auto q(const Tensor<T, 2> &qr, const Tensor<T, 1> &tau) -> Tensor<T, 2> {
    const eint m = qr.dim(1);
    const eint p = qr.dim(0);

    Tensor<T, 2> Q = qr;

    eint info = blas::orgqr(m, m, p, Q.data(), m, tau.data());
    if (info != 0) {
        println_abort("{} parameter to orgqr has an illegal value. {} {} {}", -info, m, m, p);
    }

    return Q;
}

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)
