//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Concepts/Complex.hpp>

#include <type_traits>

namespace einsums::blas {

/**
 * @brief Determines whether a type can be used in a BLAS call.
 *
 * This checks to see if a type is @c float or @c double .
 *
 * @versionadded{1.1.0}
 */
template <typename T>
struct IsBlasable : std::is_floating_point<std::remove_cvref_t<T>> {};

/**
 * @brief Determines whether a type can be used in a BLAS call.
 *
 * This checks to see if a type is @c std::complex<float> or @c std::complex<double> .
 *
 * @versionadded{1.1.0}
 */
template <typename T>
struct IsBlasable<std::complex<T>> : std::is_floating_point<T> {};

/**
 * @property IsBlasableV<T>
 *
 * @brief Boolean wrapper of IsBlasable<T>.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
constexpr bool IsBlasableV = IsBlasable<T>::value;

/**
 * @concept Blasable<T>
 *
 * @brief Concept version of IsBlasableV<T>.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
concept Blasable = IsBlasableV<T>;

#ifndef DOXYGEN
namespace detail {
// These routines take care of actually making the call to the BLAS equivalent.
void EINSUMS_EXPORT sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                          int_t ldb, float beta, float *c, int_t ldc);
void EINSUMS_EXPORT dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                          int_t ldb, double beta, double *c, int_t ldc);
void EINSUMS_EXPORT cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                          int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);

void EINSUMS_EXPORT zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                          int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c,
                          int_t ldc);
} // namespace detail
#endif

/**
 * @brief Perform a General Matrix Multiply (GEMM) operation.
 *
 * This function computes the product of two matrices,
 * \f[
 * \mathbf{C} := \alpha \mathbf{A}\mathbf{B} + \beta\mathbf{C}
 * \f]
 * where @f$\mathbf{A}@f$, @f$\mathbf{B}@f$, and @f$\mathbf{C}@f$ are matrices, and
 * @f$\alpha@f$ and @f$\beta@f$ are scalar values.
 *
 * @tparam T The datatype of the GEMM.
 * @param transa Whether to transpose matrix a :
 *   - 'N' or 'n' for no transpose,
 *   - 'T' or 't' for transpose,
 *   - 'C' or 'c' for conjugate transpose.
 * @param transb Whether to transpose matrix b .
 * @param m The number of rows in matrix A and C.
 * @param n The number of columns in matrix B and C.
 * @param k The number of columns in matrix A and rows in matrix B.
 * @param alpha The scalar alpha.
 * @param a A pointer to the matrix A with dimensions `(lda, k)` when transa is 'N' or 'n', and `(lda, m)`
 * otherwise.
 * @param lda Leading dimension of A, specifying the distance between two consecutive columns.
 * @param b A pointer to the matrix B with dimensions `(ldb, n)` when transB is 'N' or 'n', and `(ldb, k)`
 * otherwise.
 * @param ldb Leading dimension of B, specifying the distance between two consecutive columns.
 * @param beta The scalar beta.
 * @param c A pointer to the matrix C with dimensions `(ldc, n)`.
 * @param ldc Leading dimension of C, specifying the distance between two consecutive columns.
 *
 * @note The function performs one of the following matrix operations:
 * - If transA is 'N' or 'n' and transB is 'N' or 'n': \f$C = alpha * A * B + beta * C\f$
 * - If transA is 'N' or 'n' and transB is 'T' or 't': \f$C = alpha * A * B^T + beta * C\f$
 * - If transA is 'T' or 't' and transB is 'N' or 'n': \f$C = alpha * A^T * B + beta * C\f$
 * - If transA is 'T' or 't' and transB is 'T' or 't': \f$C = alpha * A^T * B^T + beta * C\f$
 * - If transA is 'C' or 'c' and transB is 'N' or 'n': \f$C = alpha * A^H * B + beta * C\f$
 * - If transA is 'C' or 'c' and transB is 'T' or 't': \f$C = alpha * A^H * B^T + beta * C\f$
 *
 * @return None.
 *
 * @throws invalid_argument If @p transA or @p transB are invalid.
 * @throws domain_error If the values of @p m , @p n , or @p k are negative, or the values of @p lda , @p ldb , or @p ldc are invalid.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void gemm(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c,
          int_t ldc);

#if !defined(DOXYGEN)
// These are the template specialization for the data types we support. If an unsupported data type
// is attempted a compiler error will occur.
template <>
inline void gemm<float>(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                        int_t ldb, float beta, float *c, int_t ldc) {
    detail::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                         int_t ldb, double beta, double *c, int_t ldc) {
    detail::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                      std::complex<float> const *a, int_t lda, std::complex<float> const *b, int_t ldb,
                                      std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    detail::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                       std::complex<double> const *a, int_t lda, std::complex<double> const *b, int_t ldb,
                                       std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    detail::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif

#if !defined(DOXYGEN)
namespace detail {
void EINSUMS_EXPORT sgemv(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta,
                          float *y, int_t incy);
void EINSUMS_EXPORT dgemv(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta,
                          double *y, int_t incy);
void EINSUMS_EXPORT cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *x, int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy);
void EINSUMS_EXPORT zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy);
} // namespace detail
#endif

/**
 * @brief Computes a matrix-vector product using a general matrix.
 *
 * The gemv routine performs a matrix-vector operation defined as:
 * @f[
 * \mathbf{y} := \alpha \mathbf{A} \mathbf{x} + \beta \mathbf{y}
 * @f]
 * or
 * @f[
 * \mathbf{y} := \alpha \mathbf{A}^T \mathbf{x} + \beta \mathbf{y}
 * @f]
 * or
 * @f[
 * \mathbf{y} := \alpha \mathbf{A}^H \mathbf{x} + \beta \mathbf{y}
 * @f]
 *
 * @tparam T the underlying data type of the matrix and vector
 * @param transa what to do with \p a - no trans, trans, conjg
 * @param m specifies the number of rows of \p a
 * @param n specifies the number of columns of \p a
 * @param alpha Specifies the scaler alpha
 * @param a Array, size lda * m
 * @param lda Specifies the leading dimension of \p a as declared in the calling function
 * @param x array, vector x
 * @param incx Specifies the increment for the elements of \p x
 * @param beta Specifies the scalar beta. When beta is set to zero, then \p y need not be set on input.
 * @param y array, vector y
 * @param incy Specifies the increment for the elements of \p y .
 *
 * @throws invalid_argument If @p transA is invalid.
 * @throws domain_error If the values of @p m or @p n are negative, the value of @p lda is invalid, or either @p incx or @p incy is
 * zero.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void gemv(char transa, int_t m, int_t n, T alpha, T const *a, int_t lda, T const *x, int_t incx, T beta, T *y, int_t incy);

#if !defined(DOXYGEN)
template <>
inline void gemv<float>(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta,
                        float *y, int_t incy) {
    detail::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta,
                         double *y, int_t incy) {
    detail::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                                      std::complex<float> const *x, int_t incx, std::complex<float> beta, std::complex<float> *y,
                                      int_t incy) {
    detail::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                                       std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y,
                                       int_t incy) {
    detail::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t;
auto EINSUMS_EXPORT dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t;
} // namespace detail
#endif

/**
 * @brief Performs diagonalization of a symmetrix matrix.
 *
 * The syev routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^T
 * @f]
 * In the above equation, @f$ \mathbf{A} @f$ is a real symmetric matrix, @f$ \mathbf{P} @f$ is a real orthogonal matrix whose columns are
 * the eigenvectors of @f$ \mathbf{A} @f$, and @f$ \mathbf{\Lambda} @f$ is a diagonal matrix, whose elements are the eigenvalues of @f$
 * \mathbf{A} @f$. The eigenvalues are stored in a vector form on exit.
 *
 * @param job Whether to compute the eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param uplo Whether the matrix data is stored in the upper or lower triangle. Can be either 'u' or 'l', case insensitive.
 * @param n The number of rows/columns of the input matrix.
 * @param a The input matrix. On output, it will be changed. If the eigenvectors are requested, then they will be placed
 * in the columns of @p a on exit.
 * @param lda The leading dimension of the input matrix.
 * @param w The output vector for the eigenvalues.
 * @param work A work array. If @p lwork is -1, then no operations are performed and the first value in the work array is the
 * optimal work buffer size.
 * @param lwork The size of the work array. If @p lwork is -1, then a workspace query is assumed. No operations will be performed
 * and the optimal workspace size will be put into the first element of @p work.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto syev(char job, char uplo, int_t n, T *a, int_t lda, T *w, T *work, int_t lwork) -> int_t;

#ifndef DOXYGEN
template <>
inline auto syev<float>(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    return detail::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

template <>
inline auto syev<double>(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    return detail::dsyev(job, uplo, n, a, lda, w, work, lwork);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT ssterf(int_t n, float *d, float *e) -> int_t;
auto EINSUMS_EXPORT dsterf(int_t n, double *d, double *e) -> int_t;
} // namespace detail
#endif

/**
 * @brief Computes the eigenvalues of a symmetric tridiagonal matrix.
 * The sterf routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^T
 * @f]
 * In the above equation, @f$ \mathbf{A} @f$ is a real symmetric tridiagonal matrix, @f$ \mathbf{P} @f$ is a real orthogonal matrix whose
 * columns are the eigenvectors of @f$ \mathbf{A} @f$, and @f$ \mathbf{\Lambda} @f$ is a diagonal matrix, whose elements are the eigenvalues
 * of @f$ \mathbf{A} @f$. The eigenvalues are stored in a vector form on exit.
 *
 * @param n The number of elements along the diagonal.
 * @param d The diagonal elements. On exit, it contains the eigenvalues.
 * @param e The off-diagonal elements. There is one fewer of these than the diagonal elements.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto sterf(int_t n, T *d, T *e) -> int_t;

#ifndef DOXYGEN
template <>
inline auto sterf<float>(int_t n, float *d, float *e) -> int_t {
    return detail::ssterf(n, d, e);
}

template <>
inline auto sterf<double>(int_t n, double *d, double *e) -> int_t {
    return detail::dsterf(n, d, e);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
                          int_t ldvr) -> int_t;
auto EINSUMS_EXPORT dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl,
                          double *vr, int_t ldvr) -> int_t;
auto EINSUMS_EXPORT cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w,
                          std::complex<float> *vl, int_t ldvl, std::complex<float> *vr, int_t ldvr) -> int_t;
auto EINSUMS_EXPORT zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w,
                          std::complex<double> *vl, int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t;
} // namespace detail
#endif

// Complex version
/**
 * @brief Performs diagonalization of a matrix.
 *
 * The syev routine finds the matrices that satisfy the following equations.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^{-1}
 * \mathbf{A}^T = \mathbf{L} \mathbf{\Lambda} \mathbf{L}^{-1}
 * @f]
 * In the above equation, @f$\mathbf{A}@f$ is a matrix, @f$\mathbf{P}@f$ is a complex-valued matrix whose columns are
 * the right eigenvectors of @f$\mathbf{A}@f$, @f$\mathbf{L}@f$ is a complex-valued matrix whose columns are the left eigenvectors of
 * @f$\mathbf{A}@f$, and @f$\mathbf{\Lambda}@f$ is a complex-valued diagonal matrix, whose elements are the eigenvalues of @f$ A @f$. The
 * eigenvalues are stored in a vector form on exit. The eigenvectors are stored in a special way if the input is a real matrix. If the input
 * is a complex matrix, then the eigenvectors are stored plainly in the columns of the appropriate output matrices.
 *
 * @param jobvl Whether to compute the left eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param jobvr Whether to compute the right eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param n The number of rows/columns of the input matrix.
 * @param a The input matrix. On output, it will be changed.
 * @param lda The leading dimension of the input matrix.
 * @param w The output vector for the eigenvalues.
 * @param vl The left eigenvector output. If @p jobvl is 'n', then this is not referenced and may be null.
 * @param ldvl The leading dimension of the left eigenvectors. Even if not referenced, it must be at least 1.
 * @param vr The right eigenvector output. If @p jobvr is 'n', then this is not referenced and may be null.
 * @param ldvr The leading dimension of the right eigenvectors. Even if not referenced, it must be at least 1.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto geev(char jobvl, char jobvr, int_t n, T *a, int_t lda, AddComplexT<T> *w, T *vl, int_t ldvl, T *vr, int_t ldvr) -> int_t;

#ifndef DOXYGEN
template <>
inline auto geev<float>(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
                        int_t ldvr) -> int_t {
    return detail::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<double>(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr,
                         int_t ldvr) -> int_t {
    return detail::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<float>>(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w,
                                      std::complex<float> *vl, int_t ldvl, std::complex<float> *vr, int_t ldvr) -> int_t {
    return detail::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<double>>(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w,
                                       std::complex<double> *vl, int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t {
    return detail::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}
#endif

#ifndef DOXGYEN
namespace detail {
auto EINSUMS_EXPORT cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
                          float *rwork) -> int_t;
auto EINSUMS_EXPORT zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work,
                          int_t lwork, double *rworl) -> int_t;
} // namespace detail
#endif

/**
 * @brief Performs diagonalization of a Hermitian matrix.
 *
 * The syev routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^H
 * @f]
 * In the above equation, @f$\mathbf{A}@f$ is a Hermitian matrix, @f$\mathbf{P}@f$ is a unitary matrix whose columns are
 * the eigenvectors of @f$\mathbf{A}@f$, and @f$\mathbf{\Lambda}@f$ is a diagonal matrix, whose elements are the eigenvalues of
 * @f$\mathbf{A}@f$. The eigenvalues are stored in a vector form on exit.
 *
 * @param job Whether to compute the eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param uplo Whether the matrix data is stored in the upper or lower triangle. Can be either 'u' or 'l', case insensitive.
 * @param n The number of rows/columns of the input matrix.
 * @param a The input matrix. On output, it will be changed. If the eigenvectors are requested, then they will be placed
 * in the columns of @p a on exit.
 * @param lda The leading dimension of the input matrix.
 * @param w The output vector for the eigenvalues.
 * @param work A work array. If @p lwork is -1, then no operations are performed and the first value in the work array is the
 * optimal work buffer size.
 * @param lwork The size of the work array. If @p lwork is -1, then a workspace query is assumed. No operations will be performed
 * and the optimal workspace size will be put into the first element of @p work.
 * @param rwork A work array for real values.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto heev(char job, char uplo, int_t n, std::complex<T> *a, int_t lda, T *w, std::complex<T> *work, int_t lwork, T *rwork) -> int_t;

#ifndef DOXYGEN
template <>
inline auto heev<float>(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
                        float *rwork) -> int_t {
    return detail::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

template <>
inline auto heev<double>(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work,
                         int_t lwork, double *rwork) -> int_t {
    return detail::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb)
    -> int_t;
} // namespace detail
#endif

/**
 * @brief Solve a system of linear equations.
 *
 * Solves equations of the following form.
 * @f[
 * \mathbf{A}\mathbf{x} = \mathbf{B}
 * @f]
 *
 * @param n The number of rows and columns of @f$\mathbf{A}@f$ and rows @f$\mathbf{B}@f$.
 * @param nrhs The number of columns of @f$\mathbf{B}@f$
 * @param a The coefficient matrix. On exit, it contains the LU decomposition of @p a, where the lower-triangle matrix has unit diagonal
 * entries, which are not stored.
 * @param lda The leading dimension of @p a.
 * @param ipiv A list of pivots used in the decomposition.
 * @param b The results matrix. On exit, it contains the values of @f$\mathbf{x}@f$ that satisfy the system of equations.
 * @param ldb The leading dimension of @p b.
 *
 * @return 0 on success. If positive, then the matrix was singular. If negative, then a bad value was passed to the function.
 * The absolute value indicates which parameter was bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gesv(int_t n, int_t nrhs, T *a, int_t lda, int_t *ipiv, T *b, int_t ldb) -> int_t;

#ifndef DOXYGEN
template <>
inline auto gesv<float>(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t {
    return detail::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<double>(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t {
    return detail::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<float>>(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b,
                                      int_t ldb) -> int_t {
    return detail::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<double>>(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b,
                                       int_t ldb) -> int_t {
    return detail::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT sscal(int_t n, float alpha, float *vec, int_t inc);
void EINSUMS_EXPORT dscal(int_t n, double alpha, double *vec, int_t inc);
void EINSUMS_EXPORT cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc);
void EINSUMS_EXPORT zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc);
void EINSUMS_EXPORT csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void EINSUMS_EXPORT zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc);

void EINSUMS_EXPORT srscl(int_t n, float alpha, float *vec, int_t inc);
void EINSUMS_EXPORT drscl(int_t n, double alpha, double *vec, int_t inc);
void EINSUMS_EXPORT csrscl(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void EINSUMS_EXPORT zdrscl(int_t n, double alpha, std::complex<double> *vec, int_t inc);
} // namespace detail
#endif

/**
 * @brief Scales a vector by a value.
 *
 * @param n The number of elements to scale the vector by.
 * @param alpha The scale factor.
 * @param vec The vector to scale.
 * @param inc The spacing between elements of the vector.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void scal(int_t n, T const alpha, T *vec, int_t inc);

/**
 * @brief Scales a complex vector by a real value.
 *
 * @param n The number of elements to scale the vector by.
 * @param alpha The scale factor.
 * @param vec The vector to scale.
 * @param inc The spacing between elements of the vector.
 *
 * @versionadded{1.0.0}
 */
template <Complex T>
void scal(int_t n, RemoveComplexT<T> const alpha, T *vec, int_t inc);

#ifndef DOXYGEN
template <>
inline void scal<float>(int_t n, float const alpha, float *vec, int_t inc) {
    detail::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(int_t n, double const alpha, double *vec, int_t inc) {
    detail::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, std::complex<float> const alpha, std::complex<float> *vec, int_t inc) {
    detail::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, std::complex<double> const alpha, std::complex<double> *vec, int_t inc) {
    detail::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, float const alpha, std::complex<float> *vec, int_t inc) {
    detail::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, double const alpha, std::complex<double> *vec, int_t inc) {
    detail::zdscal(n, alpha, vec, inc);
}
#endif

/**
 * @brief Scales a vector by the reciprocal of a value.
 *
 * @param n The number of elements in the vector.
 * @param alpha The value to divide all the elements in the vector by.
 * @param vec The vector to scale.
 * @param inc The spacing between elements in the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void rscl(int_t n, T const alpha, T *vec, int_t inc);

/**
 * @brief Scales a complex vector by the reciprocal of a real value.
 *
 * @param n The number of elements in the vector.
 * @param alpha The value to divide all the elements in the vector by.
 * @param vec The vector to scale.
 * @param inc The spacing between elements in the vector.
 *
 * @versionadded{2.0.0}
 */
template <Complex T>
void rscl(int_t n, RemoveComplexT<T> const alpha, T *vec, int_t inc);

#ifndef DOXYGEN
template <>
inline void rscl<float>(int_t n, float const alpha, float *vec, int_t inc) {
    detail::srscl(n, alpha, vec, inc);
}

template <>
inline void rscl<double>(int_t n, double const alpha, double *vec, int_t inc) {
    detail::drscl(n, alpha, vec, inc);
}

template <>
inline void rscl<std::complex<float>>(int_t n, std::complex<float> const alpha, std::complex<float> *vec, int_t inc) {
    detail::cscal(n, std::complex<float>{1.0} / alpha, vec, inc);
}

template <>
inline void rscl<std::complex<double>>(int_t n, std::complex<double> const alpha, std::complex<double> *vec, int_t inc) {
    detail::zscal(n, std::complex<double>{1.0} / alpha, vec, inc);
}

template <>
inline void rscl<std::complex<float>>(int_t n, float const alpha, std::complex<float> *vec, int_t inc) {
    detail::csrscl(n, alpha, vec, inc);
}

template <>
inline void rscl<std::complex<double>>(int_t n, double const alpha, std::complex<double> *vec, int_t inc) {
    detail::zdrscl(n, alpha, vec, inc);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float;
auto EINSUMS_EXPORT ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double;
auto EINSUMS_EXPORT cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float>;
auto EINSUMS_EXPORT zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double>;
auto EINSUMS_EXPORT cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float>;
auto EINSUMS_EXPORT zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double>;
} // namespace detail
#endif

/**
 * Computes the dot product of two vectors. For complex vectors it is the non-conjugated dot product;
 * (c|z)dotu in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param n length of the vectors
 * @param x first vector
 * @param incx how many elements to skip in x
 * @param y second vector
 * @param incy how many elements to skip in yo
 * @return result of the dot product
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto dot(int_t n, T const *x, int_t incx, T const *y, int_t incy) -> T;

#ifndef DOXYGEN
template <>
inline auto dot<float>(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    return detail::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    return detail::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float> {
    return detail::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double> {
    return detail::zdot(n, x, incx, y, incy);
}
#endif

/**
 * Computes the dot product of two vectors. For complex vector it is the conjugated dot product;
 * (c|z)dotc in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param n length of the vectors
 * @param x first vector
 * @param incx how many elements to skip in x
 * @param y second vector
 * @param incy how many elements to skip in yo
 * @return result of the dot product
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto dotc(int_t n, T const *x, int_t incx, T const *y, int_t incy) -> T;

#ifndef DOXYGEN
template <>
inline auto dotc<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float> {
    return detail::cdotc(n, x, incx, y, incy);
}

template <>
inline auto dotc<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double> {
    return detail::zdotc(n, x, incx, y, incy);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y);
void EINSUMS_EXPORT daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y);
void EINSUMS_EXPORT caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y,
                          int_t inc_y);
void EINSUMS_EXPORT zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y,
                          int_t inc_y);
} // namespace detail
#endif

/**
 * @brief Adds two vectors together with a scale factor.
 *
 * Computes the following.
 * @f[
 * \mathbf{y} := \alpha\mathbf{x} + \mathbf{y}
 * @f]
 *
 * @param n The number of elements in the vectors.
 * @param alpha_x The scale factor for the input vector.
 * @param x The input vector.
 * @param inc_x The skip value for the output vector. It can be negative to go in reverse, or zero to broadcast values to @p y.
 * @param y The output vector.
 * @param inc_y The skip value for the output vector. It can be negative to go in reverse, or zero to sum over the elements of @p x .
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void axpy(int_t n, T alpha_x, T const *x, int_t inc_x, T *y, int_t inc_y);

#ifndef DOXYGEN
template <>
inline void axpy<float>(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y) {
    detail::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y) {
    detail::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x,
                                      std::complex<float> *y, int_t inc_y) {
    detail::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x,
                                       std::complex<double> *y, int_t inc_y) {
    detail::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT saxpby(int_t n, float alpha_x, float const *x, int_t inc_x, float b, float *y, int_t inc_y);
void EINSUMS_EXPORT daxpby(int_t n, double alpha_x, double const *x, int_t inc_x, double b, double *y, int_t inc_y);
void EINSUMS_EXPORT caxpby(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> b,
                           std::complex<float> *y, int_t inc_y);
void EINSUMS_EXPORT zaxpby(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> b,
                           std::complex<double> *y, int_t inc_y);
} // namespace detail
#endif

/**
 * @brief Adds two vectors together with a scale factor.
 *
 * Computes the following.
 * @f[
 * \mathbf{y} := \alpha\mathbf{x} + \beta\mathbf{y}
 * @f]
 *
 * @param n The number of elements in the vectors.
 * @param alpha_x The scale factor for the input vector.
 * @param x The input vector.
 * @param inc_x The skip value for the output vector. It can be negative to go in reverse, or zero to broadcast values to @p y.
 * @param beta The scale factor for the output vector.
 * @param y The output vector.
 * @param inc_y The skip value for the output vector. It can be negative to go in reverse, or zero to sum over the elements of @p x .
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void axpby(int_t n, T alpha_x, T const *x, int_t inc_x, T b, T *y, int_t inc_y);

#ifndef DOXYGEN
template <>
inline void axpby<float>(int_t n, float alpha_x, float const *x, int_t inc_x, float b, float *y, int_t inc_y) {
    detail::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<double>(int_t n, double alpha_x, double const *x, int_t inc_x, double b, double *y, int_t inc_y) {
    detail::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<float>>(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x,
                                       std::complex<float> b, std::complex<float> *y, int_t inc_y) {
    detail::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<double>>(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x,
                                        std::complex<double> b, std::complex<double> *y, int_t inc_y) {
    detail::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT sger(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda);
void EINSUMS_EXPORT dger(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda);
void EINSUMS_EXPORT cger(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                         std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda);
void EINSUMS_EXPORT zger(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                         std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda);
void EINSUMS_EXPORT cgerc(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                          std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda);
void EINSUMS_EXPORT zgerc(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                          std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda);
} // namespace detail
#endif

/**
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{x}\mathbf{y}^T + \mathbf{A}
 * @f]
 *
 * @param m The number of entries in @p x.
 * @param n The number of entries in @p y.
 * @param alpha The scale factor for the outer product.
 * @param x The left input vector.
 * @param inc_x The skip value for the left input. May be negative to go in reverse.
 * @param y The right input vector.
 * @param inc_y The skip value for the right input. May be negative to go in reverse.
 * @param a The output matrix.
 * @param lda The leading dimension of @p a.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void ger(int_t m, int_t n, T alpha, T const *x, int_t inc_x, T const *y, int_t inc_y, T *a, int_t lda);

/**
 * Performs a rank-1 update of a general matrix.
 *
 * The ?gerc routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{x}\mathbf{y}^H + \mathbf{A}
 * @f]
 *
 * @param m The number of entries in @p x.
 * @param n The number of entries in @p y.
 * @param alpha The scale factor for the outer product.
 * @param x The left input vector.
 * @param inc_x The skip value for the left input. May be negative to go in reverse.
 * @param y The right input vector.
 * @param inc_y The skip value for the right input. May be negative to go in reverse.
 * @param a The output matrix.
 * @param lda The leading dimension of @p a.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void gerc(int_t m, int_t n, T alpha, T const *x, int_t inc_x, T const *y, int_t inc_y, T *a, int_t lda);

#ifndef DOXYGEN
template <>
inline void ger<float>(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                                     std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    detail::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                                      std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    detail::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<float>(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<double>(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                                      std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    detail::cgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                                       std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    detail::zgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgetrf(int_t, int_t, float *, int_t, int_t *) -> int_t;
auto EINSUMS_EXPORT dgetrf(int_t, int_t, double *, int_t, int_t *) -> int_t;
auto EINSUMS_EXPORT cgetrf(int_t, int_t, std::complex<float> *, int_t, int_t *) -> int_t;
auto EINSUMS_EXPORT zgetrf(int_t, int_t, std::complex<double> *, int_t, int_t *) -> int_t;
} // namespace detail
#endif

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row int_terchanges.
 *
 * The factorization has the form
 * @f[
 *   \mathbf{A} = \mathbf{PLU}
 * @f]
 * where @f$\mathbf{P}@f$ is a permutation matrix, @f$\mathbf{L}@f$ is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and @f$\mathbf{U}@f$ is upper
 * triangular (upper trapezoidal if m < n).
 *
 * @param m The number of rows in the input.
 * @param n The number of columns in the input.
 * @param a The input matrix. On exit, it contains the upper and lower triangular matrices. The elemnts of the lower
 * triangular matrix are not stored since they are all 1.
 * @param lda The leading dimension of the matrix.
 * @param ipiv The list of pivots.
 *
 * @return 0 on success. If positive, the matrix is singular and the result should not be used for solving systems of equations.
 * The decomposition was performed, though. If negative, one of the inputs had an invalid value. The absolute value indicates
 * which input it is.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto getrf(int_t m, int_t n, T *a, int_t lda, int_t *ipiv) -> int_t;

#ifndef DOXYGEN
template <>
inline auto getrf<float>(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::sgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<double>(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::dgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::cgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::zgetrf(m, n, a, lda, ipiv);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgetri(int_t n, float *a, int_t lda, int_t const *ipiv) -> int_t;
auto EINSUMS_EXPORT dgetri(int_t n, double *a, int_t lda, int_t const *ipiv) -> int_t;
auto EINSUMS_EXPORT cgetri(int_t n, std::complex<float> *a, int_t lda, int_t const *ipiv) -> int_t;
auto EINSUMS_EXPORT zgetri(int_t n, std::complex<double> *a, int_t lda, int_t const *ipiv) -> int_t;
} // namespace detail
#endif

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf.
 *
 * @param n The number of rows and columns of the matrix.
 * @param a The input matrix after being processed by getrf.
 * @param lda The leading dimension of the matrix.
 * @param ipiv The pivots from getrf.
 *
 * @return 0 on success. If positive, the matrix is singular and an inverse could not be computed. If negative,
 * one of the inputs is invalid, and the absolute value indicates which input is bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto getri(int_t n, T *a, int_t lda, int_t const *ipiv) -> int_t;

#ifndef DOXYGEN
template <>
inline auto getri<float>(int_t n, float *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::sgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<double>(int_t n, double *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::dgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<float>>(int_t n, std::complex<float> *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::cgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<double>>(int_t n, std::complex<double> *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::zgetri(n, a, lda, ipiv);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT slange(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float;
auto EINSUMS_EXPORT dlange(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double;
auto EINSUMS_EXPORT clange(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float;
auto EINSUMS_EXPORT zlange(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work) -> double;
} // namespace detail
#endif

/**
 * Computes various matrix norms. The available norms are the 1-norm, Frobenius norm, Max-abs norm, and the infinity norm.
 *
 * @param norm_type The norm to compute. It is case insensitive. For the 1-norm, it should be '1' or 'o'. For the Frobenius norm it should
 * be 'f' or 'e'. For the max-abs norm it should be 'm'. For the infinity norm, it should be 'i'.
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @param A The matrix.
 * @param lda The leading dimension of the matrix.
 * @param work A work array. Only needed for the infinity norm.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto lange(char norm_type, int_t m, int_t n, T const *A, int_t lda, RemoveComplexT<T> *work) -> RemoveComplexT<T>;

#ifndef DOXYGEN
template <>
inline auto lange<float>(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float {
    return detail::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double {
    return detail::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float {
    return detail::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work)
    -> double {
    return detail::zlange(norm_type, m, n, A, lda, work);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT   slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq);
void EINSUMS_EXPORT   dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq);
void EINSUMS_EXPORT   classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq);
void EINSUMS_EXPORT   zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq);
float EINSUMS_EXPORT  snrm2(int_t n, float const *x, int_t incx);
double EINSUMS_EXPORT dnrm2(int_t n, double const *x, int_t incx);
float EINSUMS_EXPORT  scnrm2(int_t n, std::complex<float> const *x, int_t incx);
double EINSUMS_EXPORT dznrm2(int_t n, std::complex<double> const *x, int_t incx);
} // namespace detail
#endif

/**
 * Compute the sum of the squares of the input vector without roundoff error.
 * @f[
 * scale^2 sumsq := \left|\mathbf{x}\right|^2 + scale^2 sumsq
 * @f]
 *
 * @param n The number of elements in the vector.
 * @param x The input vector.
 * @param incx The skip value for the vector.
 * @param scale The scale value used to avoid overflow/underflow. It is also used as an input to continue a previous calculation.
 * @param sumsq The result of the operation, scaled to avoid overflow/underflow. It is also used as an input to continue a previous
 * calculation.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void lassq(int_t n, T const *x, int_t incx, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq);

#ifndef DOXYGEN
template <>
inline void lassq<float>(int_t n, float const *x, int_t incx, float *scale, float *sumsq) {
    detail::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(int_t n, double const *x, int_t incx, double *scale, double *sumsq) {
    detail::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq) {
    detail::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq) {
    detail::zlassq(n, x, incx, scale, sumsq);
}
#endif

/**
 * Compute the Euclidean norm of a vector.
 *
 * @param n The number of elements in the vector.
 * @param x The input vector.
 * @param incx The skip value for the vector.
 *
 * @return The Euclidean norm of the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> nrm2(int_t n, T const *x, int_t incx);

#ifndef DOXYGEN
template <>
inline float nrm2<float>(int_t n, float const *x, int_t incx) {
    return detail::snrm2(n, x, incx);
}

template <>
inline double nrm2<double>(int_t n, double const *x, int_t incx) {
    return detail::dnrm2(n, x, incx);
}

template <>
inline float nrm2<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx) {
    return detail::scnrm2(n, x, incx);
}

template <>
inline double nrm2<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx) {
    return detail::dznrm2(n, x, incx);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgesdd(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t;
auto EINSUMS_EXPORT dgesdd(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt)
    -> int_t;
auto EINSUMS_EXPORT cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
                           std::complex<float> *vt, int_t ldvt) -> int_t;
auto EINSUMS_EXPORT zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
                           std::complex<double> *vt, int_t ldvt) -> int_t;
} // namespace detail
#endif

/**
 * Performs singular value decomposition for a matrix using the divide and conquer algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\Sigma V}^T
 * @f]
 *
 * @param jobz What computation to do. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param m The number of rows of the input matrix.
 * @param n The number of columns of the input matrix.
 * @param a The input matrix.
 * @param lda The leading dimension of the input matrix.
 * @param s The singular values output.
 * @param u The U matrix from the singular value decomposition.
 * @param ldu The leading dimension of U.
 * @param vt The transpose of the V matrix from the singular value decomposition.
 * @param ldvt The leading dimension of the transpose of the V matrix.
 *
 * @return 0 on success. If positive, the algorithm did not converge. If -4, then the input matrix had a NaN entry. If negative otherwise,
 * then one of the parameters had a bad value. The absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gesdd(char jobz, int_t m, int_t n, T *a, int_t lda, RemoveComplexT<T> *s, T *u, int_t ldu, T *vt, int_t ldvt) -> int_t;

#ifndef DOXYGEN
template <>
inline auto gesdd<float>(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t {
    return detail::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt)
    -> int_t {
    return detail::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<float>>(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u,
                                       int_t ldu, std::complex<float> *vt, int_t ldvt) -> int_t {
    return detail::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<double>>(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u,
                                        int_t ldu, std::complex<double> *vt, int_t ldvt) -> int_t {
    return detail::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgesvd(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt,
                           int_t ldvt, float *superb) -> int_t;
auto EINSUMS_EXPORT dgesvd(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                           int_t ldvt, double *superb) -> int_t;
auto EINSUMS_EXPORT cgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u,
                           int_t ldu, std::complex<float> *vt, int_t ldvt, std::complex<float> *superb) -> int_t;
auto EINSUMS_EXPORT zgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u,
                           int_t ldu, std::complex<double> *vt, int_t ldvt, std::complex<double> *superb) -> int_t;
} // namespace detail
#endif

/**
 * Performs singular value decomposition for a matrix using the QR algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\Sigma V}^T
 * @f]
 *
 * @param jobu Whether to compute the U matrix. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param jobvt Whether to compute the transpose of the V matrix. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param m The number of rows of the input matrix.
 * @param n The number of columns of the input matrix.
 * @param a The input matrix.
 * @param lda The leading dimension of the input matrix.
 * @param s The singular values output.
 * @param u The U matrix from the singular value decomposition.
 * @param ldu The leading dimension of U.
 * @param vt The transpose of the V matrix from the singular value decomposition.
 * @param ldvt The leading dimension of the transpose of the V matrix.
 * @param superb Temporary storage area for intermediates in the computation.
 *
 * @return 0 on success. If positive, the algorithm did not converge. If negative,
 * then one of the parameters had a bad value. The absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gesvd(char jobu, char jobvt, int_t m, int_t n, T *a, int_t lda, RemoveComplexT<T> *s, T *u, int_t ldu, T *vt, int_t ldvt, T *superb);

#ifndef DOXYGEN
template <>
inline auto gesvd<float>(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt,
                         float *superb) {
    return detail::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<double>(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                          int_t ldvt, double *superb) {
    return detail::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<std::complex<float>>(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s,
                                       std::complex<float> *u, int_t ldu, std::complex<float> *vt, int_t ldvt,
                                       std::complex<float> *superb) {
    return detail::cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<std::complex<double>>(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s,
                                        std::complex<double> *u, int_t ldu, std::complex<double> *vt, int_t ldvt,
                                        std::complex<double> *superb) {
    return detail::zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}
#endif

namespace detail {
auto EINSUMS_EXPORT sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT cgees(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w,
                          std::complex<float> *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT zgees(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w,
                          std::complex<double> *vs, int_t ldvs) -> int_t;
} // namespace detail

/**
 * Computes the Schur decomposition of a matrix.
 *
 * @param jobvs Whether to compute the unitary matrix for the decomposition.
 * @param n The number of rows and columns of the input matrix.
 * @param a The iput matrix. On exit, it contains the pseudotriangular matrix from the decomposition.
 * @param lda The leading dimension of A.
 * @param sdim The number of selected eigenvalues.
 * @param wr The real components of the eigenvalues.
 * @param wi The imaginary components of the eigenvaules.
 * @param vs The Schur vector matrix.
 * @param ldvs The leading dimension of the Schur vector matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had a bad value. The absolute value tells you which parameter it was.
 * If positive and less than or equal to the number of rows in the matrix, the QR algorithm failed to converge. If one more than the
 * number of rows in the matrix, the eigenvalues could not be reordered for some reason, usually due to eigenvalues being too close.
 * If two more than the number of rows in the matrix, then roundoff changed some of the eigenvalues.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gees(char jobvs, int_t n, T *a, int_t lda, int_t *sdim, T *wr, T *wi, T *vs, int_t ldvs) -> int_t;

#ifndef DOXYGEN
template <>
inline auto gees<float>(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t {
    return detail::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

template <>
inline auto gees<double>(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t {
    return detail::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}
#endif

/**
 * Computes the Schur decomposition of a matrix.
 *
 * @param jobvs Whether to compute the unitary matrix for the decomposition.
 * @param n The number of rows and columns of the input matrix.
 * @param a The iput matrix. On exit, it contains the pseudotriangular matrix from the decomposition.
 * @param lda The leading dimension of A.
 * @param sdim The number of selected eigenvalues.
 * @param w The  eigenvalues.
 * @param vs The Schur vector matrix.
 * @param ldvs The leading dimension of the Schur vector matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had a bad value. The absolute value tells you which parameter it was.
 * If positive and less than or equal to the number of rows in the matrix, the QR algorithm failed to converge. If one more than the
 * number of rows in the matrix, the eigenvalues could not be reordered for some reason, usually due to eigenvalues being too close.
 * If two more than the number of rows in the matrix, then roundoff changed some of the eigenvalues.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
auto gees(char jobvs, int_t n, T *a, int_t lda, int_t *sdim, T *w, T *vs, int_t ldvs) -> int_t;

#ifndef DOXYGEN
template <>
inline auto gees<std::complex<float>>(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w,
                                      std::complex<float> *vs, int_t ldvs) -> int_t {
    return detail::cgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}

template <>
inline auto gees<std::complex<double>>(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w,
                                       std::complex<double> *vs, int_t ldvs) -> int_t {
    return detail::zgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb,
                           float *c, int_t ldc, float *scale) -> int_t;
auto EINSUMS_EXPORT dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb,
                           double *c, int_t ldc, double *scale) -> int_t;
auto EINSUMS_EXPORT ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t;
auto EINSUMS_EXPORT ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *b, int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t;
} // namespace detail
#endif

/**
 * Solves a Sylvester equation. These equations look like the following.
 * @f[
 *  \mathbf{A}\mathbf{X} \pm \mathbf{X}\mathbf{B} = \alpha\mathbf{C}
 * @f]
 *
 * @param trana Whether to transpose the A matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param tranb Whether to transpose the B matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param isgn Whether the sign in the equation is positive or negative.
 * @param m The number of rows in X.
 * @param n The number of columns in X.
 * @param a The A matrix in Schur canonical form.
 * @param lda The leading dimension of A.
 * @param b The B matrix in Schur canonical form.
 * @param ldb The leading dimension of B.
 * @param c The right hand side matrix. On exit, it contains the value of the X matrix that satisfies the equation.
 * @param ldc The leading dimension of the C matrix.
 * @param scale The scale factor for the right hand side matrix.
 *
 * @return 0 on success. If 1, then some eigenvalues were close and needed to be perturbed. If negative, then one of the inputs
 * had a bad value, and the absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, T const *a, int_t lda, T const *b, int_t ldb, T *c, int_t ldc,
           RemoveComplexT<T> *scale) -> int_t;

#ifndef DOXYGEN
template <>
inline auto trsyl<float>(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb,
                         float *c, int_t ldc, float *scale) -> int_t {
    return detail::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb,
                          double *c, int_t ldc, double *scale) -> int_t {
    return detail::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda,
                                       std::complex<float> const *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t {
    return detail::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda,
                                        std::complex<double> const *b, int_t ldb, std::complex<double> *c, int_t ldc, double *scale)
    -> int_t {
    return detail::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
auto EINSUMS_EXPORT dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
auto EINSUMS_EXPORT cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
auto EINSUMS_EXPORT zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;
} // namespace detail
#endif

/**
 * Set up for computing the QR decomposition of a matrix.
 *
 * @f[
 * \mathbf{A} = \mathbf{QR}
 * @f]
 *
 * Here, @f$\mathbf{Q}@f$ is an orthogonal matrix and @f$\mathbf{R}@f$ is an upper triangular matrix.
 *
 * @param m The number of rows in the input matrix.
 * @param n The number of columns in the input matrix.
 * @param a The input matrix. On exit, contains the data needed to compute the Q and R matrices. The
 * entries on and above the diagonal are the entries of the R matrix. The rest is needed to find the Q matrix.
 * @param lda The leading dimension of the input matrix.
 * @param tau On exit, holds the Householder reflector parameters for computing the Q matrix.
 *
 * @return 0 on success. If negative, one of the inputs had a bad value, and the absolute value of the return
 * tells you which one it was.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto geqrf(int_t m, int_t n, T *a, int_t lda, T *tau) -> int_t;

#ifndef DOXYGEN
template <>
inline auto geqrf<float>(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    return detail::sgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<double>(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    return detail::dgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return detail::cgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return detail::zgeqrf(m, n, a, lda, tau);
}
#endif

#ifndef DOXYGEN
namespace detail {
auto EINSUMS_EXPORT sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t;
auto EINSUMS_EXPORT dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t;
auto EINSUMS_EXPORT cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t;
auto EINSUMS_EXPORT zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t;
} // namespace detail
#endif

/**
 * Extract the Q matrix after a call to geqrf.
 *
 * @param m The number of rows of the input matrix.
 * @param n The number of columns in the input matrix.
 * @param k The number of elementary reflectors used in the calculation.
 * @param a The input matrix after being processed by geqrf.
 * @param lda The leading dimension of the input matrix.
 * @param tau The scales for the elementary reflectors from geqrf.
 *
 * @return 0 on success. If negative, then one of the inputs had an invalid value, and the absolute value indicates
 * which parameter it is.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto orgqr(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

#ifndef DOXYGEN
template <>
inline auto orgqr<float>(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t {
    return detail::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t {
    return detail::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

template <>
inline auto ungqr<std::complex<float>>(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau)
    -> int_t {
    return detail::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau)
    -> int_t {
    return detail::zungqr(m, n, k, a, lda, tau);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT scopy(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y);
void EINSUMS_EXPORT dcopy(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y);
void EINSUMS_EXPORT ccopy(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y);
void EINSUMS_EXPORT zcopy(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y);
} // namespace detail
#endif

/**
 * Copy data from one vector to another.
 *
 * @param n The number of elements to copy.
 * @param x The input vector.
 * @param inc_x The skip value for the input vector. If negative, the vector is traversed backwards. If zero, the values are broadcast to
 * the output vector.
 * @param y The output vector.
 * @param inc_y The skip value for the output vector. If negative, the vector is traversed backwards.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void copy(int_t n, T const *x, int_t inc_x, T *y, int_t inc_y);

#ifndef DOXYGEN
template <>
inline void copy<float>(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    detail::scopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<double>(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    detail::dcopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<std::complex<float>>(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    detail::ccopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<std::complex<double>>(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    detail::zcopy(n, x, inc_x, y, inc_y);
}
#endif

#ifndef DOXYGEN
namespace detail {
int_t EINSUMS_EXPORT slascl(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda);
int_t EINSUMS_EXPORT dlascl(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda);
} // namespace detail
#endif

/**
 * Scales a general matrix. The scale factor is <tt> cto / cfrom </tt>, but the scale is performed without overflow/underflow.
 *
 * @param type The type of matrix. Case insensitive. 'g' is for general matrices, 'l' is for lower triangular matrices, 'u' if for upper
 * triangular matrices, 'h' is for hessenberg matrices, 'b' is for symmetric band matrices with lower bandwidth @p kl and upper bandwidth of
 * @p ku and with only the lower half stored, 'q' is the same as 'b' but with the upper half stored instead, and 'z' is the same as 'b' but
 * with a more complicated storage scheme.
 * @param kl The lower bandwidth of the matrix. Only used if the type is 'b', 'q', or 'z'.
 * @param ku The upper bandwidth of the matrix. Only used if the type is 'b', 'q', or 'z'.
 * @param cfrom The denominator for the scale.
 * @param cto The numerator for the scale.
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @param A The matrix being scaled.
 * @param lda The leading dimension of the matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value of the return indicates which
 * parameter it was.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
int_t lascl(char type, int_t kl, int_t ku, T cfrom, T cto, int_t m, int_t n, T *A, int_t lda);

#ifndef DOXYGEN
template <>
inline int_t lascl<float>(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda) {
    return detail::slascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}

template <>
inline int_t lascl<double>(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda) {
    return detail::dlascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz);
void EINSUMS_EXPORT ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz);
void EINSUMS_EXPORT cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y,
                             int_t incy, std::complex<float> *z, int_t incz);
void EINSUMS_EXPORT zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y,
                             int_t incy, std::complex<double> *z, int_t incz);
} // namespace detail
#endif

/**
 * Computes the direct product between two vectors.
 *
 * @f[
 * z_i := z_i + \alpha x_i y_i
 * @f]
 *
 * @param n The number of elements in the vectors.
 * @param alpha The scale factor for the product.
 * @param x The first input vector.
 * @param incx The skip value for the first vector.
 * @param y The second input vector.
 * @param incy The skip value for the second vector.
 * @param z The accumulation vector.
 * @param incz The skip value for the accumulation vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void dirprod(int_t n, T alpha, T const *x, int_t incx, T const *y, int_t incy, T *z, int_t incz);

#ifndef DOXYGEN
template <>
inline void dirprod<float>(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    detail::sdirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<double>(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    detail::ddirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<std::complex<float>>(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx,
                                         std::complex<float> const *y, int_t incy, std::complex<float> *z, int_t incz) {
    detail::cdirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<std::complex<double>>(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx,
                                          std::complex<double> const *y, int_t incy, std::complex<double> *z, int_t incz) {
    detail::zdirprod(n, alpha, x, incx, y, incy, z, incz);
}
#endif

#ifndef DOXYGEN
namespace detail {
float EINSUMS_EXPORT  sasum(int_t n, float const *x, int_t incx);
double EINSUMS_EXPORT dasum(int_t n, double const *x, int_t incx);
float EINSUMS_EXPORT  scasum(int_t n, std::complex<float> const *x, int_t incx);
double EINSUMS_EXPORT dzasum(int_t n, std::complex<double> const *x, int_t incx);
float EINSUMS_EXPORT  scsum1(int_t n, std::complex<float> const *x, int_t incx);
double EINSUMS_EXPORT dzsum1(int_t n, std::complex<double> const *x, int_t incx);
} // namespace detail
#endif

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the absolute values of the components, not the magnitudes.
 *
 * @param n The number of elements.
 * @param x The vector to process.
 * @param incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> asum(int_t n, T const *x, int_t incx);

#ifndef DOXYGEN
template <>
inline float asum(int_t n, float const *x, int_t incx) {
    return detail::sasum(n, x, incx);
}

template <>
inline double asum(int_t n, double const *x, int_t incx) {
    return detail::dasum(n, x, incx);
}

template <>
inline float asum(int_t n, std::complex<float> const *x, int_t incx) {
    return detail::scasum(n, x, incx);
}

template <>
inline double asum(int_t n, std::complex<double> const *x, int_t incx) {
    return detail::dzasum(n, x, incx);
}
#endif

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the magnitudes.
 *
 * @param n The number of elements.
 * @param x The vector to process.
 * @param incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> sum1(int_t n, T const *x, int_t incx);

#ifndef DOXYGEN
template <>
inline float sum1(int_t n, float const *x, int_t incx) {
    return detail::sasum(n, x, incx);
}

template <>
inline double sum1(int_t n, double const *x, int_t incx) {
    return detail::dasum(n, x, incx);
}

template <>
inline float sum1(int_t n, std::complex<float> const *x, int_t incx) {
    return detail::scsum1(n, x, incx);
}

template <>
inline double sum1(int_t n, std::complex<double> const *x, int_t incx) {
    return detail::dzsum1(n, x, incx);
}
#endif

#ifndef DOXYGEN
namespace detail {
void EINSUMS_EXPORT clacgv(int_t n, std::complex<float> *x, int_t incx);
void EINSUMS_EXPORT zlacgv(int_t n, std::complex<double> *x, int_t incx);
} // namespace detail
#endif

/**
 * Take the conjugate of a vector. Does nothing if the vector is real.
 *
 * @param n The number of elements in the vector.
 * @param x The input vector.
 * @param incx The skip value for the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void lacgv(int_t n, T *x, int_t incx);

#ifndef DOXYGEN
template <>
inline void lacgv<float>(int_t n, float *x, int_t incx) {
    // Conjugating real values does nothing.
}

template <>
inline void lacgv<double>(int_t n, double *x, int_t incx) {
    // Conjugating real values does nothing.
}

template <>
inline void lacgv<std::complex<float>>(int_t n, std::complex<float> *x, int_t incx) {
    detail::clacgv(n, x, incx);
}

template <>
inline void lacgv<std::complex<double>>(int_t n, std::complex<double> *x, int_t incx) {
    detail::zlacgv(n, x, incx);
}
#endif

} // namespace einsums::blas