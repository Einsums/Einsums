//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

#include "einsums/utility/ComplexTraits.hpp"

#include <vector>

// Namespace for BLAS and LAPACK routines.
namespace einsums::blas {

/**
 * @brief Initializes the underlying BLAS and LAPACK library.
 *
 * Handles any initialization that the underlying BLAS implementation requires.
 * For example, a GPU implementation would likely need to obtain a device handle to
 * run. That would be handled by this function.
 *
 * You typically will not need to call this function manually. \ref einsums::initialize()
 * will handle calling this function for you.
 *
 */
void EINSUMS_EXPORT initialize();

/**
 * @brief Handles any shutdown procedure needed by the BLAS implementation.
 *
 */
void EINSUMS_EXPORT finalize();

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {
// These routines take care of actually making the call to the BLAS equivalent.
void EINSUMS_EXPORT sgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, float alpha, const float *a, blas_int lda,
                          const float *b, blas_int ldb, float beta, float *c, blas_int ldc);
void EINSUMS_EXPORT dgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, double alpha, const double *a, blas_int lda,
                          const double *b, blas_int ldb, double beta, double *c, blas_int ldc);
void EINSUMS_EXPORT cgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<float> alpha,
                          const std::complex<float> *a, blas_int lda, const std::complex<float> *b, blas_int ldb, std::complex<float> beta,
                          std::complex<float> *c, blas_int ldc);
void EINSUMS_EXPORT zgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<double> alpha,
                          const std::complex<double> *a, blas_int lda, const std::complex<double> *b, blas_int ldb,
                          std::complex<double> beta, std::complex<double> *c, blas_int ldc);
} // namespace detail
#endif

/**
 * @brief Perform a General Matrix Multiply (GEMM) operation.
 *
 * This function computes the product of two matrices,
 * \f[
 * C = alpha * A * B + beta * C,
 * \f]
 * where A, B, and C are matrices, and
 * alpha and beta are scalar values.
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
 */
template <typename T>
void gemm(char transa, char transb, blas_int m, blas_int n, blas_int k, T alpha, const T *a, blas_int lda, const T *b, blas_int ldb, T beta,
          T *c, blas_int ldc);

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
// These are the template specialization for the data types we support. If a unsupported data type
// is attempted a compiler error will occur.
template <>
inline void gemm<float>(char transa, char transb, blas_int m, blas_int n, blas_int k, float alpha, const float *a, blas_int lda,
                        const float *b, blas_int ldb, float beta, float *c, blas_int ldc) {
    detail::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, blas_int m, blas_int n, blas_int k, double alpha, const double *a, blas_int lda,
                         const double *b, blas_int ldb, double beta, double *c, blas_int ldc) {
    detail::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<float> alpha,
                                      const std::complex<float> *a, blas_int lda, const std::complex<float> *b, blas_int ldb,
                                      std::complex<float> beta, std::complex<float> *c, blas_int ldc) {
    detail::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<double> alpha,
                                       const std::complex<double> *a, blas_int lda, const std::complex<double> *b, blas_int ldb,
                                       std::complex<double> beta, std::complex<double> *c, blas_int ldc) {
    detail::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {
void EINSUMS_EXPORT sgemv(char transa, blas_int m, blas_int n, float alpha, const float *a, blas_int lda, const float *x, blas_int incx,
                          float beta, float *y, blas_int incy);
void EINSUMS_EXPORT dgemv(char transa, blas_int m, blas_int n, double alpha, const double *a, blas_int lda, const double *x, blas_int incx,
                          double beta, double *y, blas_int incy);
void EINSUMS_EXPORT cgemv(char transa, blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *a, blas_int lda,
                          const std::complex<float> *x, blas_int incx, std::complex<float> beta, std::complex<float> *y, blas_int incy);
void EINSUMS_EXPORT zgemv(char transa, blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *a, blas_int lda,
                          const std::complex<double> *x, blas_int incx, std::complex<double> beta, std::complex<double> *y, blas_int incy);
} // namespace detail
#endif

/**
 * @brief Computes a matrix-vector product using a general matrix.
 *
 * The gemv routine performs a matrix-vector operation defined as:
 * @f[
 * y := \alpha * A * x + \beta * y
 * @f]
 * or
 * @f[
 * y := \alpha * A' * x + beta * y
 * @f]
 * or
 * @f[
 * y := \alpha * conjg(A') * x + beta * y
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
 */
template <typename T>
void gemv(char transa, blas_int m, blas_int n, T alpha, const T *a, blas_int lda, const T *x, blas_int incx, T beta, T *y, blas_int incy);

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <>
inline void gemv<float>(char transa, blas_int m, blas_int n, float alpha, const float *a, blas_int lda, const float *x, blas_int incx,
                        float beta, float *y, blas_int incy) {
    detail::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, blas_int m, blas_int n, double alpha, const double *a, blas_int lda, const double *x, blas_int incx,
                         double beta, double *y, blas_int incy) {
    detail::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *a,
                                      blas_int lda, const std::complex<float> *x, blas_int incx, std::complex<float> beta,
                                      std::complex<float> *y, blas_int incy) {
    detail::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *a,
                                       blas_int lda, const std::complex<double> *x, blas_int incx, std::complex<double> beta,
                                       std::complex<double> *y, blas_int incy) {
    detail::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif

/*!
 * Performs symmetric matrix diagonalization.
 */
namespace detail {
auto EINSUMS_EXPORT ssyev(char job, char uplo, blas_int n, float *a, blas_int lda, float *w, float *work, blas_int lwork) -> blas_int;
auto EINSUMS_EXPORT dsyev(char job, char uplo, blas_int n, double *a, blas_int lda, double *w, double *work, blas_int lwork) -> blas_int;
} // namespace detail

template <typename T>
auto syev(char job, char uplo, blas_int n, T *a, blas_int lda, T *w, T *work, blas_int lwork) -> blas_int;

template <>
inline auto syev<float>(char job, char uplo, blas_int n, float *a, blas_int lda, float *w, float *work, blas_int lwork) -> blas_int {
    return detail::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

template <>
inline auto syev<double>(char job, char uplo, blas_int n, double *a, blas_int lda, double *w, double *work, blas_int lwork) -> blas_int {
    return detail::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

/*!
 * Performs matrix diagonalization on a general matrix.
 */
namespace detail {
auto EINSUMS_EXPORT sgeev(char jobvl, char jobvr, blas_int n, float *a, blas_int lda, std::complex<float> *w, float *vl, blas_int ldvl,
                          float *vr, blas_int ldvr) -> blas_int;
auto EINSUMS_EXPORT dgeev(char jobvl, char jobvr, blas_int n, double *a, blas_int lda, std::complex<double> *w, double *vl, blas_int ldvl,
                          double *vr, blas_int ldvr) -> blas_int;
auto EINSUMS_EXPORT cgeev(char jobvl, char jobvr, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *w,
                          std::complex<float> *vl, blas_int ldvl, std::complex<float> *vr, blas_int ldvr) -> blas_int;
auto EINSUMS_EXPORT zgeev(char jobvl, char jobvr, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *w,
                          std::complex<double> *vl, blas_int ldvl, std::complex<double> *vr, blas_int ldvr) -> blas_int;
} // namespace detail

// Complex version
template <typename T>
auto geev(char jobvl, char jobvr, blas_int n, T *a, blas_int lda, AddComplexT<T> *w, T *vl, blas_int ldvl, T *vr,
          blas_int ldvr) -> blas_int;

template <>
inline auto geev<float>(char jobvl, char jobvr, blas_int n, float *a, blas_int lda, std::complex<float> *w, float *vl, blas_int ldvl,
                        float *vr, blas_int ldvr) -> blas_int {
    return detail::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<double>(char jobvl, char jobvr, blas_int n, double *a, blas_int lda, std::complex<double> *w, double *vl, blas_int ldvl,
                         double *vr, blas_int ldvr) -> blas_int {
    return detail::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<float>>(char jobvl, char jobvr, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *w,
                                      std::complex<float> *vl, blas_int ldvl, std::complex<float> *vr, blas_int ldvr) -> blas_int {
    return detail::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<double>>(char jobvl, char jobvr, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *w,
                                       std::complex<double> *vl, blas_int ldvl, std::complex<double> *vr, blas_int ldvr) -> blas_int {
    return detail::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
namespace detail {
auto EINSUMS_EXPORT cheev(char job, char uplo, blas_int n, std::complex<float> *a, blas_int lda, float *w, std::complex<float> *work,
                          blas_int lwork, float *rwork) -> blas_int;
auto EINSUMS_EXPORT zheev(char job, char uplo, blas_int n, std::complex<double> *a, blas_int lda, double *w, std::complex<double> *work,
                          blas_int lwork, double *rworl) -> blas_int;
} // namespace detail

template <typename T>
auto heev(char job, char uplo, blas_int n, std::complex<T> *a, blas_int lda, T *w, std::complex<T> *work, blas_int lwork,
          T *rwork) -> blas_int;

template <>
inline auto heev<float>(char job, char uplo, blas_int n, std::complex<float> *a, blas_int lda, float *w, std::complex<float> *work,
                        blas_int lwork, float *rwork) -> blas_int {
    return detail::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

template <>
inline auto heev<double>(char job, char uplo, blas_int n, std::complex<double> *a, blas_int lda, double *w, std::complex<double> *work,
                         blas_int lwork, double *rwork) -> blas_int {
    return detail::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
namespace detail {
auto EINSUMS_EXPORT sgesv(blas_int n, blas_int nrhs, float *a, blas_int lda, blas_int *ipiv, float *b, blas_int ldb) -> blas_int;
auto EINSUMS_EXPORT dgesv(blas_int n, blas_int nrhs, double *a, blas_int lda, blas_int *ipiv, double *b, blas_int ldb) -> blas_int;
auto EINSUMS_EXPORT cgesv(blas_int n, blas_int nrhs, std::complex<float> *a, blas_int lda, blas_int *ipiv, std::complex<float> *b,
                          blas_int ldb) -> blas_int;
auto EINSUMS_EXPORT zgesv(blas_int n, blas_int nrhs, std::complex<double> *a, blas_int lda, blas_int *ipiv, std::complex<double> *b,
                          blas_int ldb) -> blas_int;
} // namespace detail

template <typename T>
auto gesv(blas_int n, blas_int nrhs, T *a, blas_int lda, blas_int *ipiv, T *b, blas_int ldb) -> blas_int;

template <>
inline auto gesv<float>(blas_int n, blas_int nrhs, float *a, blas_int lda, blas_int *ipiv, float *b, blas_int ldb) -> blas_int {
    return detail::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<double>(blas_int n, blas_int nrhs, double *a, blas_int lda, blas_int *ipiv, double *b, blas_int ldb) -> blas_int {
    return detail::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<float>>(blas_int n, blas_int nrhs, std::complex<float> *a, blas_int lda, blas_int *ipiv,
                                      std::complex<float> *b, blas_int ldb) -> blas_int {
    return detail::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<double>>(blas_int n, blas_int nrhs, std::complex<double> *a, blas_int lda, blas_int *ipiv,
                                       std::complex<double> *b, blas_int ldb) -> blas_int {
    return detail::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

namespace detail {
void EINSUMS_EXPORT sscal(blas_int n, const float alpha, float *vec, blas_int inc);
void EINSUMS_EXPORT dscal(blas_int n, const double alpha, double *vec, blas_int inc);
void EINSUMS_EXPORT cscal(blas_int n, const std::complex<float> alpha, std::complex<float> *vec, blas_int inc);
void EINSUMS_EXPORT zscal(blas_int n, const std::complex<double> alpha, std::complex<double> *vec, blas_int inc);
void EINSUMS_EXPORT csscal(blas_int n, const float alpha, std::complex<float> *vec, blas_int inc);
void EINSUMS_EXPORT zdscal(blas_int n, const double alpha, std::complex<double> *vec, blas_int inc);
} // namespace detail

template <typename T>
void scal(blas_int n, const T alpha, T *vec, blas_int inc);

template <Complex T>
void scal(blas_int n, const RemoveComplexT<T> alpha, T *vec, blas_int inc);

template <>
inline void scal<float>(blas_int n, const float alpha, float *vec, blas_int inc) {
    detail::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(blas_int n, const double alpha, double *vec, blas_int inc) {
    detail::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(blas_int n, const std::complex<float> alpha, std::complex<float> *vec, blas_int inc) {
    detail::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(blas_int n, const std::complex<double> alpha, std::complex<double> *vec, blas_int inc) {
    detail::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(blas_int n, const float alpha, std::complex<float> *vec, blas_int inc) {
    detail::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(blas_int n, const double alpha, std::complex<double> *vec, blas_int inc) {
    detail::zdscal(n, alpha, vec, inc);
}

namespace detail {
auto EINSUMS_EXPORT sdot(blas_int n, const float *x, blas_int incx, const float *y, blas_int incy) -> float;
auto EINSUMS_EXPORT ddot(blas_int n, const double *x, blas_int incx, const double *y, blas_int incy) -> double;
auto EINSUMS_EXPORT cdot(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y,
                         blas_int incy) -> std::complex<float>;
auto EINSUMS_EXPORT zdot(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y,
                         blas_int incy) -> std::complex<double>;
auto EINSUMS_EXPORT cdotc(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y,
                         blas_int incy) -> std::complex<float>;
auto EINSUMS_EXPORT zdotc(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y,
                         blas_int incy) -> std::complex<double>;
} // namespace detail

/**
 * Computes the dot product of two vectors. For complex vector it is the non-conjugated dot product;
 * (c|z)dotu in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param n length of the vectors
 * @param x first vector
 * @param incx how many elements to skip in x
 * @param y second vector
 * @param incy how many elements to skip in yo
 * @return result of the dot product
 */
template <typename T>
auto dot(blas_int n, const T *x, blas_int incx, const T *y, blas_int incy) -> T;

template <>
inline auto dot<float>(blas_int n, const float *x, blas_int incx, const float *y, blas_int incy) -> float {
    return detail::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(blas_int n, const double *x, blas_int incx, const double *y, blas_int incy) -> double {
    return detail::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y,
                                     blas_int incy) -> std::complex<float> {
    return detail::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y,
                                      blas_int incy) -> std::complex<double> {
    return detail::zdot(n, x, incx, y, incy);
}

template <typename T>
auto dotc(blas_int n, const T *x, blas_int incx, const T *y, blas_int incy) -> T;

template <>
inline auto dotc<std::complex<float>>(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y,
                                     blas_int incy) -> std::complex<float> {
    return detail::cdotc(n, x, incx, y, incy);
}

template <>
inline auto dotc<std::complex<double>>(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y,
                                      blas_int incy) -> std::complex<double> {
    return detail::zdotc(n, x, incx, y, incy);
}

namespace detail {
void EINSUMS_EXPORT saxpy(blas_int n, float alpha_x, const float *x, blas_int inc_x, float *y, blas_int inc_y);
void EINSUMS_EXPORT daxpy(blas_int n, double alpha_x, const double *x, blas_int inc_x, double *y, blas_int inc_y);
void EINSUMS_EXPORT caxpy(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x, std::complex<float> *y,
                          blas_int inc_y);
void EINSUMS_EXPORT zaxpy(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x, std::complex<double> *y,
                          blas_int inc_y);
} // namespace detail

template <typename T>
void axpy(blas_int n, T alpha_x, const T *x, blas_int inc_x, T *y, blas_int inc_y);

template <>
inline void axpy<float>(blas_int n, float alpha_x, const float *x, blas_int inc_x, float *y, blas_int inc_y) {
    detail::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(blas_int n, double alpha_x, const double *x, blas_int inc_x, double *y, blas_int inc_y) {
    detail::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x,
                                      std::complex<float> *y, blas_int inc_y) {
    detail::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x,
                                       std::complex<double> *y, blas_int inc_y) {
    detail::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT saxpby(blas_int n, float alpha_x, const float *x, blas_int inc_x, float b, float *y, blas_int inc_y);
void EINSUMS_EXPORT daxpby(blas_int n, double alpha_x, const double *x, blas_int inc_x, double b, double *y, blas_int inc_y);
void EINSUMS_EXPORT caxpby(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x, std::complex<float> b,
                           std::complex<float> *y, blas_int inc_y);
void EINSUMS_EXPORT zaxpby(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x, std::complex<double> b,
                           std::complex<double> *y, blas_int inc_y);
} // namespace detail

template <typename T>
void axpby(blas_int n, T alpha_x, const T *x, blas_int inc_x, T b, T *y, blas_int inc_y);

template <>
inline void axpby<float>(blas_int n, float alpha_x, const float *x, blas_int inc_x, float b, float *y, blas_int inc_y) {
    detail::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<double>(blas_int n, double alpha_x, const double *x, blas_int inc_x, double b, double *y, blas_int inc_y) {
    detail::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<float>>(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x,
                                       std::complex<float> b, std::complex<float> *y, blas_int inc_y) {
    detail::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<double>>(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x,
                                        std::complex<double> b, std::complex<double> *y, blas_int inc_y) {
    detail::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT sger(blas_int m, blas_int n, float alpha, const float *x, blas_int inc_x, const float *y, blas_int inc_y, float *a,
                         blas_int lda);
void EINSUMS_EXPORT dger(blas_int m, blas_int n, double alpha, const double *x, blas_int inc_x, const double *y, blas_int inc_y, double *a,
                         blas_int lda);
void EINSUMS_EXPORT cger(blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *x, blas_int inc_x,
                         const std::complex<float> *y, blas_int inc_y, std::complex<float> *a, blas_int lda);
void EINSUMS_EXPORT zger(blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *x, blas_int inc_x,
                         const std::complex<double> *y, blas_int inc_y, std::complex<double> *a, blas_int lda);
} // namespace detail

/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 *    A := alpha*x*y' + A,
 * where:
 *   alpha is a scalar
 *   x is an m-element vector,
 *   y is an n-element vector,
 *   A is an m-by-n general matrix
 */
template <typename T>
void ger(blas_int m, blas_int n, T alpha, const T *x, blas_int inc_x, const T *y, blas_int inc_y, T *a, blas_int lda);

template <>
inline void ger<float>(blas_int m, blas_int n, float alpha, const float *x, blas_int inc_x, const float *y, blas_int inc_y, float *a,
                       blas_int lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(blas_int m, blas_int n, double alpha, const double *x, blas_int inc_x, const double *y, blas_int inc_y, double *a,
                        blas_int lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *x, blas_int inc_x,
                                     const std::complex<float> *y, blas_int inc_y, std::complex<float> *a, blas_int lda) {
    detail::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *x, blas_int inc_x,
                                      const std::complex<double> *y, blas_int inc_y, std::complex<double> *a, blas_int lda) {
    detail::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

namespace detail {
auto EINSUMS_EXPORT sgetrf(blas_int, blas_int, float *, blas_int, blas_int *) -> blas_int;
auto EINSUMS_EXPORT dgetrf(blas_int, blas_int, double *, blas_int, blas_int *) -> blas_int;
auto EINSUMS_EXPORT cgetrf(blas_int, blas_int, std::complex<float> *, blas_int, blas_int *) -> blas_int;
auto EINSUMS_EXPORT zgetrf(blas_int, blas_int, std::complex<double> *, blas_int, blas_int *) -> blas_int;
} // namespace detail

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row blas_interchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
template <typename T>
auto getrf(blas_int, blas_int, T *, blas_int, blas_int *) -> blas_int;

template <>
inline auto getrf<float>(blas_int m, blas_int n, float *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return detail::sgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<double>(blas_int m, blas_int n, double *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return detail::dgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<float>>(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return detail::cgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<double>>(blas_int m, blas_int n, std::complex<double> *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return detail::zgetrf(m, n, a, lda, ipiv);
}

namespace detail {
auto EINSUMS_EXPORT sgetri(blas_int n, float *a, blas_int lda, const blas_int *ipiv) -> blas_int;
auto EINSUMS_EXPORT dgetri(blas_int n, double *a, blas_int lda, const blas_int *ipiv) -> blas_int;
auto EINSUMS_EXPORT cgetri(blas_int n, std::complex<float> *a, blas_int lda, const blas_int *ipiv) -> blas_int;
auto EINSUMS_EXPORT zgetri(blas_int n, std::complex<double> *a, blas_int lda, const blas_int *ipiv) -> blas_int;
} // namespace detail

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
template <typename T>
auto getri(blas_int n, T *a, blas_int lda, const blas_int *ipiv) -> blas_int;

template <>
inline auto getri<float>(blas_int n, float *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return detail::sgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<double>(blas_int n, double *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return detail::dgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<float>>(blas_int n, std::complex<float> *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return detail::cgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<double>>(blas_int n, std::complex<double> *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return detail::zgetri(n, a, lda, ipiv);
}

/*!
 * Return the value of the 1-norm, Frobenius norm, infinity-norm, or the
 * largest absolute value of any element of a general rectangular matrix
 */
namespace detail {
auto EINSUMS_EXPORT slange(char norm_type, blas_int m, blas_int n, const float *A, blas_int lda, float *work) -> float;
auto EINSUMS_EXPORT dlange(char norm_type, blas_int m, blas_int n, const double *A, blas_int lda, double *work) -> double;
auto EINSUMS_EXPORT clange(char norm_type, blas_int m, blas_int n, const std::complex<float> *A, blas_int lda, float *work) -> float;
auto EINSUMS_EXPORT zlange(char norm_type, blas_int m, blas_int n, const std::complex<double> *A, blas_int lda, double *work) -> double;
} // namespace detail

template <typename T>
auto lange(char norm_type, blas_int m, blas_int n, const T *A, blas_int lda, RemoveComplexT<T> *work) -> RemoveComplexT<T>;

template <>
inline auto lange<float>(char norm_type, blas_int m, blas_int n, const float *A, blas_int lda, float *work) -> float {
    return detail::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, blas_int m, blas_int n, const double *A, blas_int lda, double *work) -> double {
    return detail::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, blas_int m, blas_int n, const std::complex<float> *A, blas_int lda,
                                       float *work) -> float {
    return detail::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, blas_int m, blas_int n, const std::complex<double> *A, blas_int lda,
                                        double *work) -> double {
    return detail::zlange(norm_type, m, n, A, lda, work);
}

namespace detail {
void EINSUMS_EXPORT slassq(blas_int n, const float *x, blas_int incx, float *scale, float *sumsq);
void EINSUMS_EXPORT dlassq(blas_int n, const double *x, blas_int incx, double *scale, double *sumsq);
void EINSUMS_EXPORT classq(blas_int n, const std::complex<float> *x, blas_int incx, float *scale, float *sumsq);
void EINSUMS_EXPORT zlassq(blas_int n, const std::complex<double> *x, blas_int incx, double *scale, double *sumsq);
} // namespace detail

template <typename T>
void lassq(blas_int n, const T *x, blas_int incx, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq);

template <>
inline void lassq<float>(blas_int n, const float *x, blas_int incx, float *scale, float *sumsq) {
    detail::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(blas_int n, const double *x, blas_int incx, double *scale, double *sumsq) {
    detail::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(blas_int n, const std::complex<float> *x, blas_int incx, float *scale, float *sumsq) {
    detail::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(blas_int n, const std::complex<double> *x, blas_int incx, double *scale, double *sumsq) {
    detail::zlassq(n, x, incx, scale, sumsq);
}

/*!
 * Computes the singular value decomposition of a general rectangular
 * matrix using a divide and conquer method.
 */
namespace detail {
auto EINSUMS_EXPORT sgesdd(char jobz, blas_int m, blas_int n, float *a, blas_int lda, float *s, float *u, blas_int ldu, float *vt,
                           blas_int ldvt) -> blas_int;
auto EINSUMS_EXPORT dgesdd(char jobz, blas_int m, blas_int n, double *a, blas_int lda, double *s, double *u, blas_int ldu, double *vt,
                           blas_int ldvt) -> blas_int;
auto EINSUMS_EXPORT cgesdd(char jobz, blas_int m, blas_int n, std::complex<float> *a, blas_int lda, float *s, std::complex<float> *u,
                           blas_int ldu, std::complex<float> *vt, blas_int ldvt) -> blas_int;
auto EINSUMS_EXPORT zgesdd(char jobz, blas_int m, blas_int n, std::complex<double> *a, blas_int lda, double *s, std::complex<double> *u,
                           blas_int ldu, std::complex<double> *vt, blas_int ldvt) -> blas_int;
} // namespace detail

template <typename T>
auto gesdd(char jobz, blas_int m, blas_int n, T *a, blas_int lda, RemoveComplexT<T> *s, T *u, blas_int ldu, T *vt,
           blas_int ldvt) -> blas_int;

template <>
inline auto gesdd<float>(char jobz, blas_int m, blas_int n, float *a, blas_int lda, float *s, float *u, blas_int ldu, float *vt,
                         blas_int ldvt) -> blas_int {
    return detail::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, blas_int m, blas_int n, double *a, blas_int lda, double *s, double *u, blas_int ldu, double *vt,
                          blas_int ldvt) -> blas_int {
    return detail::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<float>>(char jobz, blas_int m, blas_int n, std::complex<float> *a, blas_int lda, float *s,
                                       std::complex<float> *u, blas_int ldu, std::complex<float> *vt, blas_int ldvt) -> blas_int {
    return detail::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<double>>(char jobz, blas_int m, blas_int n, std::complex<double> *a, blas_int lda, double *s,
                                        std::complex<double> *u, blas_int ldu, std::complex<double> *vt, blas_int ldvt) -> blas_int {
    return detail::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

namespace detail {
auto EINSUMS_EXPORT sgesvd(char jobu, char jobvt, blas_int m, blas_int n, float *a, blas_int lda, float *s, float *u, blas_int ldu,
                           float *vt, blas_int ldvt, float *superb) -> blas_int;
auto EINSUMS_EXPORT dgesvd(char jobu, char jobvt, blas_int m, blas_int n, double *a, blas_int lda, double *s, double *u, blas_int ldu,
                           double *vt, blas_int ldvt, double *superb) -> blas_int;
} // namespace detail

template <typename T>
auto gesvd(char jobu, char jobvt, blas_int m, blas_int n, T *a, blas_int lda, T *s, T *u, blas_int ldu, T *vt, blas_int ldvt, T *superb);

template <>
inline auto gesvd<float>(char jobu, char jobvt, blas_int m, blas_int n, float *a, blas_int lda, float *s, float *u, blas_int ldu, float *vt,
                         blas_int ldvt, float *superb) {
    return detail::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<double>(char jobu, char jobvt, blas_int m, blas_int n, double *a, blas_int lda, double *s, double *u, blas_int ldu,
                          double *vt, blas_int ldvt, double *superb) {
    return detail::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

namespace detail {
auto EINSUMS_EXPORT sgees(char jobvs, blas_int n, float *a, blas_int lda, blas_int *sdim, float *wr, float *wi, float *vs,
                          blas_int ldvs) -> blas_int;
auto EINSUMS_EXPORT dgees(char jobvs, blas_int n, double *a, blas_int lda, blas_int *sdim, double *wr, double *wi, double *vs,
                          blas_int ldvs) -> blas_int;
} // namespace detail

/*!
 * Computes the Schur Decomposition of a Matrix
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto gees(char jobvs, blas_int n, T *a, blas_int lda, blas_int *sdim, T *wr, T *wi, T *vs, blas_int ldvs) -> blas_int;

template <>
inline auto gees<float>(char jobvs, blas_int n, float *a, blas_int lda, blas_int *sdim, float *wr, float *wi, float *vs,
                        blas_int ldvs) -> blas_int {
    return detail::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

template <>
inline auto gees<double>(char jobvs, blas_int n, double *a, blas_int lda, blas_int *sdim, double *wr, double *wi, double *vs,
                         blas_int ldvs) -> blas_int {
    return detail::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

namespace detail {
auto EINSUMS_EXPORT strsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const float *a, blas_int lda, const float *b,
                           blas_int ldb, float *c, blas_int ldc, float *scale) -> blas_int;
auto EINSUMS_EXPORT dtrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const double *a, blas_int lda, const double *b,
                           blas_int ldb, double *c, blas_int ldc, double *scale) -> blas_int;
auto EINSUMS_EXPORT ctrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const std::complex<float> *a, blas_int lda,
                           const std::complex<float> *b, blas_int ldb, std::complex<float> *c, blas_int ldc, float *scale) -> blas_int;
auto EINSUMS_EXPORT ztrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const std::complex<double> *a, blas_int lda,
                           const std::complex<double> *b, blas_int ldb, std::complex<double> *c, blas_int ldc, double *scale) -> blas_int;
} // namespace detail

/*!
 * Sylvester Solve
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto trsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const T *a, blas_int lda, const T *b, blas_int ldb, T *c,
           blas_int ldc, RemoveComplexT<T> *scale) -> blas_int;

template <>
inline auto trsyl<float>(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const float *a, blas_int lda, const float *b,
                         blas_int ldb, float *c, blas_int ldc, float *scale) -> blas_int {
    return detail::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const double *a, blas_int lda, const double *b,
                          blas_int ldb, double *c, blas_int ldc, double *scale) -> blas_int {
    return detail::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const std::complex<float> *a,
                                       blas_int lda, const std::complex<float> *b, blas_int ldb, std::complex<float> *c, blas_int ldc,
                                       float *scale) -> blas_int {
    return detail::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const std::complex<double> *a,
                                        blas_int lda, const std::complex<double> *b, blas_int ldb, std::complex<double> *c, blas_int ldc,
                                        double *scale) -> blas_int {
    return detail::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

/*!
 * Computes a QR factorizaton (Useful for orthonormalizing matrices)
 */
namespace detail {
auto EINSUMS_EXPORT sgeqrf(blas_int m, blas_int n, float *a, blas_int lda, float *tau) -> blas_int;
auto EINSUMS_EXPORT dgeqrf(blas_int m, blas_int n, double *a, blas_int lda, double *tau) -> blas_int;
auto EINSUMS_EXPORT cgeqrf(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *tau) -> blas_int;
auto EINSUMS_EXPORT zgeqrf(blas_int m, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *tau) -> blas_int;
} // namespace detail

template <typename T>
auto geqrf(blas_int m, blas_int n, T *a, blas_int lda, T *tau) -> blas_int;

template <>
inline auto geqrf<float>(blas_int m, blas_int n, float *a, blas_int lda, float *tau) -> blas_int {
    return detail::sgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<double>(blas_int m, blas_int n, double *a, blas_int lda, double *tau) -> blas_int {
    return detail::dgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<float>>(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *tau) -> blas_int {
    return detail::cgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<double>>(blas_int m, blas_int n, std::complex<double> *a, blas_int lda,
                                        std::complex<double> *tau) -> blas_int {
    return detail::zgeqrf(m, n, a, lda, tau);
}

/*!
 * Returns the orthogonal/unitary matrix Q from the output of dgeqrf
 */
namespace detail {
auto EINSUMS_EXPORT sorgqr(blas_int m, blas_int n, blas_int k, float *a, blas_int lda, const float *tau) -> blas_int;
auto EINSUMS_EXPORT dorgqr(blas_int m, blas_int n, blas_int k, double *a, blas_int lda, const double *tau) -> blas_int;
auto EINSUMS_EXPORT cungqr(blas_int m, blas_int n, blas_int k, std::complex<float> *a, blas_int lda,
                           const std::complex<float> *tau) -> blas_int;
auto EINSUMS_EXPORT zungqr(blas_int m, blas_int n, blas_int k, std::complex<double> *a, blas_int lda,
                           const std::complex<double> *tau) -> blas_int;
} // namespace detail

template <typename T>
auto orgqr(blas_int m, blas_int n, blas_int k, T *a, blas_int lda, const T *tau) -> blas_int;

template <>
inline auto orgqr<float>(blas_int m, blas_int n, blas_int k, float *a, blas_int lda, const float *tau) -> blas_int {
    return detail::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(blas_int m, blas_int n, blas_int k, double *a, blas_int lda, const double *tau) -> blas_int {
    return detail::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(blas_int m, blas_int n, blas_int k, T *a, blas_int lda, const T *tau) -> blas_int;

template <>
inline auto ungqr<std::complex<float>>(blas_int m, blas_int n, blas_int k, std::complex<float> *a, blas_int lda,
                                       const std::complex<float> *tau) -> blas_int {
    return detail::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(blas_int m, blas_int n, blas_int k, std::complex<double> *a, blas_int lda,
                                        const std::complex<double> *tau) -> blas_int {
    return detail::zungqr(m, n, k, a, lda, tau);
}

} // namespace einsums::blas