#pragma once

#include "einsums/STL.hpp"
#include "einsums/_Export.hpp"

#include <complex>
#include <vector>

// Namespace for BLAS and LAPACK routines.
namespace einsums::blas {

// Some of the backends may require additional initialization before their use.
void EINSUMS_EXPORT initialize();
void EINSUMS_EXPORT finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
namespace detail {
void EINSUMS_EXPORT sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                          float beta, float *c, int ldc);
void EINSUMS_EXPORT dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb,
                          double beta, double *c, int ldc);
void EINSUMS_EXPORT cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
                          const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
void EINSUMS_EXPORT zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
                          const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);
} // namespace detail

template <typename T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta, T *c, int ldc);

template <>
inline void gemm<float>(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                        float beta, float *c, int ldc) {
    detail::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb,
                         double beta, double *c, int ldc) {
    detail::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, int m, int n, int k, std::complex<float> alpha,
                                      const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
                                      std::complex<float> beta, std::complex<float> *c, int ldc) {
    detail::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, int m, int n, int k, std::complex<double> alpha,
                                       const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
                                       std::complex<double> beta, std::complex<double> *c, int ldc) {
    detail::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/*!
 * Performs matrix vector multiplication.
 */
namespace detail {
void EINSUMS_EXPORT sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y,
                          int incy);
void EINSUMS_EXPORT dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta,
                          double *y, int incy);
void EINSUMS_EXPORT cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda,
                          const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y, int incy);
void EINSUMS_EXPORT zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda,
                          const std::complex<double> *x, int incx, std::complex<double> beta, std::complex<double> *y, int incy);
} // namespace detail

template <typename T>
void gemv(char transa, int m, int n, T alpha, const T *a, int lda, const T *x, int incx, T beta, T *y, int incy);

template <>
inline void gemv<float>(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y,
                        int incy) {
    detail::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta,
                         double *y, int incy) {
    detail::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda,
                                      const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    detail::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda,
                                       const std::complex<double> *x, int incx, std::complex<double> beta, std::complex<double> *y,
                                       int incy) {
    detail::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

/*!
 * Performs symmetric matrix diagonalization.
 */
namespace detail {
auto EINSUMS_EXPORT ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int;
auto EINSUMS_EXPORT dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;
} // namespace detail

template <typename T>
auto syev(char job, char uplo, int n, T *a, int lda, T *w, T *work, int lwork) -> int;

template <>
inline auto syev<float>(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int {
    return detail::ssyev(job, uplo, n, a, lda, w, work, lwork);
}
template <>
inline auto syev<double>(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int {
    return detail::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
namespace detail {
auto EINSUMS_EXPORT cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork,
                          float *rwork) -> int;
auto EINSUMS_EXPORT zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork,
                          double *rworl) -> int;
} // namespace detail

template <typename T>
auto heev(char job, char uplo, int n, std::complex<T> *a, int lda, T *w, std::complex<T> *work, int lwork, T *rwork) -> int;

template <>
inline auto heev<float>(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork,
                        float *rwork) -> int {
    return detail::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

template <>
inline auto heev<double>(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork,
                         double *rwork) -> int {
    return detail::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
namespace detail {
auto EINSUMS_EXPORT sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int;
auto EINSUMS_EXPORT dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int;
auto EINSUMS_EXPORT cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int;
auto EINSUMS_EXPORT zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int;
} // namespace detail

template <typename T>
auto gesv(int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb) -> int;

template <>
inline auto gesv<float>(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int {
    return detail::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<double>(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    return detail::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<float>>(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int {
    return detail::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<double>>(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb)
    -> int {
    return detail::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

namespace detail {
void EINSUMS_EXPORT sscal(int n, const float alpha, float *vec, int inc);
void EINSUMS_EXPORT dscal(int n, const double alpha, double *vec, int inc);
void EINSUMS_EXPORT cscal(int n, const std::complex<float> alpha, std::complex<float> *vec, int inc);
void EINSUMS_EXPORT zscal(int n, const std::complex<double> alpha, std::complex<double> *vec, int inc);
void EINSUMS_EXPORT csscal(int n, const float alpha, std::complex<float> *vec, int inc);
void EINSUMS_EXPORT zdscal(int n, const double alpha, std::complex<double> *vec, int inc);
} // namespace detail

template <typename T>
void scal(int n, const T alpha, T *vec, int inc);

template <typename T>
auto scal(int n, const remove_complex_t<T> alpha, T *vec, int inc) -> std::enable_if_t<is_complex_v<T>>;

template <>
inline void scal<float>(int n, const float alpha, float *vec, int inc) {
    detail::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(int n, const double alpha, double *vec, int inc) {
    detail::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int n, const std::complex<float> alpha, std::complex<float> *vec, int inc) {
    detail::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int n, const std::complex<double> alpha, std::complex<double> *vec, int inc) {
    detail::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int n, const float alpha, std::complex<float> *vec, int inc) {
    detail::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int n, const double alpha, std::complex<double> *vec, int inc) {
    detail::zdscal(n, alpha, vec, inc);
}

namespace detail {
auto EINSUMS_EXPORT sdot(int n, const float *x, int incx, const float *y, int incy) -> float;
auto EINSUMS_EXPORT ddot(int n, const double *x, int incx, const double *y, int incy) -> double;
auto EINSUMS_EXPORT cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float>;
auto EINSUMS_EXPORT zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double>;
} // namespace detail

template <typename T>
auto dot(int n, const T *x, int incx, const T *y, int incy) -> T;

template <>
inline auto dot<float>(int n, const float *x, int incx, const float *y, int incy) -> float {
    return detail::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(int n, const double *x, int incx, const double *y, int incy) -> double {
    return detail::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy)
    -> std::complex<float> {
    return detail::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy)
    -> std::complex<double> {
    return detail::zdot(n, x, incx, y, incy);
}

namespace detail {
void EINSUMS_EXPORT saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y);
void EINSUMS_EXPORT daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);
void EINSUMS_EXPORT caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y);
void EINSUMS_EXPORT zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y,
                          int inc_y);
} // namespace detail

template <typename T>
void axpy(int n, T alpha_x, const T *x, int inc_x, T *y, int inc_y);

template <>
inline void axpy<float>(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y) {
    detail::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    detail::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y,
                                      int inc_y) {
    detail::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x,
                                       std::complex<double> *y, int inc_y) {
    detail::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT sger(int m, int n, float alpha, const float *x, int inc_x, const float *y, int inc_y, float *a, int lda);
void EINSUMS_EXPORT dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);
void EINSUMS_EXPORT cger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int inc_x, const std::complex<float> *y,
                         int inc_y, std::complex<float> *a, int lda);
void EINSUMS_EXPORT zger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int inc_x, const std::complex<double> *y,
                         int inc_y, std::complex<double> *a, int lda);
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
void ger(int m, int n, T alpha, const T *x, int inc_x, const T *y, int inc_y, T *a, int lda);

template <>
inline void ger<float>(int m, int n, float alpha, const float *x, int inc_x, const float *y, int inc_y, float *a, int lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int inc_x,
                                     const std::complex<float> *y, int inc_y, std::complex<float> *a, int lda) {
    detail::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int inc_x,
                                      const std::complex<double> *y, int inc_y, std::complex<double> *a, int lda) {
    detail::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

namespace detail {
auto EINSUMS_EXPORT sgetrf(int, int, float *, int, int *) -> int;
auto EINSUMS_EXPORT dgetrf(int, int, double *, int, int *) -> int;
auto EINSUMS_EXPORT cgetrf(int, int, std::complex<float> *, int, int *) -> int;
auto EINSUMS_EXPORT zgetrf(int, int, std::complex<double> *, int, int *) -> int;
} // namespace detail

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
template <typename T>
auto getrf(int, int, T *, int, int *) -> int;

template <>
inline auto getrf<float>(int m, int n, float *a, int lda, int *ipiv) -> int {
    return detail::sgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<double>(int m, int n, double *a, int lda, int *ipiv) -> int {
    return detail::dgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<float>>(int m, int n, std::complex<float> *a, int lda, int *ipiv) -> int {
    return detail::cgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<double>>(int m, int n, std::complex<double> *a, int lda, int *ipiv) -> int {
    return detail::zgetrf(m, n, a, lda, ipiv);
}

namespace detail {
auto EINSUMS_EXPORT sgetri(int n, float *a, int lda, const int *ipiv) -> int;
auto EINSUMS_EXPORT dgetri(int n, double *a, int lda, const int *ipiv) -> int;
auto EINSUMS_EXPORT cgetri(int n, std::complex<float> *a, int lda, const int *ipiv) -> int;
auto EINSUMS_EXPORT zgetri(int n, std::complex<double> *a, int lda, const int *ipiv) -> int;
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
auto getri(int n, T *a, int lda, const int *ipiv) -> int;

template <>
inline auto getri<float>(int n, float *a, int lda, const int *ipiv) -> int {
    return detail::sgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<double>(int n, double *a, int lda, const int *ipiv) -> int {
    return detail::dgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<float>>(int n, std::complex<float> *a, int lda, const int *ipiv) -> int {
    return detail::cgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<double>>(int n, std::complex<double> *a, int lda, const int *ipiv) -> int {
    return detail::zgetri(n, a, lda, ipiv);
}

/*!
 * Return the value of the 1-norm, Frobenius norm, infinity-norm, or the
 * largest absolute value of any element of a general rectangular matrix
 */
namespace detail {
auto EINSUMS_EXPORT slange(char norm_type, int m, int n, const float *A, int lda, float *work) -> float;
auto EINSUMS_EXPORT dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double;
auto EINSUMS_EXPORT clange(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float;
auto EINSUMS_EXPORT zlange(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double;
} // namespace detail

template <typename T>
auto lange(char norm_type, int m, int n, const T *A, int lda, remove_complex_t<T> *work) -> remove_complex_t<T>;

template <>
inline auto lange<float>(char norm_type, int m, int n, const float *A, int lda, float *work) -> float {
    return detail::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, int m, int n, const double *A, int lda, double *work) -> double {
    return detail::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float {
    return detail::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double {
    return detail::zlange(norm_type, m, n, A, lda, work);
}

namespace detail {
void EINSUMS_EXPORT slassq(int n, const float *x, int incx, float *scale, float *sumsq);
void EINSUMS_EXPORT dlassq(int n, const double *x, int incx, double *scale, double *sumsq);
void EINSUMS_EXPORT classq(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq);
void EINSUMS_EXPORT zlassq(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq);
} // namespace detail

template <typename T>
void lassq(int n, const T *x, int incx, remove_complex_t<T> *scale, remove_complex_t<T> *sumsq);

template <>
inline void lassq<float>(int n, const float *x, int incx, float *scale, float *sumsq) {
    detail::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(int n, const double *x, int incx, double *scale, double *sumsq) {
    detail::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq) {
    detail::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq) {
    detail::zlassq(n, x, incx, scale, sumsq);
}

/*!
 * Computes the singular value decomposition of a general rectangular
 * matrix using a divide and conquer method.
 */
namespace detail {
auto EINSUMS_EXPORT sgesdd(char jobz, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt) -> int;
auto EINSUMS_EXPORT dgesdd(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt) -> int;
auto EINSUMS_EXPORT cgesdd(char jobz, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu,
                           std::complex<float> *vt, int ldvt) -> int;
auto EINSUMS_EXPORT zgesdd(char jobz, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
                           std::complex<double> *vt, int ldvt) -> int;
} // namespace detail

template <typename T>
auto gesdd(char jobz, int m, int n, T *a, int lda, remove_complex_t<T> *s, T *u, int ldu, T *vt, int ldvt) -> int;

template <>
inline auto gesdd<float>(char jobz, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt) -> int {
    return detail::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt) -> int {
    return detail::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<float>>(char jobz, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu,
                                       std::complex<float> *vt, int ldvt) -> int {
    return detail::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<double>>(char jobz, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u,
                                        int ldu, std::complex<double> *vt, int ldvt) -> int {
    return detail::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

namespace detail {
auto EINSUMS_EXPORT sgees(char jobvs, int n, float *a, int lda, int *sdim, float *wr, float *wi, float *vs, int ldvs) -> int;
auto EINSUMS_EXPORT dgees(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int;
} // namespace detail

/*!
 * Computes the Schur Decomposition of a Matrix
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto gees(char jobvs, int n, T *a, int lda, int *sdim, T *wr, T *wi, T *vs, int ldvs) -> int;

template <>
inline auto gees<float>(char jobvs, int n, float *a, int lda, int *sdim, float *wr, float *wi, float *vs, int ldvs) -> int {
    return detail::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

template <>
inline auto gees<double>(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int {
    return detail::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

namespace detail {
auto EINSUMS_EXPORT strsyl(char trana, char tranb, int isgn, int m, int n, const float *a, int lda, const float *b, int ldb, float *c,
                           int ldc, float *scale) -> int;
auto EINSUMS_EXPORT dtrsyl(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c,
                           int ldc, double *scale) -> int;
auto EINSUMS_EXPORT ctrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<float> *a, int lda,
                           const std::complex<float> *b, int ldb, std::complex<float> *c, int ldc, float *scale) -> int;
auto EINSUMS_EXPORT ztrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<double> *a, int lda,
                           const std::complex<double> *b, int ldb, std::complex<double> *c, int ldc, double *scale) -> int;
} // namespace detail

/*!
 * Sylvester Solve
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto trsyl(char trana, char tranb, int isgn, int m, int n, const T *a, int lda, const T *b, int ldb, T *c, int ldc,
           remove_complex_t<T> *scale) -> int;

template <>
inline auto trsyl<float>(char trana, char tranb, int isgn, int m, int n, const float *a, int lda, const float *b, int ldb, float *c,
                         int ldc, float *scale) -> int {
    return detail::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c,
                          int ldc, double *scale) -> int {
    return detail::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, int isgn, int m, int n, const std::complex<float> *a, int lda,
                                       const std::complex<float> *b, int ldb, std::complex<float> *c, int ldc, float *scale) -> int {
    return detail::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, int isgn, int m, int n, const std::complex<double> *a, int lda,
                                        const std::complex<double> *b, int ldb, std::complex<double> *c, int ldc, double *scale) -> int {
    return detail::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

/*!
 * Computes a QR factorizaton (Useful for orthonormalizing matrices)
 */
namespace detail {
auto EINSUMS_EXPORT sgeqrf(int m, int n, float *a, int lda, float *tau) -> int;
auto EINSUMS_EXPORT dgeqrf(int m, int n, double *a, int lda, double *tau) -> int;
auto EINSUMS_EXPORT cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int;
auto EINSUMS_EXPORT zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int;
} // namespace detail

template <typename T>
auto geqrf(int m, int n, T *a, int lda, T *tau) -> int;

template <>
inline auto geqrf<float>(int m, int n, float *a, int lda, float *tau) -> int {
    return detail::sgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<double>(int m, int n, double *a, int lda, double *tau) -> int {
    return detail::dgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<float>>(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int {
    return detail::cgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<double>>(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int {
    return detail::zgeqrf(m, n, a, lda, tau);
}

/*!
 * Returns the orthogonal/unitary matrix Q from the output of dgeqrf
 */
namespace detail {
auto EINSUMS_EXPORT sorgqr(int m, int n, int k, float *a, int lda, const float *tau) -> int;
auto EINSUMS_EXPORT dorgqr(int m, int n, int k, double *a, int lda, const double *tau) -> int;
auto EINSUMS_EXPORT cungqr(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int;
auto EINSUMS_EXPORT zungqr(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int;
} // namespace detail

template <typename T>
auto orgqr(int m, int n, int k, T *a, int lda, const T *tau) -> int;

template <>
inline auto orgqr<float>(int m, int n, int k, float *a, int lda, const float *tau) -> int {
    return detail::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(int m, int n, int k, double *a, int lda, const double *tau) -> int {
    return detail::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(int m, int n, int k, T *a, int lda, const T *tau) -> int;

template <>
inline auto ungqr<std::complex<float>>(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int {
    return detail::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int {
    return detail::zungqr(m, n, k, a, lda, tau);
}

} // namespace einsums::blas