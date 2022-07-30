#include "einsums/Blas.hpp"

#include "backends/cblas/cblas.hpp"
#include "backends/netlib/Netlib.hpp"
#include "backends/onemkl/onemkl.hpp"
#include "backends/vendor/Vendor.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace einsums::blas {

void initialize() {
#if defined(SYCL_LANGUAGE_VERSION)
    ::einsums::backend::onemkl::initialize();
#endif
}

void finalize() {
#if defined(SYCL_LANGUAGE_VERSION)
    ::einsums::backend::onemkl::finalize();
#endif
}

namespace detail {
void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc) {
    ::einsums::backend::vendor::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc) {
    ::einsums::backend::vendor::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    ::einsums::backend::vendor::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    ::einsums::backend::vendor::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    ::einsums::backend::vendor::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    ::einsums::backend::vendor::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    ::einsums::backend::vendor::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy) {
    ::einsums::backend::vendor::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int {
    return ::einsums::backend::vendor::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int {
    return ::einsums::backend::vendor::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int {
    return ::einsums::backend::vendor::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    return ::einsums::backend::vendor::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int {
    return ::einsums::backend::vendor::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int {
    return ::einsums::backend::vendor::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *rwork)
    -> int {
    return ::einsums::backend::vendor::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rwork)
    -> int {
    return ::einsums::backend::vendor::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

void sscal(int n, float alpha, float *vec, int inc) {
    ::einsums::backend::vendor::sscal(n, alpha, vec, inc);
}

void dscal(int n, double alpha, double *vec, int inc) {
    ::einsums::backend::vendor::dscal(n, alpha, vec, inc);
}

void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc) {
    ::einsums::backend::vendor::cscal(n, alpha, vec, inc);
}

void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc) {
    ::einsums::backend::vendor::zscal(n, alpha, vec, inc);
}

void csscal(int n, float alpha, std::complex<float> *vec, int inc) {
    ::einsums::backend::vendor::csscal(n, alpha, vec, inc);
}

void zdscal(int n, double alpha, std::complex<double> *vec, int inc) {
    ::einsums::backend::vendor::zdscal(n, alpha, vec, inc);
}

auto sdot(int n, const float *x, int incx, const float *y, int incy) -> float {
    return ::einsums::backend::vendor::sdot(n, x, incx, y, incy);
}

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double {
    return ::einsums::backend::vendor::ddot(n, x, incx, y, incy);
}

auto cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float> {
    return ::einsums::backend::vendor::cdot(n, x, incx, y, incy);
}

auto zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double> {
    return ::einsums::backend::vendor::zdot(n, x, incx, y, incy);
}

void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y) {
    ::einsums::backend::vendor::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    ::einsums::backend::vendor::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y) {
    ::einsums::backend::vendor::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y) {
    ::einsums::backend::vendor::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void sdger(int m, int n, float alpha, const float *x, int inc_x, const float *y, int inc_y, float *a, int lda) {
    ::einsums::backend::vendor::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    ::einsums::backend::vendor::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void cger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int inc_x, const std::complex<float> *y, int inc_y,
          std::complex<float> *a, int lda) {
    ::einsums::backend::vendor::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void zger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int inc_x, const std::complex<double> *y, int inc_y,
          std::complex<double> *a, int lda) {
    ::einsums::backend::vendor::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

auto sgetrf(int m, int n, float *a, int lda, int *ipiv) -> int {
    return ::einsums::backend::vendor::sgetrf(m, n, a, lda, ipiv);
}

auto dgetrf(int m, int n, double *a, int lda, int *ipiv) -> int {
    return ::einsums::backend::vendor::dgetrf(m, n, a, lda, ipiv);
}

auto cgetrf(int m, int n, std::complex<float> *a, int lda, int *ipiv) -> int {
    return ::einsums::backend::vendor::cgetrf(m, n, a, lda, ipiv);
}

auto zgetrf(int m, int n, std::complex<double> *a, int lda, int *ipiv) -> int {
    return ::einsums::backend::vendor::zgetrf(m, n, a, lda, ipiv);
}

auto sgetri(int n, float *a, int lda, const int *ipiv) -> int {
    return ::einsums::backend::vendor::sgetri(n, a, lda, ipiv);
}

auto dgetri(int n, double *a, int lda, const int *ipiv) -> int {
    return ::einsums::backend::vendor::dgetri(n, a, lda, ipiv);
}

auto cgetri(int n, std::complex<float> *a, int lda, const int *ipiv) -> int {
    return ::einsums::backend::vendor::cgetri(n, a, lda, ipiv);
}

auto zgetri(int n, std::complex<double> *a, int lda, const int *ipiv) -> int {
    return ::einsums::backend::vendor::zgetri(n, a, lda, ipiv);
}

auto slange(char norm_type, int m, int n, const float *A, int lda, float *work) -> float {
    return ::einsums::backend::vendor::slange(norm_type, n, m, A, lda, work);
}

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double {
    return ::einsums::backend::vendor::dlange(norm_type, n, m, A, lda, work);
}

auto clange(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float {
    return ::einsums::backend::vendor::clange(norm_type, n, m, A, lda, work);
}

auto zlange(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double {
    return ::einsums::backend::vendor::zlange(norm_type, n, m, A, lda, work);
}

void slassq(int n, const float *x, int incx, float *scale, float *sumsq) {
    return ::einsums::backend::vendor::slassq(n, x, incx, scale, sumsq);
}

void dlassq(int n, const double *x, int incx, double *scale, double *sumsq) {
    return ::einsums::backend::vendor::dlassq(n, x, incx, scale, sumsq);
}

void classq(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq) {
    return ::einsums::backend::vendor::classq(n, x, incx, scale, sumsq);
}

void zlassq(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq) {
    return ::einsums::backend::vendor::zlassq(n, x, incx, scale, sumsq);
}

auto sgesdd(char jobz, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto dgesdd(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto cgesdd(char jobz, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu, std::complex<float> *vt,
            int ldvt) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto zgesdd(char jobz, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
            std::complex<double> *vt, int ldvt) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto dgees(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
#else
    throw std::runtime_error("dgees not implemented.");
#endif
}

auto strsyl(char trana, char tranb, int isgn, int m, int n, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
            float *scale) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto dtrsyl(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c, int ldc,
            double *scale) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto ctrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
            std::complex<float> *c, int ldc, float *scale) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto ztrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
            std::complex<double> *c, int ldc, double *scale) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto sgeqrf(int m, int n, float *a, int lda, float *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::sgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto dgeqrf(int m, int n, double *a, int lda, double *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::dgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::cgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::zgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto sorgqr(int m, int n, int k, float *a, int lda, const float *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::sorgqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

auto dorgqr(int m, int n, int k, double *a, int lda, const double *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::dorgqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

auto cungqr(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::cungqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

auto zungqr(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int {
#if defined(EINSUMS_HAVE_LAPACKE_H) || defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::cblas::zungqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

} // namespace detail
} // namespace einsums::blas