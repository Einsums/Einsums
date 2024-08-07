//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Blas.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "backends/linear_algebra/onemkl/onemkl.hpp"
#include "backends/linear_algebra/vendor/Vendor.hpp"

namespace einsums::blas {

void initialize() {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::initialize();
}

void finalize() {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::finalize();
}

namespace detail {
void sgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, float alpha, const float *a, blas_int lda, const float *b,
           blas_int ldb, float beta, float *c, blas_int ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}

void dgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, double alpha, const double *a, blas_int lda, const double *b,
           blas_int ldb, double beta, double *c, blas_int ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}

void cgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<float> alpha, const std::complex<float> *a,
           blas_int lda, const std::complex<float> *b, blas_int ldb, std::complex<float> beta, std::complex<float> *c, blas_int ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}
void zgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<double> alpha, const std::complex<double> *a,
           blas_int lda, const std::complex<double> *b, blas_int ldb, std::complex<double> beta, std::complex<double> *c, blas_int ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}

void sgemv(char transa, blas_int m, blas_int n, float alpha, const float *a, blas_int lda, const float *x, blas_int incx, float beta,
           float *y, blas_int incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, blas_int m, blas_int n, double alpha, const double *a, blas_int lda, const double *x, blas_int incx, double beta,
           double *y, blas_int incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *a, blas_int lda,
           const std::complex<float> *x, blas_int incx, std::complex<float> beta, std::complex<float> *y, blas_int incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv(char transa, blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *a, blas_int lda,
           const std::complex<double> *x, blas_int incx, std::complex<double> beta, std::complex<double> *y, blas_int incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

auto ssyev(char job, char uplo, blas_int n, float *a, blas_int lda, float *w, float *work, blas_int lwork) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

auto dsyev(char job, char uplo, blas_int n, double *a, blas_int lda, double *w, double *work, blas_int lwork) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

auto sgeev(char jobvl, char jobvr, blas_int n, float *a, blas_int lda, std::complex<float> *w, float *vl, blas_int ldvl, float *vr,
           blas_int ldvr) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto dgeev(char jobvl, char jobvr, blas_int n, double *a, blas_int lda, std::complex<double> *w, double *vl, blas_int ldvl, double *vr,
           blas_int ldvr) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto cgeev(char jobvl, char jobvr, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *w, std::complex<float> *vl,
           blas_int ldvl, std::complex<float> *vr, blas_int ldvr) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto zgeev(char jobvl, char jobvr, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *w, std::complex<double> *vl,
           blas_int ldvl, std::complex<double> *vr, blas_int ldvr) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto sgesv(blas_int n, blas_int nrhs, float *a, blas_int lda, blas_int *ipiv, float *b, blas_int ldb) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgesv(blas_int n, blas_int nrhs, double *a, blas_int lda, blas_int *ipiv, double *b, blas_int ldb) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgesv(blas_int n, blas_int nrhs, std::complex<float> *a, blas_int lda, blas_int *ipiv, std::complex<float> *b, blas_int ldb)
    -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto zgesv(blas_int n, blas_int nrhs, std::complex<double> *a, blas_int lda, blas_int *ipiv, std::complex<double> *b, blas_int ldb)
    -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cheev(char job, char uplo, blas_int n, std::complex<float> *a, blas_int lda, float *w, std::complex<float> *work, blas_int lwork,
           float *rwork) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

auto zheev(char job, char uplo, blas_int n, std::complex<double> *a, blas_int lda, double *w, std::complex<double> *work, blas_int lwork,
           double *rwork) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

void sscal(blas_int n, float alpha, float *vec, blas_int inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sscal(n, alpha, vec, inc);
}

void dscal(blas_int n, double alpha, double *vec, blas_int inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dscal(n, alpha, vec, inc);
}

void cscal(blas_int n, std::complex<float> alpha, std::complex<float> *vec, blas_int inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cscal(n, alpha, vec, inc);
}

void zscal(blas_int n, std::complex<double> alpha, std::complex<double> *vec, blas_int inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zscal(n, alpha, vec, inc);
}

void csscal(blas_int n, float alpha, std::complex<float> *vec, blas_int inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::csscal(n, alpha, vec, inc);
}

void zdscal(blas_int n, double alpha, std::complex<double> *vec, blas_int inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zdscal(n, alpha, vec, inc);
}

auto sdot(blas_int n, const float *x, blas_int incx, const float *y, blas_int incy) -> float {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sdot(n, x, incx, y, incy);
}

auto ddot(blas_int n, const double *x, blas_int incx, const double *y, blas_int incy) -> double {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::ddot(n, x, incx, y, incy);
}

auto cdot(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y, blas_int incy) -> std::complex<float> {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cdot(n, x, incx, y, incy);
}

auto zdot(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y, blas_int incy) -> std::complex<double> {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zdot(n, x, incx, y, incy);
}

auto cdotc(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y, blas_int incy) -> std::complex<float> {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cdotc(n, x, incx, y, incy);
}

auto zdotc(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y, blas_int incy) -> std::complex<double> {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zdotc(n, x, incx, y, incy);
}

void saxpy(blas_int n, float alpha_x, const float *x, blas_int inc_x, float *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void daxpy(blas_int n, double alpha_x, const double *x, blas_int inc_x, double *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void caxpy(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x, std::complex<float> *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void zaxpy(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x, std::complex<double> *y,
           blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void saxpby(blas_int n, float alpha_x, const float *x, blas_int inc_x, float b, float *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void daxpby(blas_int n, double alpha_x, const double *x, blas_int inc_x, double b, double *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void caxpby(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x, std::complex<float> b,
            std::complex<float> *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void zaxpby(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x, std::complex<double> b,
            std::complex<double> *y, blas_int inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void sger(blas_int m, blas_int n, float alpha, const float *x, blas_int inc_x, const float *y, blas_int inc_y, float *a, blas_int lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void dger(blas_int m, blas_int n, double alpha, const double *x, blas_int inc_x, const double *y, blas_int inc_y, double *a, blas_int lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void cger(blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *x, blas_int inc_x, const std::complex<float> *y,
          blas_int inc_y, std::complex<float> *a, blas_int lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void zger(blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *x, blas_int inc_x, const std::complex<double> *y,
          blas_int inc_y, std::complex<double> *a, blas_int lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

auto sgetrf(blas_int m, blas_int n, float *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgetrf(m, n, a, lda, ipiv);
}

auto dgetrf(blas_int m, blas_int n, double *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgetrf(m, n, a, lda, ipiv);
}

auto cgetrf(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgetrf(m, n, a, lda, ipiv);
}

auto zgetrf(blas_int m, blas_int n, std::complex<double> *a, blas_int lda, blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgetrf(m, n, a, lda, ipiv);
}

auto sgetri(blas_int n, float *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgetri(n, a, lda, ipiv);
}

auto dgetri(blas_int n, double *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgetri(n, a, lda, ipiv);
}

auto cgetri(blas_int n, std::complex<float> *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgetri(n, a, lda, ipiv);
}

auto zgetri(blas_int n, std::complex<double> *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgetri(n, a, lda, ipiv);
}

auto slange(char norm_type, blas_int m, blas_int n, const float *A, blas_int lda, float *work) -> float {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::slange(norm_type, m, n, A, lda, work);
}

auto dlange(char norm_type, blas_int m, blas_int n, const double *A, blas_int lda, double *work) -> double {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dlange(norm_type, m, n, A, lda, work);
}

auto clange(char norm_type, blas_int m, blas_int n, const std::complex<float> *A, blas_int lda, float *work) -> float {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::clange(norm_type, m, n, A, lda, work);
}

auto zlange(char norm_type, blas_int m, blas_int n, const std::complex<double> *A, blas_int lda, double *work) -> double {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zlange(norm_type, m, n, A, lda, work);
}

void slassq(blas_int n, const float *x, blas_int incx, float *scale, float *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::slassq(n, x, incx, scale, sumsq);
}

void dlassq(blas_int n, const double *x, blas_int incx, double *scale, double *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dlassq(n, x, incx, scale, sumsq);
}

void classq(blas_int n, const std::complex<float> *x, blas_int incx, float *scale, float *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::classq(n, x, incx, scale, sumsq);
}

void zlassq(blas_int n, const std::complex<double> *x, blas_int incx, double *scale, double *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zlassq(n, x, incx, scale, sumsq);
}

auto sgesdd(char jobz, blas_int m, blas_int n, float *a, blas_int lda, float *s, float *u, blas_int ldu, float *vt, blas_int ldvt)
    -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgesdd(char jobz, blas_int m, blas_int n, double *a, blas_int lda, double *s, double *u, blas_int ldu, double *vt, blas_int ldvt)
    -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto cgesdd(char jobz, blas_int m, blas_int n, std::complex<float> *a, blas_int lda, float *s, std::complex<float> *u, blas_int ldu,
            std::complex<float> *vt, blas_int ldvt) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto zgesdd(char jobz, blas_int m, blas_int n, std::complex<double> *a, blas_int lda, double *s, std::complex<double> *u, blas_int ldu,
            std::complex<double> *vt, blas_int ldvt) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto sgesvd(char jobu, char jobvt, blas_int m, blas_int n, float *a, blas_int lda, float *s, float *u, blas_int ldu, float *vt,
            blas_int ldvt, float *superb) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                                                                        superb);
}

auto dgesvd(char jobu, char jobvt, blas_int m, blas_int n, double *a, blas_int lda, double *s, double *u, blas_int ldu, double *vt,
            blas_int ldvt, double *superb) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                                                                        superb);
}

auto dgees(char jobvs, blas_int n, double *a, blas_int lda, blas_int *sdim, double *wr, double *wi, double *vs, blas_int ldvs) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto strsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const float *a, blas_int lda, const float *b, blas_int ldb,
            float *c, blas_int ldc, float *scale) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc,
                                                                                        scale);
}

auto dtrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const double *a, blas_int lda, const double *b, blas_int ldb,
            double *c, blas_int ldc, double *scale) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc,
                                                                                        scale);
}

auto ctrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const std::complex<float> *a, blas_int lda,
            const std::complex<float> *b, blas_int ldb, std::complex<float> *c, blas_int ldc, float *scale) -> blas_int {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("ctrsyl not implemented.");
#endif
}

auto ztrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const std::complex<double> *a, blas_int lda,
            const std::complex<double> *b, blas_int ldb, std::complex<double> *c, blas_int ldc, double *scale) -> blas_int {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("ztrsyl not implemented.");
#endif
}

auto sgeqrf(blas_int m, blas_int n, float *a, blas_int lda, float *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgeqrf(m, n, a, lda, tau);
}

auto dgeqrf(blas_int m, blas_int n, double *a, blas_int lda, double *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgeqrf(m, n, a, lda, tau);
}

auto cgeqrf(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgeqrf(m, n, a, lda, tau);
}

auto zgeqrf(blas_int m, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgeqrf(m, n, a, lda, tau);
}

auto sorgqr(blas_int m, blas_int n, blas_int k, float *a, blas_int lda, const float *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sorgqr(m, n, k, a, lda, tau);
}

auto dorgqr(blas_int m, blas_int n, blas_int k, double *a, blas_int lda, const double *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dorgqr(m, n, k, a, lda, tau);
}

auto cungqr(blas_int m, blas_int n, blas_int k, std::complex<float> *a, blas_int lda, const std::complex<float> *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cungqr(m, n, k, a, lda, tau);
}

auto zungqr(blas_int m, blas_int n, blas_int k, std::complex<double> *a, blas_int lda, const std::complex<double> *tau) -> blas_int {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zungqr(m, n, k, a, lda, tau);
}

} // namespace detail
} // namespace einsums::blas