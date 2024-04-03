//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"

#include <complex>

namespace einsums::backend::linear_algebra::vendor {

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
void sgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, float alpha, const float *a, blas_int lda, const float *b,
           blas_int ldb, float beta, float *c, blas_int ldc);
void dgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, double alpha, const double *a, blas_int lda, const double *b,
           blas_int ldb, double beta, double *c, blas_int ldc);
void cgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<float> alpha, const std::complex<float> *a,
           blas_int lda, const std::complex<float> *b, blas_int ldb, std::complex<float> beta, std::complex<float> *c, blas_int ldc);
void zgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<double> alpha, const std::complex<double> *a,
           blas_int lda, const std::complex<double> *b, blas_int ldb, std::complex<double> beta, std::complex<double> *c, blas_int ldc);

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char transa, blas_int m, blas_int n, float alpha, const float *a, blas_int lda, const float *x, blas_int incx, float beta,
           float *y, blas_int incy);
void dgemv(char transa, blas_int m, blas_int n, double alpha, const double *a, blas_int lda, const double *x, blas_int incx, double beta,
           double *y, blas_int incy);
void cgemv(char transa, blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *a, blas_int lda,
           const std::complex<float> *x, blas_int incx, std::complex<float> beta, std::complex<float> *y, blas_int incy);
void zgemv(char transa, blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *a, blas_int lda,
           const std::complex<double> *x, blas_int incx, std::complex<double> beta, std::complex<double> *y, blas_int incy);

void saxpby(const blas_int n, const float a, const float *x, const blas_int incx, const float b, float *y, const blas_int incy);
void daxpby(const blas_int n, const double a, const double *x, const blas_int incx, const double b, double *y, const blas_int incy);
void caxpby(const blas_int n, const std::complex<float> a, const std::complex<float> *x, const blas_int incx, const std::complex<float> b,
            std::complex<float> *y, const blas_int incy);
void zaxpby(const blas_int n, const std::complex<double> a, const std::complex<double> *x, const blas_int incx,
            const std::complex<double> b, std::complex<double> *y, const blas_int incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char job, char uplo, blas_int n, float *a, blas_int lda, float *w, float *work, blas_int lwork) -> blas_int;
auto dsyev(char job, char uplo, blas_int n, double *a, blas_int lda, double *w, double *work, blas_int lwork) -> blas_int;

/*!
 * Computes all eigenvalues and left and right eigenvectors of a general matrix.
 */
auto sgeev(char jobvl, char jobvr, blas_int n, float *a, blas_int lda, std::complex<float> *w, float *vl, blas_int ldvl, float *vr,
           blas_int ldvr) -> blas_int;
auto dgeev(char jobvl, char jobvr, blas_int n, double *a, blas_int lda, std::complex<double> *w, double *vl, blas_int ldvl, double *vr,
           blas_int ldvr) -> blas_int;
auto cgeev(char jobvl, char jobvr, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *w, std::complex<float> *vl,
           blas_int ldvl, std::complex<float> *vr, blas_int ldvr) -> blas_int;
auto zgeev(char jobvl, char jobvr, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *w, std::complex<double> *vl,
           blas_int ldvl, std::complex<double> *vr, blas_int ldvr) -> blas_int;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, blas_int n, std::complex<float> *a, blas_int lda, float *w, std::complex<float> *work, blas_int lwork,
           float *rwork) -> blas_int;
auto zheev(char job, char uplo, blas_int n, std::complex<double> *a, blas_int lda, double *w, std::complex<double> *work, blas_int lwork,
           double *rwork) -> blas_int;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(blas_int n, blas_int nrhs, float *a, blas_int lda, blas_int *ipiv, float *b, blas_int ldb) -> blas_int;
auto dgesv(blas_int n, blas_int nrhs, double *a, blas_int lda, blas_int *ipiv, double *b, blas_int ldb) -> blas_int;
auto cgesv(blas_int n, blas_int nrhs, std::complex<float> *a, blas_int lda, blas_int *ipiv, std::complex<float> *b, blas_int ldb)
    -> blas_int;
auto zgesv(blas_int n, blas_int nrhs, std::complex<double> *a, blas_int lda, blas_int *ipiv, std::complex<double> *b, blas_int ldb)
    -> blas_int;

void sscal(blas_int n, float alpha, float *vec, blas_int inc);
void dscal(blas_int n, double alpha, double *vec, blas_int inc);
void cscal(blas_int n, std::complex<float> alpha, std::complex<float> *vec, blas_int inc);
void zscal(blas_int n, std::complex<double> alpha, std::complex<double> *vec, blas_int inc);
void csscal(blas_int n, float alpha, std::complex<float> *vec, blas_int inc);
void zdscal(blas_int n, double alpha, std::complex<double> *vec, blas_int inc);

auto sdot(blas_int n, const float *x, blas_int incx, const float *y, blas_int incy) -> float;
auto ddot(blas_int n, const double *x, blas_int incx, const double *y, blas_int incy) -> double;
auto cdot(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y, blas_int incy) -> std::complex<float>;
auto zdot(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y, blas_int incy) -> std::complex<double>;

void saxpy(blas_int n, float alpha_x, const float *x, blas_int inc_x, float *y, blas_int inc_y);
void daxpy(blas_int n, double alpha_x, const double *x, blas_int inc_x, double *y, blas_int inc_y);
void caxpy(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x, std::complex<float> *y, blas_int inc_y);
void zaxpy(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x, std::complex<double> *y,
           blas_int inc_y);

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
void sger(blas_int m, blas_int n, float alpha, const float *x, blas_int inc_x, const float *y, blas_int inc_y, float *a, blas_int lda);
void dger(blas_int m, blas_int n, double alpha, const double *x, blas_int inc_x, const double *y, blas_int inc_y, double *a, blas_int lda);
void cger(blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *x, blas_int inc_x, const std::complex<float> *y,
          blas_int inc_y, std::complex<float> *a, blas_int lda);
void zger(blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *x, blas_int inc_x, const std::complex<double> *y,
          blas_int inc_y, std::complex<double> *a, blas_int lda);

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
auto sgetrf(blas_int, blas_int, float *, blas_int, blas_int *) -> blas_int;
auto dgetrf(blas_int, blas_int, double *, blas_int, blas_int *) -> blas_int;
auto cgetrf(blas_int, blas_int, std::complex<float> *, blas_int, blas_int *) -> blas_int;
auto zgetrf(blas_int, blas_int, std::complex<double> *, blas_int, blas_int *) -> blas_int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(blas_int, float *, blas_int, const blas_int *) -> blas_int;
auto dgetri(blas_int, double *, blas_int, const blas_int *) -> blas_int;
auto cgetri(blas_int, std::complex<float> *, blas_int, const blas_int *) -> blas_int;
auto zgetri(blas_int, std::complex<double> *, blas_int, const blas_int *) -> blas_int;

auto slange(char norm_type, blas_int m, blas_int n, const float *A, blas_int lda, float *work) -> float;
auto dlange(char norm_type, blas_int m, blas_int n, const double *A, blas_int lda, double *work) -> double;
auto clange(char norm_type, blas_int m, blas_int n, const std::complex<float> *A, blas_int lda, float *work) -> float;
auto zlange(char norm_type, blas_int m, blas_int n, const std::complex<double> *A, blas_int lda, double *work) -> double;

void slassq(blas_int n, const float *x, blas_int incx, float *scale, float *sumsq);
void dlassq(blas_int n, const double *x, blas_int incx, double *scale, double *sumsq);
void classq(blas_int n, const std::complex<float> *x, blas_int incx, float *scale, float *sumsq);
void zlassq(blas_int n, const std::complex<double> *x, blas_int incx, double *scale, double *sumsq);

auto sgesvd(char, char, blas_int, blas_int, float *, blas_int, float *, float *, blas_int, float *, blas_int, float *) -> blas_int;
auto dgesvd(char, char, blas_int, blas_int, double *, blas_int, double *, double *, blas_int, double *, blas_int, double *) -> blas_int;

auto dgesdd(char, blas_int, blas_int, double *, blas_int, double *, double *, blas_int, double *, blas_int) -> blas_int;
auto sgesdd(char, blas_int, blas_int, float *, blas_int, float *, float *, blas_int, float *, blas_int) -> blas_int;
auto zgesdd(char jobz, blas_int m, blas_int n, std::complex<double> *a, blas_int lda, double *s, std::complex<double> *u, blas_int ldu,
            std::complex<double> *vt, blas_int ldvt) -> blas_int;
auto cgesdd(char jobz, blas_int m, blas_int n, std::complex<float> *a, blas_int lda, float *s, std::complex<float> *u, blas_int ldu,
            std::complex<float> *vt, blas_int ldvt) -> blas_int;

auto dgees(char jobvs, blas_int n, double *a, blas_int lda, blas_int *sdim, double *wr, double *wi, double *vs, blas_int ldvs) -> blas_int;
auto sgees(char jobvs, blas_int n, float *a, blas_int lda, blas_int *sdim, float *wr, float *wi, float *vs, blas_int ldvs) -> blas_int;

auto dtrsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const double *a, blas_int lda, const double *b, blas_int ldb,
            double *c, blas_int ldc, double *scale) -> blas_int;
auto strsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const float *a, blas_int lda, const float *b, blas_int ldb,
            float *c, blas_int ldc, float *scale) -> blas_int;

auto sorgqr(blas_int m, blas_int n, blas_int k, float *a, blas_int lda, const float *tau) -> blas_int;
auto dorgqr(blas_int m, blas_int n, blas_int k, double *a, blas_int lda, const double *tau) -> blas_int;
auto cungqr(blas_int m, blas_int n, blas_int k, std::complex<float> *a, blas_int lda, const std::complex<float> *tau) -> blas_int;
auto zungqr(blas_int m, blas_int n, blas_int k, std::complex<double> *a, blas_int lda, const std::complex<double> *tau) -> blas_int;

auto dgeqrf(blas_int m, blas_int n, double *a, blas_int lda, double *tau) -> blas_int;
auto sgeqrf(blas_int m, blas_int n, float *a, blas_int lda, float *tau) -> blas_int;
auto cgeqrf(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, std::complex<float> *tau) -> blas_int;
auto zgeqrf(blas_int m, blas_int n, std::complex<double> *a, blas_int lda, std::complex<double> *tau) -> blas_int;

} // namespace einsums::backend::linear_algebra::vendor
