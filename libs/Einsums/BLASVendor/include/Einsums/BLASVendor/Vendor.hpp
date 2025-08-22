//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#include <complex>

extern "C" {
/**
 * Internal kernel for performing the direct product. The direct product function is not in BLAS, so we have to write our own.
 *
 * @param n The number of elements in the vectors.
 * @param alpha The scale factor for the product.
 * @param x The first input vector.
 * @param y The second input vector.
 * @param z The output vector.
 *
 * @versionadded{2.0.0}
 *
 * @version 2.0.0
 *  Added AVX2 vectorized kernels and default unvectorized kernels.
 */
extern EINSUMS_EXPORT void sdirprod_kernel(size_t n, float alpha, float const *x, float const *y, float *z);

/// @copydoc sdirprod_kernel
extern EINSUMS_EXPORT void ddirprod_kernel(size_t n, double alpha, double const *x, double const *y, double *z);

/// @copydoc sdirprod_kernel
extern EINSUMS_EXPORT void cdirprod_kernel(size_t n, std::complex<float> alpha, std::complex<float> const *x, std::complex<float> const *y,
                                           std::complex<float> *z);

/// @copydoc sdirprod_kernel
extern EINSUMS_EXPORT void zdirprod_kernel(size_t n, std::complex<double> alpha, std::complex<double> const *x,
                                           std::complex<double> const *y, std::complex<double> *z);
}

namespace einsums::blas::vendor {

/**
 * @brief Initializes the BLAS library.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize();

/**
 * @brief Tears down the BLAS library.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void finalize();

/**
 * Performs matrix multiplication for general square matrices.
 *
 * @param transa Whether to transpose the left matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param transb Whether to transpose the right matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param m The number of rows in the output matrix.
 * @param n The number of columns in the output matrix.
 * @param k The number of columns in the first input matrix if @p transa is 'n'. May be different.
 * @param alpha The scale factor for the product.
 * @param a The left matrix.
 * @param lda The leading dimension for the left matrix.
 * @param b The right matrix.
 * @param ldb The leading dimension for the right matrix.
 * @param beta The scale factor for the output matrix.
 * @param c The output matrix.
 * @param ldc The leading dimension for the output matrix.
 *
 * @versionadded{1.0.0}
 */
void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
           float beta, float *c, int_t ldc);
/// @copydoc sgemm
void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
           double beta, double *c, int_t ldc);
/// @copydoc sgemm
void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
/// @copydoc sgemm
void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);

/**
 * Performs matrix vector multiplication.
 *
 * @param transa Whether to transpose the matrix. Case insensitive. Can be 'n', 'c', or 't'.
 * @param m The number of entries in the input vector.
 * @param n The number of entries in the output vector.
 * @param alpha The scale factor for the multiplication.
 * @param a The matrix.
 * @param lda The leading dimension for the matrix.
 * @param x The input vector.
 * @param incx The skip value for the input vector. If it is negative, then the multiplication proceeds in reverse.
 * @param beta The scale factor for the output vector.
 * @param y The output vector.
 * @param incy The skip value for the output vector. If it is negative, then the multiplication proceeds in reverse.
 *
 * @versionadded{1.0.0}
 */
void sgemv(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta, float *y,
           int_t incy);
/// @copydoc sgemv
void dgemv(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta, double *y,
           int_t incy);
/// @copydoc sgemv
void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda, std::complex<float> const *x,
           int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy);
/// @copydoc sgemv
void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy);

/**
 * Scales and adds two vectors.
 *
 * @param n The number of elements in the vectors.
 * @param a The scale factor for the first vector.
 * @param x The input vector.
 * @param incx The skip value for the input vector. If it is negative, then the vector is traversed backwards. If it is zero, then the
 * result will be broadcast to the output vector.
 * @param b The scale factor for the output vector.
 * @param y The output vector.
 * @param incy The skip value for the output vector. If it is negative, then the vector is traversed backwards.
 *
 * @versionadded{1.0.0}
 */
void saxpby(int_t const n, float const a, float const *x, int_t const incx, float const b, float *y, int_t const incy);
/// @copydoc saxpby
void daxpby(int_t const n, double const a, double const *x, int_t const incx, double const b, double *y, int_t const incy);
/// @copydoc saxpby
void caxpby(int_t const n, std::complex<float> const a, std::complex<float> const *x, int_t const incx, std::complex<float> const b,
            std::complex<float> *y, int_t const incy);
/// @copydoc saxpby
void zaxpby(int_t const n, std::complex<double> const a, std::complex<double> const *x, int_t const incx, std::complex<double> const b,
            std::complex<double> *y, int_t const incy);

/**
 * Performs symmetric matrix diagonalization.
 *
 * @param job Whether to compute the eigenvectors. Case insensitive. Can be either 'n' or 'v'.
 * @param uplo Whether the data is stored in the upper or lower triangle of the input. Case insensitive. Can be either 'u' or 'l'.
 * @param n The number of rows and columns of the matrix.
 * @param a The matrix to factor. On exit, its data is overwritten. If the vectors are computed, they are placed in the columns of the
 * matrix.
 * @param lda The leading dimension of the matrix.
 * @param w The output for the eigenvalues.
 * @param work A work array. If @p lwork is -1, then the optimal work array size is placed in the first element of this array.
 * @param lwork The size of the work array. If it is -1, then a workspace query is performed, and the optimal workspace size is put in the
 * first element of @p work.
 *
 * @return 0 on success. If positive, then the eigenvalue algorithm did not converge. If negative, then one of the parameters was given a
 * bad value. The absolute value indicates which parameter it was.
 *
 * @versionadded{1.0.0}
 */
auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t;
/// @copydoc ssyev
auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t;

/**
 * Performs symmetric tridiagonal matrix diagonalization.
 *
 * @param n The number of elements on the diagonal.
 * @param d The diagonal. On exit, it contains the eigenvalues.
 * @param e The subdiagonal.
 *
 * @return 0 on successs. If positive, then the eigenvalue algorithm failed to converge. If negative, then
 * one of the inputs had a bad value. The absolute value indicates which parameter it was.
 *
 * @versionadded{2.0.0}
 */
auto ssterf(int_t n, float *d, float *e) -> int_t;
/// @copydoc ssterf
auto dsterf(int_t n, double *d, double *e) -> int_t;

/*!
 * Computes all eigenvalues and left and right eigenvectors of a general matrix.
 */
auto sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr, int_t ldvr)
    -> int_t;
auto dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr, int_t ldvr)
    -> int_t;
auto cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w, std::complex<float> *vl, int_t ldvl,
           std::complex<float> *vr, int_t ldvr) -> int_t;
auto zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w, std::complex<double> *vl,
           int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork, float *rwork)
    -> int_t;
auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t;
auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t;
auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t;
auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t;

void sscal(int_t n, float alpha, float *vec, int_t inc);
void dscal(int_t n, double alpha, double *vec, int_t inc);
void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc);
void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc);
void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc);

void srscl(int_t n, float alpha, float *vec, int_t inc);
void drscl(int_t n, double alpha, double *vec, int_t inc);
void csrscl(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void zdrscl(int_t n, double alpha, std::complex<double> *vec, int_t inc);

auto sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float;
auto ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double;
auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float>;
auto zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double>;
auto cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float>;
auto zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double>;

void saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y);
void daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y);
void caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y);
void zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y);

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
void sger(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda);
void dger(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda);
void cger(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x, std::complex<float> const *y, int_t inc_y,
          std::complex<float> *a, int_t lda);
void zger(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x, std::complex<double> const *y,
          int_t inc_y, std::complex<double> *a, int_t lda);
void cgerc(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x, std::complex<float> const *y,
           int_t inc_y, std::complex<float> *a, int_t lda);
void zgerc(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x, std::complex<double> const *y,
           int_t inc_y, std::complex<double> *a, int_t lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row int_terchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
auto sgetrf(int_t, int_t, float *, int_t, int_t *) -> int_t;
auto dgetrf(int_t, int_t, double *, int_t, int_t *) -> int_t;
auto cgetrf(int_t, int_t, std::complex<float> *, int_t, int_t *) -> int_t;
auto zgetrf(int_t, int_t, std::complex<double> *, int_t, int_t *) -> int_t;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(int_t, float *, int_t, int_t const *) -> int_t;
auto dgetri(int_t, double *, int_t, int_t const *) -> int_t;
auto cgetri(int_t, std::complex<float> *, int_t, int_t const *) -> int_t;
auto zgetri(int_t, std::complex<double> *, int_t, int_t const *) -> int_t;

auto slange(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float;
auto dlange(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double;
auto clange(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float;
auto zlange(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work) -> double;

void slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq);
void dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq);
void classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq);
void zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq);

float  snrm2(int_t n, float const *x, int_t incx);
double dnrm2(int_t n, double const *x, int_t incx);
float  scnrm2(int_t n, std::complex<float> const *x, int_t incx);
double dznrm2(int_t n, std::complex<double> const *x, int_t incx);

auto sgesvd(char, char, int_t, int_t, float *, int_t, float *, float *, int_t, float *, int_t, float *) -> int_t;
auto dgesvd(char, char, int_t, int_t, double *, int_t, double *, double *, int_t, double *, int_t, double *) -> int_t;
auto cgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt, std::complex<float> *superb) -> int_t;
auto zgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt, std::complex<double> *superb) -> int_t;

auto dgesdd(char, int_t, int_t, double *, int_t, double *, double *, int_t, double *, int_t) -> int_t;
auto sgesdd(char, int_t, int_t, float *, int_t, float *, float *, int_t, float *, int_t) -> int_t;
auto zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt) -> int_t;
auto cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt) -> int_t;

auto dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t;
auto sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t;
auto cgees(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w, std::complex<float> *vs, int_t ldvs)
    -> int_t;
auto zgees(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w, std::complex<double> *vs,
           int_t ldvs) -> int_t;

auto dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb, double *c,
            int_t ldc, double *scale) -> int_t;
auto strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb, float *c, int_t ldc,
            float *scale) -> int_t;
auto ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> const *b,
            int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t;
auto ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> const *b,
            int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t;

auto sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t;
auto dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t;
auto cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t;
auto zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t;

auto dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
auto sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
auto cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
auto zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;

void scopy(int_t n, float const *x, int_t incx, float *y, int_t incy);
void dcopy(int_t n, double const *x, int_t incx, double *y, int_t incy);
void ccopy(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> *y, int_t incy);
void zcopy(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> *y, int_t incy);

int_t slascl(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda);
int_t dlascl(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda);

void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz);
void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz);
void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy,
              std::complex<float> *z, int_t incz);
void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy,
              std::complex<double> *z, int_t incz);

float  sasum(int_t n, float const *x, int_t incx);
double dasum(int_t n, double const *x, int_t incx);
float  scasum(int_t n, std::complex<float> const *x, int_t incx);
double dzasum(int_t n, std::complex<double> const *x, int_t incx);
float  scsum1(int_t n, std::complex<float> const *x, int_t incx);
double dzsum1(int_t n, std::complex<double> const *x, int_t incx);

void clacgv(int_t n, std::complex<float> *x, int_t incx);
void zlacgv(int_t n, std::complex<double> *x, int_t incx);
} // namespace einsums::blas::vendor