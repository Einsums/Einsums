//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
//----------------------------------------------------------------------------------------------
/**
 * @file hipblas.hpp
 *
 * This file defines a wrapper around HIPBlas for doing graphics card linear
 * algebra. All matrix arguments are assumed to not be stored or mapped onto the
 * graphics hardware. These are designed to be drop-in replacements for the
 * other backends, so all memory transfer is done by the wrapper functions.
 */
#pragma once

// #include "einsums.hpp"
#include "einsums/_Export.hpp"

#include <complex>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

namespace einsums::backend::linear_algebra::hipblas {

namespace detail {

/**
 * @brief Convert a transpose character to an hipBLAS operation enum value.
 *
 * @param trans A transpose character. These are 'N', 'n', 'T', 't', 'C', and 'c'.
 *
 * @return The converted enum value. Defaults to HIPBLAS_OP_N if an invalid character is passed.
 */
__host__ __device__ EINSUMS_EXPORT hipblasOperation_t hipblas_char_to_op(char trans);

/**
 * @brief Convert a transpose character to an hipSolver operation enum value.
 *
 * @param trans A transpose character. These are 'N', 'n', 'T', 't', 'C', and 'c'.
 *
 * @return The converted enum value. Defaults to HIPSOLVER_OP_N if an invalid character is passed.
 */
__host__ __device__ EINSUMS_EXPORT hipsolverOperation_t hipsolver_char_to_op(char trans);

/**
 * @brief Convert a job character to an hipSolver job enum value.
 *
 * @param job The job character. This can be 'N', 'n', 'V', or 'v'.
 *
 * @return The converted enum value. Defaults to HIPSOLVER_EIG_MODE_NOVECTOR if an invalid character is passed.
 */
__host__ __device__ EINSUMS_EXPORT hipsolverEigMode_t hipsolver_job(char job);

/**
 * @brief Convert a fill character to an hipSolver fill mode enum value.
 *
 * @param fill The fill character. This can be 'U', 'u', 'L', or 'l'.
 *
 * @return The converted enum value. Defaults to HIPSOLVER_FILL_MODE_UPPER if an invalid character is passed.
 */
__host__ __device__ EINSUMS_EXPORT hipsolverFillMode_t hipsolver_fill(char fill);
}

/**
 * @brief Performs matrix multiplication between two general matrices.
 */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc);
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc);
void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);

/**
 * @brief Performs general matrix-vector multiplication.
 */
void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy);
void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);
void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy);
void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy);

/**
 * @brief Performs symmetric matrix diagonalization.
 */
auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int;
auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;

/**
 * @brief Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *ignored)
    -> int;
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *ignored)
    -> int;

/**
 * @brief Computes the solution to system of linear equations A * x = B for general matrices.
 */
auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int;
auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int;
auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int;
auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int;

/**
 * @brief Scale a vector by a scalar.
 */
void sscal(int n, float alpha, float *vec, int inc);
void dscal(int n, double alpha, double *vec, int inc);
void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc);
void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc);

/**
 * @brief Take the dot product of two vectors.
 */
auto sdot(int n, const float *x, int incx, const float *y, int incy) -> float;
auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double;
auto cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float>;
auto zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double>;

/**
 * Compute y = alpha * x + y
 */
void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y);
void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);
void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y);
void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y);

/**
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
void sger(int m, int n, float alpha, const float *x, int inc_x, const float *y, int inc_y, float *a, int lda);
void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);
void cger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int inc_x, const std::complex<float> *y, int inc_y,
          std::complex<float> *a, int lda);
void zger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int inc_x, const std::complex<double> *y, int inc_y,
          std::complex<double> *a, int lda);

/**
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
int sgetrf(int m, int n, float *a, int lda, int *ipiv);
int dgetrf(int m, int n, double *a, int lda, int *ipiv);
int cgetrf(int m, int n, std::complex<float> *a, int lda, int *ipiv);
int zgetrf(int m, int n, std::complex<double> *a, int lda, int *ipiv);

/**
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(int, float *, int, const int *, float *, int) -> int;
auto dgetri(int, double *, int, const int *, double *, int) -> int;
auto cgetri(int, std::complex<float> *, int, const int *, std::complex<float> *, int) -> int;
auto zgetri(int, std::complex<double> *, int, const int *, std::complex<double> *, int) -> int;

/**
 * Computes the singular value decomposition.
 */
auto sgesvd(char, char, int, int, float *, int, float *, float *, int, float *, int, float *) -> int;
auto dgesvd(char, char, int, int, double *, int, double *, double *, int, double *, int, double *) -> int;
auto cgesvd(char, char, int, int, std::complex<float> *, int, std::complex<float> *, std::complex<float> *, int, std::complex<float> *, int,
            std::complex<float> *) -> int;
auto zgesvd(char, char, int, int, std::complex<double> *, int, std::complex<double> *, std::complex<double> *, int, std::complex<double> *,
            int, std::complex<double> *) -> int;

/**
 * Computes the QR factorization.
 */
auto sgeqrf(int m, int n, float *a, int lda, float *tau) -> int;
auto dgeqrf(int m, int n, double *a, int lda, double *tau) -> int;
auto cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int;
auto zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int;

/**
 * Generates the Q matrix found by Xgeqrf.
 */
auto sorgqr(int m, int n, int k, float *a, int lda, const float *tau) -> int;
auto dorgqr(int m, int n, int k, double *a, int lda, const double *tau) -> int;
auto cungqr(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int;
auto zungqr(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int;

} // namespace einsums::backend::linear_algebra::hipblas
