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

#include <complex>
#include <hipblas/hipblas.h>

namespace einsums::backend::linear_algebra::hipblas {

namespace detail {

template <hipblasStatus_t error>
struct hipblas_exception : public std::exception {
  public:
    /**
     * Construct an empty exception which represents a success.
     */
    hipblas_exception() = default;

    /**
     * Return the error string.
     */
    const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return hipblasStatusToString(error); }

    /**
     * Equality operators.
     */
    template <hipblasStatus_t other_error>
    bool operator==(const hipblas_exception<other_error> &other) const {
        return error == other_error;
    }

    bool operator==(hipblasStatus_t other) const { return error == other; }

    template <hipblasStatus_t other_error>
    bool operator!=(const hipblas_exception<other_error> &other) const {
        return error != other_error;
    }

    bool operator!=(hipblasStatus_t other) const { return error != other; }

    friend bool operator==(hipblasStatus_t, const hipblas_exception<error> &);
    friend bool operator!=(hipblasStatus_t, const hipblas_exception<error> &);
};

template <hipblasStatus_t error>
bool operator==(hipblasStatus_t first, const hipblas_exception<error> &second) {
    return first == error;
}

template <hipblasStatus_t error>
bool operator!=(hipblasStatus_t first, const hipblas_exception<error> &second) {
    return first != error;
}

using Success         = hipblas_exception<HIPBLAS_STATUS_SUCCESS>;
using NotInitialized  = hipblas_exception<HIPBLAS_STATUS_NOT_INITIALIZED>;
using AllocFailed     = hipblas_exception<HIPBLAS_STATUS_ALLOC_FAILED>;
using InvalidValue    = hipblas_exception<HIPBLAS_STATUS_INVALID_VALUE>;
using MappingError    = hipblas_exception<HIPBLAS_STATUS_MAPPING_ERROR>;
using ExecutionFailed = hipblas_exception<HIPBLAS_STATUS_EXECUTION_FAILED>;
using InternalError   = hipblas_exception<HIPBLAS_STATUS_INTERNAL_ERROR>;
using NotSupported    = hipblas_exception<HIPBLAS_STATUS_NOT_SUPPORTED>;
using ArchMismatch    = hipblas_exception<HIPBLAS_STATUS_ARCH_MISMATCH>;
using HandleIsNullptr = hipblas_exception<HIPBLAS_STATUS_HANDLE_IS_NULLPTR>;
using InvalidEnum     = hipblas_exception<HIPBLAS_STATUS_INVALID_ENUM>;
using Unknown         = hipblas_exception<HIPBLAS_STATUS_UNKNOWN>;

EINSUMS_EXPORT hipblasHandle_t handle;

__host__ __device__ EINSUMS_EXPORT hipblasOperation_t char_to_op(char trans);

__host__ EINSUMS_EXPORT void hipblas_catch(hipblasStatus_t, bool throw_success = false);
} // namespace detail

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc);
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc);
void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy);
void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);
void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy);
void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int;
auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *rwork)
    -> int;
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rwork)
    -> int;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int;
auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int;
auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int;
auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int;

void sscal(int n, float alpha, float *vec, int inc);
void dscal(int n, double alpha, double *vec, int inc);
void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc);
void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc);

auto sdot(int n, const float *x, int incx, const float *y, int incy) -> float;
auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double;
auto cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float>;
auto zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double>;

void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y);
void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);
void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y);
void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y);

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
void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);

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
auto dgetrf(int, int, double *, int, int *) -> int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto dgetri(int, double *, int, const int *, double *, int) -> int;

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double;

auto sgesvd(char, char, int, int, float *, int, float *, float *, int, float *, int, float *) -> int;
auto dgesvd(char, char, int, int, double *, int, double *, double *, int, double *, int, double *) -> int;

auto sgesdd(char, int, int, float *, int, float *, float *, int, float *, int) -> int;
auto dgesdd(char, int, int, double *, int, double *, double *, int, double *, int) -> int;
auto cgesdd(char, int, int, std::complex<float> *, int, float *, std::complex<float> *, int, std::complex<float> *, int) -> int;
auto zgesdd(char, int, int, std::complex<double> *, int, double *, std::complex<double> *, int, std::complex<double> *, int) -> int;

auto sgees(char jobvs, int n, float *a, int lda, int *sdim, float *wr, float *wi, float *vs, int ldvs) -> int;
auto dgees(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int;

auto strsyl(char trana, char tranb, int isgn, int m, int n, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
            float *scale) -> int;
auto dtrsyl(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c, int ldc,
            double *scale) -> int;
auto ctrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
            std::complex<float> *c, int ldc, float *scale) -> int;
auto ztrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
            std::complex<double> *c, int ldc, double *scale) -> int;

auto sgeqrf(int m, int n, float *a, int lda, float *tau) -> int;
auto dgeqrf(int m, int n, double *a, int lda, double *tau) -> int;
auto cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int;
auto zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int;

auto sorgqr(int m, int n, int k, float *a, int lda, const float *tau) -> int;
auto dorgqr(int m, int n, int k, double *a, int lda, const double *tau) -> int;
auto cungqr(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int;
auto zungqr(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int;

} // namespace einsums::backend::linear_algebra::hipblas
