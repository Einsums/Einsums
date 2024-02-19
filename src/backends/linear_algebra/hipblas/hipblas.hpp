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
 * @struct hipblas_exception
 * @brief Wraps hipBLAS status codes into an exception.
 *
 * Wraps hipBLAS status codes so that they can be thrown and caught. There is one code for each named hipBLAS status code.
 *
 * @tparam error The status code handled by this exception.
 */
template <hipblasStatus_t error>
struct hipblas_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * @brief Construct a new hipblas_exception.
     */
    hipblas_exception(const char *diagnostic) : message{""} {
        message += diagnostic;
        message += hipblasStatusToString(error);
    }

    /**
     * @brief Return the error string corresponding to the status code.
     */
    const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     */
    template <hipblasStatus_t other_error>
    bool operator==(const hipblas_exception<other_error> &other) const {
        return error == other_error;
    }

    /**
     * @brief Equality operator.
     */
    bool operator==(hipblasStatus_t other) const { return error == other; }

    /**
     * @brief Inequality operator.
     */
    template <hipblasStatus_t other_error>
    bool operator!=(const hipblas_exception<other_error> &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(hipblasStatus_t other) const { return error != other; }

    friend bool operator==(hipblasStatus_t, const hipblas_exception<error> &);
    friend bool operator!=(hipblasStatus_t, const hipblas_exception<error> &);
};

/**
 * @brief Reverse equality operator.
 */
template <hipblasStatus_t error>
bool operator==(hipblasStatus_t first, const hipblas_exception<error> &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 */
template <hipblasStatus_t error>
bool operator!=(hipblasStatus_t first, const hipblas_exception<error> &second) {
    return first != error;
}

// Put the status code documentaion in a different header for cleaner code.
#include "hipblas_status_doc.hpp"

using blasSuccess         = hipblas_exception<HIPBLAS_STATUS_SUCCESS>;
using blasNotInitialized  = hipblas_exception<HIPBLAS_STATUS_NOT_INITIALIZED>;
using blasAllocFailed     = hipblas_exception<HIPBLAS_STATUS_ALLOC_FAILED>;
using blasInvalidValue    = hipblas_exception<HIPBLAS_STATUS_INVALID_VALUE>;
using blasMappingError    = hipblas_exception<HIPBLAS_STATUS_MAPPING_ERROR>;
using blasExecutionFailed = hipblas_exception<HIPBLAS_STATUS_EXECUTION_FAILED>;
using blasInternalError   = hipblas_exception<HIPBLAS_STATUS_INTERNAL_ERROR>;
using blasNotSupported    = hipblas_exception<HIPBLAS_STATUS_NOT_SUPPORTED>;
using blasArchMismatch    = hipblas_exception<HIPBLAS_STATUS_ARCH_MISMATCH>;
using blasHandleIsNullptr = hipblas_exception<HIPBLAS_STATUS_HANDLE_IS_NULLPTR>;
using blasInvalidEnum     = hipblas_exception<HIPBLAS_STATUS_INVALID_ENUM>;
using blasUnknown         = hipblas_exception<HIPBLAS_STATUS_UNKNOWN>;

/**
 * @brief Create a string representation of an hipsolverStatus_t value.
 * Create a string representation of an hipsolverStatus_t value. There is no
 * equivalent function in hipSolver at this point in time, so a custom one
 * had to be made.
 *
 * @param status The status code to convert.
 *
 * @return A pointer to a string containing a brief message detailing the status.
 */
EINSUMS_EXPORT const char *hipsolverStatusToString(hipsolverStatus_t status);

/**
 * @struct hipsolver_exception
 *
 * @brief Wraps hipSolver status codes into an exception.
 *
 * Wraps hipSolver status codes into an exception which allows them to be thrown and caught.
 *
 * @tparam error The status code wrapped by the object.
 */
template <hipsolverStatus_t error>
struct hipsolver_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * Construct a new exception.
     */
    hipsolver_exception(const char *diagnostic) : message{""} {
        message += diagnostic;
        message += hipsolverStatusToString(error);
    }

    /**
     * Return the error string.
     */
    const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * Equality operator.
     */
    template <hipsolverStatus_t other_error>
    bool operator==(const hipsolver_exception<other_error> &other) const {
        return error == other_error;
    }

    /**
     * Equality operator.
     */
    bool operator==(hipsolverStatus_t other) const { return error == other; }

    /**
     * Inquality operator.
     */
    template <hipsolverStatus_t other_error>
    bool operator!=(const hipsolver_exception<other_error> &other) const {
        return error != other_error;
    }

    /**
     * Inquality operator.
     */
    bool operator!=(hipsolverStatus_t other) const { return error != other; }

    friend bool operator==(hipsolverStatus_t, const hipsolver_exception<error> &);
    friend bool operator!=(hipsolverStatus_t, const hipsolver_exception<error> &);
};

/**
 * Reverse equality operator.
 */
template <hipsolverStatus_t error>
bool operator==(hipsolverStatus_t first, const hipsolver_exception<error> &second) {
    return first == error;
}

/**
 * Reverse inequality operator.
 */
template <hipsolverStatus_t error>
bool operator!=(hipsolverStatus_t first, const hipsolver_exception<error> &second) {
    return first != error;
}

using solverSuccess          = hipsolver_exception<HIPSOLVER_STATUS_SUCCESS>;
using solverNotInitialized   = hipsolver_exception<HIPSOLVER_STATUS_NOT_INITIALIZED>;
using solverAllocFailed      = hipsolver_exception<HIPSOLVER_STATUS_ALLOC_FAILED>;
using solverInvalidValue     = hipsolver_exception<HIPSOLVER_STATUS_INVALID_VALUE>;
using solverMappingError     = hipsolver_exception<HIPSOLVER_STATUS_MAPPING_ERROR>;
using solverExecutionFailed  = hipsolver_exception<HIPSOLVER_STATUS_EXECUTION_FAILED>;
using solverInternalError    = hipsolver_exception<HIPSOLVER_STATUS_INTERNAL_ERROR>;
using solverFuncNotSupported = hipsolver_exception<HIPSOLVER_STATUS_NOT_SUPPORTED>;
using solverArchMismatch     = hipsolver_exception<HIPSOLVER_STATUS_ARCH_MISMATCH>;
using solverHandleIsNullptr  = hipsolver_exception<HIPSOLVER_STATUS_HANDLE_IS_NULLPTR>;
using solverInvalidEnum      = hipsolver_exception<HIPSOLVER_STATUS_INVALID_ENUM>;
using solverUnknown          = hipsolver_exception<HIPSOLVER_STATUS_UNKNOWN>;
// using solverZeroPivot = hipsolver_exception<HIPSOLVER_STATUS_ZERO_PIVOT>;

// Get the handles used internally. Can be used by other blas and solver processes.
/**
 * @brief Get the internal hipBLAS handle.
 *
 * @return The current internal hipBLAS handle.
 */
EINSUMS_EXPORT hipblasHandle_t get_blas_handle();

/**
 * @brief Get the internal hipSolver handle.
 *
 * @return The current internal hipSolver handle.
 */
EINSUMS_EXPORT hipsolverHandle_t get_solver_handle();

// Set the handles used internally. Useful to avoid creating multiple contexts.
/**
 * @brief Set the internal hipBLAS handle.
 *
 * @param value The new handle.
 *
 * @return The new handle.
 */
EINSUMS_EXPORT hipblasHandle_t set_blas_handle(hipblasHandle_t value);

/**
 * @brief Set the internal hipSolver handle.
 *
 * @param value The new handle.
 *
 * @return The new handle.
 */
EINSUMS_EXPORT hipsolverHandle_t set_solver_handle(hipsolverHandle_t value);

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

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 */
__host__ EINSUMS_EXPORT void __hipblas_catch__(hipblasStatus_t status, const char *diagnostic, bool throw_success = false);

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 */
__host__ EINSUMS_EXPORT void __hipsolver_catch__(hipsolverStatus_t status, const char *diagnostic, bool throw_success = false);

#define hipblas_catch_STR1(x) #x
#define hipblas_catch_STR(x)  hipblas_catch_STR1(x)
#define hipblas_catch(condition, ...)                                                                                                      \
    __hipblas_catch__((condition), __FILE__ ":" hipblas_catch_STR(__LINE__) ": " __VA_OPT__(, ) __VA_ARGS__)

#define hipsolver_catch_STR1(x) #x
#define hipsolver_catch_STR(x)  hipblas_catch_STR1(x)
#define hipsolver_catch(condition, ...)                                                                                                    \
    __hipsolver_catch__((condition), __FILE__ ":" hipsolver_catch_STR(__LINE__) ": " __VA_OPT__(, ) __VA_ARGS__)
} // namespace detail

/**
 * @brief Initialize the hipBLAS and hipSolver environment.
 */
void initialize();

/**
 * @brief Clean up and finalize the hipBLAS and hipSolver environment.
 */
void finalize();

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
